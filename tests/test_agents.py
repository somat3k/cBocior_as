"""
tests/test_agents.py -- Unit tests for the Groq agent and orchestrator.

Mocks external API calls to verify the correct payload schema and fallback
behaviour.  (E8 + E9)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base_agent import AgentError, BaseAgent
from src.agents.orchestrator import AgentOrchestrator
from src.utils.payload import TradingAction, TradingPayload

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_payload(
    action: TradingAction = TradingAction.HOLD,
    confidence: float = 0.75,
    source: str = "test",
) -> TradingPayload:
    return TradingPayload(
        symbol="EURUSD",
        action=action,
        confidence=confidence,
        source=source,
    )


def _mock_llm_response(action: str, confidence: float, reasoning: str = "ok") -> str:
    import json
    return json.dumps({"action": action, "confidence": confidence, "reasoning": reasoning})


# ---------------------------------------------------------------------------
# BaseAgent — retry + timeout + degraded
# ---------------------------------------------------------------------------

class _SometimesFailAgent(BaseAgent):
    """Fails for the first N calls, then succeeds."""

    agent_id: str = "sometimes_fail"

    def __init__(self, fail_times: int = 1) -> None:
        super().__init__(timeout=5)
        self._fail_times = fail_times
        self._call_count = 0

    async def _call(self, payload: TradingPayload) -> TradingPayload:
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise RuntimeError("Simulated failure")
        return payload.model_copy(update={"source": self.agent_id, "confidence": 0.8})


class TestBaseAgentRetry:
    def test_retries_on_transient_failure(self) -> None:
        """Agent should succeed after one transient failure."""
        agent = _SometimesFailAgent(fail_times=1)
        payload = _make_payload()
        with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            result = asyncio.run(agent.analyse(payload))
        assert result.source == "sometimes_fail"
        assert agent._call_count == 2  # 1 fail + 1 success

    def test_marked_degraded_after_max_retries(self) -> None:
        """Agent must be marked degraded after MAX_RETRIES failed analyse() calls."""
        from src.agents.base_agent import MAX_RETRIES

        agent = _SometimesFailAgent(fail_times=9999)  # always fails
        with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            for _ in range(MAX_RETRIES):
                with pytest.raises(AgentError):
                    asyncio.run(agent.analyse(_make_payload()))
        assert agent.degraded

    def test_degraded_reset_on_success(self) -> None:
        """A degraded agent should clear its degraded flag on next success."""
        from src.agents.base_agent import MAX_RETRIES

        agent = _SometimesFailAgent(fail_times=9999)
        with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            for _ in range(MAX_RETRIES):
                with pytest.raises(AgentError):
                    asyncio.run(agent.analyse(_make_payload()))
        assert agent.degraded

        # Now allow success
        agent._fail_times = 0
        result = asyncio.run(agent.analyse(_make_payload()))
        assert not agent.degraded
        assert result is not None

    def test_timeout_counts_as_failure(self) -> None:
        """Async timeout on _call should raise AgentError."""
        class _SlowAgent(BaseAgent):
            agent_id: str = "slow"

            async def _call(self, payload: TradingPayload) -> TradingPayload:
                # Use real asyncio.sleep so wait_for actually times out
                import asyncio as _asyncio
                await _asyncio.sleep(100)
                return payload  # pragma: no cover

        agent = _SlowAgent(timeout=0.01)
        # Do NOT mock asyncio.sleep here — we need the real one for wait_for to fire
        with pytest.raises(AgentError):
            asyncio.run(agent.analyse(_make_payload()))


# ---------------------------------------------------------------------------
# BaseAgent — _parse_llm_response
# ---------------------------------------------------------------------------

class TestParseLlmResponse:
    def _make_base_agent(self) -> BaseAgent:
        class _Dummy(BaseAgent):
            agent_id = "dummy"
            async def _call(self, p: TradingPayload) -> TradingPayload: ...
        return _Dummy()

    def test_parses_valid_json(self) -> None:
        agent = self._make_base_agent()
        raw = _mock_llm_response("BUY", 0.8)
        result = agent._parse_llm_response(raw, _make_payload())
        assert result.action == TradingAction.BUY
        assert result.confidence == pytest.approx(0.8)

    def test_parses_json_in_markdown_fence(self) -> None:
        agent = self._make_base_agent()
        raw = "```json\n" + _mock_llm_response("SELL", 0.7) + "\n```"
        result = agent._parse_llm_response(raw, _make_payload())
        assert result.action == TradingAction.SELL

    def test_defaults_to_hold_on_invalid_json(self) -> None:
        agent = self._make_base_agent()
        result = agent._parse_llm_response("NOT JSON {{{", _make_payload())
        assert result.action == TradingAction.HOLD
        assert result.confidence == pytest.approx(0.0)

    def test_clamps_confidence_to_01(self) -> None:
        agent = self._make_base_agent()
        raw = _mock_llm_response("BUY", 5.0)
        result = agent._parse_llm_response(raw, _make_payload())
        assert result.confidence <= 1.0

    def test_unknown_action_defaults_to_hold(self) -> None:
        agent = self._make_base_agent()
        import json
        raw = json.dumps({"action": "SIDEWAYS", "confidence": 0.7, "reasoning": "x"})
        result = agent._parse_llm_response(raw, _make_payload())
        assert result.action == TradingAction.HOLD

    def test_source_set_to_agent_id(self) -> None:
        agent = self._make_base_agent()
        raw = _mock_llm_response("HOLD", 0.5)
        result = agent._parse_llm_response(raw, _make_payload())
        assert result.source == "dummy"


# ---------------------------------------------------------------------------
# Individual agent mock tests  (E11)
# ---------------------------------------------------------------------------

class TestGroqAgent:
    def test_returns_sell_signal_from_mock(self) -> None:
        from src.agents.groq_agent import GroqAgent

        mock_message = MagicMock()
        mock_message.content = _mock_llm_response("SELL", 0.72)
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("src.agents.groq_agent.AsyncGroq") as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            agent = GroqAgent()
            result = asyncio.run(agent._call(_make_payload()))
        assert result.action == TradingAction.SELL
        assert result.source == "groq"

    def test_empty_model_list_raises(self) -> None:
        from src.agents.groq_agent import GroqAgent

        with (
            patch("src.agents.groq_agent.GROQ_MODEL", "   "),
            patch("src.agents.groq_agent.AsyncGroq") as MockGroq,
        ):
            MockGroq.return_value = MagicMock()
            agent = GroqAgent()
            with pytest.raises(RuntimeError, match="No Groq models configured"):
                asyncio.run(agent._call(_make_payload()))

    def test_falls_back_to_next_model(self) -> None:
        from groq import GroqError
        from src.agents.groq_agent import GroqAgent

        mock_message = MagicMock()
        mock_message.content = _mock_llm_response("BUY", 0.71)
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with (
            patch("src.agents.groq_agent.GROQ_MODEL", "oss-120b,backup-model"),
            patch("src.agents.groq_agent.AsyncGroq") as MockGroq,
        ):
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[GroqError("bad model"), mock_response]
            )
            agent = GroqAgent()
            result = asyncio.run(agent._call(_make_payload()))

        assert result.action == TradingAction.BUY
        assert (
            mock_client.chat.completions.create.call_args_list[0].kwargs["model"]
            == "oss-120b"
        )
        assert (
            mock_client.chat.completions.create.call_args_list[1].kwargs["model"]
            == "backup-model"
        )

    def test_raises_when_all_models_fail(self) -> None:
        from groq import GroqError
        from src.agents.groq_agent import GroqAgent

        with (
            patch("src.agents.groq_agent.GROQ_MODEL", "oss-120b,backup-model"),
            patch("src.agents.groq_agent.AsyncGroq") as MockGroq,
        ):
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            first_error = GroqError("bad model")
            last_error = GroqError("down")
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[first_error, last_error]
            )
            agent = GroqAgent()
            with pytest.raises(GroqError) as excinfo:
                asyncio.run(agent._call(_make_payload()))
            assert str(excinfo.value) == "down"


# ---------------------------------------------------------------------------
# Orchestrator Groq-only flow  (E9)
# ---------------------------------------------------------------------------

class TestAgentOrchestratorGroqOnly:
    def _make_orchestrator_with_mock(self) -> AgentOrchestrator:
        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        mock_agent = MagicMock()
        mock_agent.agent_id = "groq"
        mock_agent.degraded = False
        orch._groq = mock_agent
        return orch

    def test_returns_groq_payload(self) -> None:
        orch = self._make_orchestrator_with_mock()
        orch._groq.analyse = AsyncMock(
            return_value=_make_payload(action=TradingAction.BUY, source="groq")
        )
        result = asyncio.run(orch.run(_make_payload()))
        assert result.action == TradingAction.BUY
        assert result.source == "groq"

    def test_groq_failure_returns_hold(self) -> None:
        orch = self._make_orchestrator_with_mock()
        orch._groq.analyse = AsyncMock(side_effect=AgentError("down"))
        result = asyncio.run(orch.run(_make_payload()))
        assert result.action == TradingAction.HOLD
        assert result.source == "groq"
