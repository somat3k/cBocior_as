"""
tests/test_agents.py — Unit tests for AI agents and orchestrator.

Mocks all external API calls to verify the correct payload schema,
fallback behaviour, and majority-vote logic.  (E11 + E12)
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
            result = asyncio.get_event_loop().run_until_complete(agent.analyse(payload))
        assert result.source == "sometimes_fail"
        assert agent._call_count == 2  # 1 fail + 1 success

    def test_marked_degraded_after_max_retries(self) -> None:
        """Agent must be marked degraded after MAX_RETRIES failed analyse() calls."""
        from src.agents.base_agent import MAX_RETRIES

        agent = _SometimesFailAgent(fail_times=9999)  # always fails
        with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            for _ in range(MAX_RETRIES):
                with pytest.raises(AgentError):
                    asyncio.get_event_loop().run_until_complete(
                        agent.analyse(_make_payload())
                    )
        assert agent.degraded

    def test_degraded_reset_on_success(self) -> None:
        """A degraded agent should clear its degraded flag on next success."""
        from src.agents.base_agent import MAX_RETRIES

        agent = _SometimesFailAgent(fail_times=9999)
        with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            for _ in range(MAX_RETRIES):
                with pytest.raises(AgentError):
                    asyncio.get_event_loop().run_until_complete(
                        agent.analyse(_make_payload())
                    )
        assert agent.degraded

        # Now allow success
        agent._fail_times = 0
        result = asyncio.get_event_loop().run_until_complete(
            agent.analyse(_make_payload())
        )
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
            asyncio.get_event_loop().run_until_complete(
                agent.analyse(_make_payload())
            )


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

class TestOpenAIAgent:
    def test_returns_buy_signal_from_mock(self) -> None:
        from src.agents.openai_agent import OpenAIAgent

        mock_response = MagicMock()
        mock_response.choices[0].message.content = _mock_llm_response("BUY", 0.85)

        with patch("src.agents.openai_agent.AsyncOpenAI") as MockClient:
            mock_client_instance = MagicMock()
            MockClient.return_value = mock_client_instance
            mock_client_instance.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            agent = OpenAIAgent()
            result = asyncio.get_event_loop().run_until_complete(
                agent._call(_make_payload())
            )
        assert result.action == TradingAction.BUY
        assert result.source == "openai"

    def test_falls_back_to_hold_on_empty_response(self) -> None:
        from src.agents.openai_agent import OpenAIAgent

        mock_response = MagicMock()
        mock_response.choices[0].message.content = ""

        with patch("src.agents.openai_agent.AsyncOpenAI") as MockClient:
            mock_client_instance = MagicMock()
            MockClient.return_value = mock_client_instance
            mock_client_instance.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            agent = OpenAIAgent()
            result = asyncio.get_event_loop().run_until_complete(
                agent._call(_make_payload())
            )
        assert result.action == TradingAction.HOLD


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
            result = asyncio.get_event_loop().run_until_complete(
                agent._call(_make_payload())
            )
        assert result.action == TradingAction.SELL
        assert result.source == "groq"


class TestGeminiAgent:
    def test_returns_hold_signal_from_mock(self) -> None:
        from src.agents.gemini_agent import GeminiAgent

        mock_part = MagicMock()
        mock_part.text = _mock_llm_response("HOLD", 0.55)
        mock_response = MagicMock()
        mock_response.candidates[0].content.parts = [mock_part]

        with patch("src.agents.gemini_agent.genai") as MockGenai:
            mock_model = MagicMock()
            MockGenai.GenerativeModel.return_value = mock_model
            mock_model.generate_content_async = AsyncMock(return_value=mock_response)
            agent = GeminiAgent()
            result = asyncio.get_event_loop().run_until_complete(
                agent._call(_make_payload())
            )
        assert result.action == TradingAction.HOLD
        assert result.source == "gemini"


# ---------------------------------------------------------------------------
# Orchestrator majority vote  (E12)
# ---------------------------------------------------------------------------

class TestAgentOrchestratorMajority:
    def _make_orchestrator_with_mocks(
        self,
        groq_action: TradingAction,
        gemini_action: TradingAction,
        openai_action: TradingAction,
        openrouter_action: TradingAction,
    ) -> AgentOrchestrator:
        orch = AgentOrchestrator.__new__(AgentOrchestrator)

        def _agent_mock(action: TradingAction, source: str) -> MagicMock:
            m = MagicMock()
            m.agent_id = source
            m.degraded = False
            m.analyse = AsyncMock(
                return_value=_make_payload(action=action, confidence=0.75, source=source)
            )
            return m

        orch._groq = _agent_mock(groq_action, "groq")
        orch._gemini = _agent_mock(gemini_action, "gemini")
        orch._openai = _agent_mock(openai_action, "openai")
        orch._openrouter = _agent_mock(openrouter_action, "openrouter")
        return orch

    def test_unanimous_buy_produces_buy(self) -> None:
        orch = self._make_orchestrator_with_mocks(
            TradingAction.BUY, TradingAction.BUY, TradingAction.BUY, TradingAction.BUY
        )
        result = asyncio.get_event_loop().run_until_complete(
            orch.run(_make_payload())
        )
        assert result.action == TradingAction.BUY

    def test_majority_sell_produces_sell(self) -> None:
        orch = self._make_orchestrator_with_mocks(
            TradingAction.SELL, TradingAction.SELL, TradingAction.SELL, TradingAction.BUY
        )
        result = asyncio.get_event_loop().run_until_complete(
            orch.run(_make_payload())
        )
        assert result.action == TradingAction.SELL

    def test_tie_vote_produces_hold(self) -> None:
        """2 BUY vs 2 SELL → HOLD (no majority)."""
        orch = self._make_orchestrator_with_mocks(
            TradingAction.BUY, TradingAction.BUY,
            TradingAction.SELL, TradingAction.SELL,
        )
        result = asyncio.get_event_loop().run_until_complete(
            orch.run(_make_payload())
        )
        # 2 vs 2 is not ≥ 3 majority → HOLD
        assert result.action == TradingAction.HOLD

    def test_all_hold_produces_hold(self) -> None:
        orch = self._make_orchestrator_with_mocks(
            TradingAction.HOLD, TradingAction.HOLD,
            TradingAction.HOLD, TradingAction.HOLD,
        )
        result = asyncio.get_event_loop().run_until_complete(
            orch.run(_make_payload())
        )
        assert result.action == TradingAction.HOLD

    def test_one_agent_fails_still_majority(self) -> None:
        """If one agent raises, remaining majority should still fire."""
        orch = self._make_orchestrator_with_mocks(
            TradingAction.BUY, TradingAction.BUY,
            TradingAction.BUY, TradingAction.BUY,
        )
        # Make openrouter raise
        orch._openrouter.analyse = AsyncMock(side_effect=AgentError("down"))
        result = asyncio.get_event_loop().run_until_complete(
            orch.run(_make_payload())
        )
        # 3 agents BUY ≥ required majority threshold → BUY
        assert result.action == TradingAction.BUY

    def test_all_agents_fail_produces_hold(self) -> None:
        """When all agents fail the orchestrator must return HOLD."""
        orch = self._make_orchestrator_with_mocks(
            TradingAction.BUY, TradingAction.BUY,
            TradingAction.BUY, TradingAction.BUY,
        )
        for agent in (orch._groq, orch._gemini, orch._openai, orch._openrouter):
            agent.analyse = AsyncMock(side_effect=AgentError("down"))
        result = asyncio.get_event_loop().run_until_complete(
            orch.run(_make_payload())
        )
        assert result.action == TradingAction.HOLD
