"""
tests/conftest.py — Pytest configuration and shared fixtures.

Sets up required environment variables for tests so that constants.py
doesn't raise EnvironmentError when run without real secrets.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Set all required environment variables BEFORE importing constants.py
# This must run before any project module that imports constants is loaded.
# ---------------------------------------------------------------------------

_TEST_ENV = {
    # cTrader (mock values — no real API calls in unit tests)
    "CTRADER_CLIENT_ID": "test_client_id",
    "CTRADER_CLIENT_SECRET": "test_client_secret",
    "CTRADER_ACCESS_TOKEN": "test_access_token",
    "CTRADER_REFRESH_TOKEN": "test_refresh_token",
    "CTRADER_ACCOUNT_ID": "12345678",
    "CTRADER_ACCOUNT_ID_ACC2": "87654321",
    "CTRADER_ENVIRONMENT": "DEMO",
    # AI providers (mock keys)
    "OPENAI_API_KEY": "sk-test-openai",
    "GEMINI_API_KEY": "AIza-test-gemini",
    "GROQ_API_KEY": "gsk_test_groq",
    "OPENROUTER_API_KEY": "sk-or-test",
    # LangSmith
    "LANGSMITH_API_KEY": "ls__test",
    "LANGSMITH_PROJECT": "cBocior_as_test",
    "LANGCHAIN_TRACING_V2": "false",
    # Redis — disabled in tests; in-memory fallback is used instead
    "REDIS_ENABLED": "false",
    "REDIS_URL": "redis://localhost:6379/0",
    "REDIS_CACHE_TTL_SECONDS": "3600",
    # Account capitals
    "INITIAL_CAPITAL_ACC1": "10000",
    "INITIAL_CAPITAL_ACC2": "50",
    # Overrides
    "LOG_LEVEL": "WARNING",
}

for key, value in _TEST_ENV.items():
    os.environ.setdefault(key, value)
