"""
tests/test_cache.py — Unit tests for src/utils/cache.py.

Redis is disabled via REDIS_ENABLED=false in conftest.py, so all tests
exercise the in-memory fallback.  The Redis path is tested with a mock.
"""

from __future__ import annotations

import time

import pytest

from src.utils.cache import (
    _MemoryCache,
    _RedisCache,
    cache_get_json,
    cache_set_json,
    get_cache,
    reset_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Ensure each test starts with a fresh cache singleton."""
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def mem() -> _MemoryCache:
    return _MemoryCache()


# ---------------------------------------------------------------------------
# _MemoryCache
# ---------------------------------------------------------------------------

class TestMemoryCache:
    def test_set_and_get(self, mem: _MemoryCache) -> None:
        mem.set("k", b"value")
        assert mem.get("k") == b"value"

    def test_missing_key_returns_none(self, mem: _MemoryCache) -> None:
        assert mem.get("nonexistent") is None

    def test_overwrite(self, mem: _MemoryCache) -> None:
        mem.set("k", b"v1")
        mem.set("k", b"v2")
        assert mem.get("k") == b"v2"

    def test_delete(self, mem: _MemoryCache) -> None:
        mem.set("k", b"v")
        mem.delete("k")
        assert mem.get("k") is None

    def test_delete_missing_is_noop(self, mem: _MemoryCache) -> None:
        mem.delete("nonexistent")  # should not raise

    def test_ttl_expiry(self, mem: _MemoryCache) -> None:
        mem.set("k", b"v", ttl=1)
        time.sleep(1.05)
        assert mem.get("k") is None

    def test_clear_prefix(self, mem: _MemoryCache) -> None:
        mem.set("abc:1", b"v")
        mem.set("abc:2", b"v")
        mem.set("xyz:1", b"v")
        removed = mem.clear_prefix("abc:")
        assert removed == 2
        assert mem.get("abc:1") is None
        assert mem.get("abc:2") is None
        assert mem.get("xyz:1") == b"v"

    def test_ping(self, mem: _MemoryCache) -> None:
        assert mem.ping() is True


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

class TestJsonHelpers:
    def test_set_and_get_dict(self, mem: _MemoryCache) -> None:
        obj = {"action": "BUY", "confidence": 0.85, "items": [1, 2, 3]}
        cache_set_json(mem, "payload", obj)
        result = cache_get_json(mem, "payload")
        assert result == obj

    def test_get_missing_returns_none(self, mem: _MemoryCache) -> None:
        assert cache_get_json(mem, "missing") is None

    def test_get_corrupt_returns_none(self, mem: _MemoryCache) -> None:
        mem.set("bad", b"not-json-{{{{")
        assert cache_get_json(mem, "bad") is None


# ---------------------------------------------------------------------------
# get_cache factory (in-memory path, REDIS_ENABLED=false in conftest)
# ---------------------------------------------------------------------------

class TestGetCacheFactory:
    def test_returns_memory_cache_when_redis_disabled(self) -> None:
        cache = get_cache()
        assert isinstance(cache, _MemoryCache)

    def test_singleton(self) -> None:
        c1 = get_cache()
        c2 = get_cache()
        assert c1 is c2


# ---------------------------------------------------------------------------
# _RedisCache error handling (using a mock to avoid real Redis)
# ---------------------------------------------------------------------------

class TestRedisCacheErrorHandling:
    def test_get_returns_none_on_exception(self, mocker: pytest.MonkeyPatch) -> None:
        import redis

        mocker.patch.object(
            redis.Redis, "from_url", return_value=mocker.MagicMock()
        )
        rc = _RedisCache("redis://localhost:6379/0")
        rc._client.get.side_effect = Exception("connection refused")  # type: ignore[attr-defined]
        assert rc.get("k") is None

    def test_set_silent_on_exception(self, mocker: pytest.MonkeyPatch) -> None:
        import redis

        mocker.patch.object(
            redis.Redis, "from_url", return_value=mocker.MagicMock()
        )
        rc = _RedisCache("redis://localhost:6379/0")
        rc._client.setex.side_effect = Exception("connection refused")  # type: ignore[attr-defined]
        rc.set("k", b"v")  # must not raise

    def test_ping_returns_false_on_exception(self, mocker: pytest.MonkeyPatch) -> None:
        import redis

        mocker.patch.object(
            redis.Redis, "from_url", return_value=mocker.MagicMock()
        )
        rc = _RedisCache("redis://localhost:6379/0")
        rc._client.ping.side_effect = Exception("connection refused")  # type: ignore[attr-defined]
        assert rc.ping() is False

    def test_get_cache_falls_back_to_memory_when_redis_ping_fails(
        self, mocker: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REDIS_ENABLED", "true")
        import constants
        mocker.patch.object(constants, "REDIS_ENABLED", True)

        import src.utils.cache as cache_mod
        mocker.patch.object(cache_mod, "REDIS_ENABLED", True)
        mocker.patch.object(
            cache_mod._RedisCache, "ping", return_value=False
        )
        c = cache_mod.get_cache()
        assert isinstance(c, cache_mod._MemoryCache)
