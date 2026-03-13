"""
src/utils/cache.py — Redis-backed cache with in-memory fallback.

Every task that needs caching (data fetching, model results, indicator
snapshots) uses this module.  When Redis is available and REDIS_ENABLED is
``true``, data is stored in a Redis instance identified by REDIS_URL.  If
Redis is unreachable or disabled the module transparently falls back to a
process-local in-memory dict so the bot continues running without a cache
daemon.

Usage::

    cache = get_cache()
    cache.set("key", b"value", ttl=3600)
    raw = cache.get("key")          # bytes or None
    cache.delete("key")
    cache.clear_prefix("EURUSD:")   # remove all keys starting with prefix
"""

from __future__ import annotations

import threading
import time
from typing import Any

try:
    import ujson as json
except ImportError:
    import json  # type: ignore[no-redef]

from constants import REDIS_CACHE_TTL_SECONDS, REDIS_ENABLED, REDIS_URL
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# In-memory fallback cache
# ---------------------------------------------------------------------------

class _MemoryCache:
    """Thread-safe in-process TTL cache used when Redis is unavailable."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[bytes, float]] = {}  # key → (value, expires_at)
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: bytes, ttl: int = REDIS_CACHE_TTL_SECONDS) -> None:
        with self._lock:
            self._store[key] = (value, time.monotonic() + ttl)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear_prefix(self, prefix: str) -> int:
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]
            return len(keys)

    def ping(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Redis-backed cache
# ---------------------------------------------------------------------------

class _RedisCache:
    """Thin wrapper around redis.Redis with graceful error handling."""

    def __init__(self, url: str) -> None:
        import redis  # imported lazily to keep startup fast when unavailable

        self._client = redis.Redis.from_url(url, decode_responses=False)

    def get(self, key: str) -> bytes | None:
        try:
            return self._client.get(key)
        except Exception as exc:
            logger.warning("Redis GET failed, returning None", key=key, error=str(exc))
            return None

    def set(self, key: str, value: bytes, ttl: int = REDIS_CACHE_TTL_SECONDS) -> None:
        try:
            self._client.setex(key, ttl, value)
        except Exception as exc:
            logger.warning("Redis SET failed", key=key, error=str(exc))

    def delete(self, key: str) -> None:
        try:
            self._client.delete(key)
        except Exception as exc:
            logger.warning("Redis DELETE failed", key=key, error=str(exc))

    def clear_prefix(self, prefix: str) -> int:
        try:
            keys = list(self._client.scan_iter(f"{prefix}*"))
            if keys:
                self._client.delete(*keys)
            return len(keys)
        except Exception as exc:
            logger.warning("Redis CLEAR_PREFIX failed", prefix=prefix, error=str(exc))
            return 0

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Public Cache type alias
# ---------------------------------------------------------------------------

CacheBackend = _MemoryCache | _RedisCache


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def cache_set_json(
    cache: CacheBackend,
    key: str,
    obj: Any,
    ttl: int = REDIS_CACHE_TTL_SECONDS,
) -> None:
    """Serialise *obj* as JSON bytes and store in *cache*."""
    cache.set(key, json.dumps(obj).encode(), ttl=ttl)


def cache_get_json(cache: CacheBackend, key: str) -> Any:
    """Retrieve and deserialise a JSON value from *cache*, or ``None``."""
    raw = cache.get(key)
    if raw is None:
        return None
    try:
        # Decode bytes → str before parsing so both stdlib json and ujson work
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        return json.loads(text)
    except Exception as exc:
        logger.warning("Cache JSON decode failed", key=key, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_cache_instance: CacheBackend | None = None
_cache_lock = threading.Lock()


def get_cache() -> CacheBackend:
    """
    Return a process-singleton cache backend.

    Tries to connect to Redis on first call.  If Redis is disabled or
    unreachable the in-memory fallback is returned instead.
    """
    global _cache_instance  # noqa: PLW0603

    with _cache_lock:
        if _cache_instance is not None:
            return _cache_instance

        if not REDIS_ENABLED:
            logger.info("Redis disabled by config; using in-memory cache")
            _cache_instance = _MemoryCache()
            return _cache_instance

        try:
            rc = _RedisCache(REDIS_URL)
            if rc.ping():
                logger.info("Redis cache connected", url=REDIS_URL)
                _cache_instance = rc
            else:
                raise RuntimeError("Redis ping failed")
        except Exception as exc:
            logger.warning(
                "Redis unavailable; falling back to in-memory cache",
                error=str(exc),
            )
            _cache_instance = _MemoryCache()

        return _cache_instance


def reset_cache() -> None:
    """Reset the singleton (useful in tests)."""
    global _cache_instance  # noqa: PLW0603
    with _cache_lock:
        _cache_instance = None
