"""
cache.py — Simple TTL-based in-memory cache for the Retail Sentiment Engine.
"""
import time
from typing import Any, Optional


class TTLCache:
    """Thread-safe TTL cache with max entries eviction."""

    def __init__(self, ttl: int = 900, max_entries: int = 100):
        self._store: dict = {}
        self._ttl = ttl
        self._max_entries = max_entries

    def get(self, key: str) -> Optional[Any]:
        """Return cached value if not expired, else None."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """Store value with current timestamp. Evict oldest if over limit."""
        if len(self._store) >= self._max_entries and key not in self._store:
            oldest_key = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest_key]
        self._store[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all entries."""
        self._store.clear()


# Global cache instance
_cache = TTLCache()


def get_cached(key: str) -> Optional[Any]:
    return _cache.get(key)


def set_cached(key: str, value: Any) -> None:
    _cache.set(key, value)
