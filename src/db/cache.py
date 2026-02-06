"""Redis caching utilities."""

import hashlib
import json
import logging
from typing import Any

import redis

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize Redis client
redis_client = redis.from_url(settings.redis_url, decode_responses=True)

# Cache TTL in seconds (1 hour default)
DEFAULT_TTL = 3600


def get_cache_key(query: str) -> str:
    """Generate a cache key from a query string."""
    # Normalize query and hash it
    normalized = query.lower().strip()
    return f"research:{hashlib.sha256(normalized.encode()).hexdigest()[:16]}"


def get_cached_result(query: str) -> dict | None:
    """
    Get cached research result for a query.
    
    Returns None if not cached or cache miss.
    """
    try:
        key = get_cache_key(query)
        cached = redis_client.get(key)
        if cached:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return json.loads(cached)
        logger.debug(f"Cache miss for query: {query[:50]}...")
        return None
    except redis.RedisError as e:
        logger.warning(f"Redis error on get: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid cached JSON: {e}")
        return None


def set_cached_result(query: str, result: dict, ttl: int = DEFAULT_TTL) -> bool:
    """
    Cache a research result.
    
    Args:
        query: The original query string
        result: The research result to cache
        ttl: Time-to-live in seconds
        
    Returns:
        True if cached successfully, False otherwise
    """
    try:
        key = get_cache_key(query)
        # Convert result to JSON-serializable format
        serializable = _make_serializable(result)
        redis_client.setex(key, ttl, json.dumps(serializable))
        logger.info(f"Cached result for query: {query[:50]}...")
        return True
    except redis.RedisError as e:
        logger.warning(f"Redis error on set: {e}")
        return False
    except (TypeError, ValueError) as e:
        logger.warning(f"Serialization error: {e}")
        return False


def invalidate_cache(query: str) -> bool:
    """Invalidate cached result for a query."""
    try:
        key = get_cache_key(query)
        redis_client.delete(key)
        return True
    except redis.RedisError as e:
        logger.warning(f"Redis error on delete: {e}")
        return False


def check_redis_health() -> bool:
    """Check if Redis is healthy."""
    try:
        return redis_client.ping()
    except redis.RedisError:
        return False


def _make_serializable(obj: Any) -> Any:
    """Convert an object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        return _make_serializable(obj.__dict__)
    elif hasattr(obj, "value"):  # Enum
        return obj.value
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)
