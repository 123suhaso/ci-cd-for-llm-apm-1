import os
import logging
from typing import Optional, Tuple

try:
    import redis.asyncio as aioredis
except Exception:
    aioredis = None

logger = logging.getLogger(__name__)

QUOTA_MAX_REQUESTS = int(os.getenv("QUOTA_MAX_REQUESTS", "100"))
QUOTA_WINDOW_SECONDS = int(os.getenv("QUOTA_WINDOW_SECONDS", "3600"))  
QUOTA_COOLDOWN_SECONDS = int(os.getenv("QUOTA_COOLDOWN_SECONDS", "3600"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_redis_client: Optional["aioredis.Redis"] = None

async def init_quota_redis(redis_client: Optional[aioredis.Redis] = None):
    global _redis_client
    if redis_client is not None:
        _redis_client = redis_client
        return _redis_client
    if _redis_client is None:
        if aioredis is None:
            raise RuntimeError("redis.asyncio not available. Install redis>=4.6.0")
        _redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client

async def close_quota_redis():
    global _redis_client
    try:
        if _redis_client:
            await _redis_client.close()
    except Exception:
        pass
    _redis_client = None

def _keys(user_id: str):
    return {
        "count": f"quota:{user_id}:count",
        "cooldown": f"quota:cooldown:{user_id}",
    }

async def is_in_cooldown(user_id: str, redis_client: Optional["aioredis.Redis"] = None) -> Tuple[bool, int]:
    client = redis_client or _redis_client
    if client is None:
        await init_quota_redis()
        client = _redis_client
    keys = _keys(user_id)
    ttl = await client.ttl(keys["cooldown"])
    if ttl is None or ttl == -2:
        return False, 0
    if ttl == -1:
        return True, QUOTA_COOLDOWN_SECONDS
    return True, int(ttl)

async def increment_and_check(user_id: str, redis_client: Optional[aioredis.Redis] = None, increment: int = 1) -> Tuple[bool, int]:
 
    client = redis_client or _redis_client
    if client is None:
        await init_quota_redis()
        client = _redis_client
    keys = _keys(user_id)

    lua = """
    local new = redis.call("INCRBY", KEYS[1], ARGV[1])
    if tonumber(redis.call("TTL", KEYS[1])) == -1 then
      -- if key exists and no TTL, set TTL
      redis.call("EXPIRE", KEYS[1], ARGV[2])
    elseif tonumber(redis.call("TTL", KEYS[1])) == -2 then
      -- key was just created, set TTL
      redis.call("EXPIRE", KEYS[1], ARGV[2])
    end
    return new
    """
    try:
        new_count = await client.eval(lua, 1, keys["count"], str(increment), str(QUOTA_WINDOW_SECONDS))
        new_count = int(new_count)
    except Exception:
        new_count = await client.incrby(keys["count"], increment)
        await client.expire(keys["count"], QUOTA_WINDOW_SECONDS)
        new_count = int(new_count)

    exceeded = new_count > QUOTA_MAX_REQUESTS
    return exceeded, new_count

async def set_cooldown(user_id: str, seconds: Optional[int] = None, redis_client: Optional[aioredis.Redis] = None):
    client = redis_client or _redis_client
    if client is None:
        await init_quota_redis()
        client = _redis_client
    keys = _keys(user_id)
    s = seconds if seconds is not None else QUOTA_COOLDOWN_SECONDS
    await client.set(keys["cooldown"], "1", ex=s)

async def reset_quota(user_id: str, redis_client: Optional[aioredis.Redis] = None):
    client = redis_client or _redis_client
    if client is None:
        await init_quota_redis()
        client = _redis_client
    keys = _keys(user_id)
    await client.delete(keys["count"], keys["cooldown"])
