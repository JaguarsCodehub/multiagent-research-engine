import json
import redis.asyncio as redis
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import REDIS_URL

class SessionState:
    def __init__(self):
        # We try to initialize redis, but if it fails we might gracefully degrade or just let it raise
        try:
            self.redis = redis.from_url(REDIS_URL, decode_responses=True)
        except Exception as e:
            print(f"Warning: Failed to connect to Redis at {REDIS_URL}: {e}")
            self.redis = None
        
    async def set_state(self, session_id: str, key: str, value: dict):
        if not self.redis:
            print("Redis not available, skipping state save.")
            return
            
        state_key = f"session:{session_id}:{key}"
        try:
            await self.redis.set(state_key, json.dumps(value))
            # Expire session data after 24 hours to prevent memory bloat
            await self.redis.expire(state_key, 86400)
        except Exception as e:
            print(f"Failed to set state for {state_key}: {e}")
        
    async def get_state(self, session_id: str, key: str) -> dict:
        if not self.redis:
            return None
            
        state_key = f"session:{session_id}:{key}"
        try:
            data = await self.redis.get(state_key)
            if data:
                return json.loads(data)
        except Exception as e:
            print(f"Failed to get state for {state_key}: {e}")
        return None
        
    async def close(self):
        if self.redis:
            await self.redis.aclose()
