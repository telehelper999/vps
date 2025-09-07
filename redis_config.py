#!/usr/bin/env python3
"""
Redis configuration and pub/sub setup for bonus code broadcasting
"""
import redis
import json
import asyncio
from typing import Dict, Any

class RedisManager:
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        import os
        # Use environment variable for password if not provided
        if password is None:
            password = os.getenv('REDIS_PASSWORD')
        
        self.redis_client = redis.Redis(
            host=host, 
            port=port, 
            db=db,
            password=password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )
        self.pubsub = self.redis_client.pubsub()
        
    def publish_code(self, code_data: Dict[str, Any]):
        """Publish bonus code to all subscribers"""
        try:
            message = json.dumps(code_data)
            result = self.redis_client.publish('bonus_codes', message)
            print(f"📡 Published code to {result} subscribers")
            return result
        except Exception as e:
            print(f"❌ Redis publish error: {e}")
            return 0
    
    def subscribe_to_codes(self):
        """Subscribe to bonus code channel"""
        self.pubsub.subscribe('bonus_codes')
        return self.pubsub
    
    def test_connection(self):
        """Test Redis connection"""
        try:
            self.redis_client.ping()
            print("✅ Redis connection successful")
            return True
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            return False
    
    def set_cache(self, key: str, value: str, expire: int = 300):
        """Set cache with expiration (default 5 minutes)"""
        return self.redis_client.setex(key, expire, value)
    
    def get_cache(self, key: str):
        """Get cached value"""
        return self.redis_client.get(key)

# Test Redis functionality
if __name__ == "__main__":
    redis_manager = RedisManager()
    
    # Test connection
    if redis_manager.test_connection():
        print("🚀 Redis is ready for pub/sub!")
        
        # Test publishing
        test_code = {
            "type": "code",
            "code": "TEST123",
            "source": "redis_test",
            "timestamp": "2025-09-07T05:15:00Z"
        }
        redis_manager.publish_code(test_code)
    else:
        print("⚠️ Redis not available - will work without caching")