import redis
from typing import Any, Optional, Union
import json

from config import REDIS_URL


class RedisUtils:
    """
    Redis工具类，提供缓存设置和获取功能
    """

    def __init__(self, decode_responses: bool = True):
        """
        初始化Redis连接
        """
        self.redis_client = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=decode_responses
        )

    def set_cache(self, key: str, value: Any, expire_seconds: Optional[int] = None) -> bool:
        """
        设置缓存，支持设置过期时间
        
        Args:
            key: 缓存键
            value: 缓存值（会自动序列化）
            expire_seconds: 过期时间（秒），None表示永不过期
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 对复杂对象进行JSON序列化
            if not isinstance(value, (str, int, float, bool)):
                value = json.dumps(value)

            if expire_seconds is not None:
                return self.redis_client.setex(key, expire_seconds, value)
            else:
                return self.redis_client.set(key, value)
        except Exception as e:
            print(f"设置缓存失败: {e}")
            return False

    def get_cache(self, key: str, default: Any = None) -> Any:
        """
        获取缓存
        
        Args:
            key: 缓存键
            default: 如果缓存不存在，返回的默认值
            
        Returns:
            缓存值或默认值
        """
        try:
            value = self.redis_client.get(key)
            if value is None:
                return default

            # 尝试解析JSON
            try:
                return json.loads(value)
            except (TypeError, json.JSONDecodeError):
                # 如果不是JSON格式，直接返回
                return value
        except Exception as e:
            print(f"获取缓存失败: {e}")
            return default

    def delete_cache(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 操作是否成功
        """
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            print(f"删除缓存失败: {e}")
            return False

    def exists(self, key: str) -> bool:
        """
        检查键是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 键是否存在
        """
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            print(f"检查键是否存在失败: {e}")
            return False

    def get_ttl(self, key: str) -> int:
        """
        获取键的剩余生存时间（秒）
        
        Args:
            key: 缓存键
            
        Returns:
            int: 剩余生存时间（秒）
                -2表示键不存在
                -1表示键没有设置过期时间
        """
        try:
            return self.redis_client.ttl(key)
        except Exception as e:
            print(f"获取键的剩余生存时间失败: {e}")
            return -2

    def set_expire(self, key: str, expire_seconds: int) -> bool:
        """
        设置键的过期时间
        
        Args:
            key: 缓存键
            expire_seconds: 过期时间（秒）
            
        Returns:
            bool: 操作是否成功
        """
        try:
            return bool(self.redis_client.expire(key, expire_seconds))
        except Exception as e:
            print(f"设置键的过期时间失败: {e}")
            return False
