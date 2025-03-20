from config import BOCHA_API_KEY
import requests
import json

from utils.redis_utils import RedisUtils
from utils.string_utils import md5_encrypt


class BoChaNews:
    """
    博查联网搜索服务
    """

    def get_news(self, query: str, freshness: str = 'noLimit', page: int = 1, count: int = 10, is_cache: bool = True):
        """
        获取联网搜索结果数据
        """
        cache_key = "bocha_news:" + md5_encrypt(f"{query}:{freshness}:{page}:{count}")
        redis_utils = RedisUtils()
        if is_cache and redis_utils.exists(cache_key):
            return redis_utils.get_cache(cache_key)

        url = "https://api.bochaai.com/v1/web-search"
        headers = {
            'Authorization': f'Bearer {BOCHA_API_KEY}',
            'Content-Type': 'application/json'
        }

        payload = {
            "query": query,
            "page": page,
            "count": count,
            "freshness": freshness,
            "summary": True
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        if response.status_code != 200:
            return None

        if is_cache:
            redis_utils.set_cache(cache_key, response.json(), expire_seconds=60 * 60 * 8)

        return response.json()


if __name__ == '__main__':
    news = BoChaNews()
    result = news.get_news("请提供美股股票代码: VSCO 的最新相关新闻和所属行业信息以及官方公告")
    print(result)
