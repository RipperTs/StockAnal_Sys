from config import BOCHA_API_KEY
import requests
import json


class BoChaNews:
    """
    博查联网搜索服务
    """

    def get_news(self, query: str, freshness: str = 'noLimit', page: int = 1, count: int = 10):
        """
        获取联网搜索结果数据
        """
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

        return response.json()


if __name__ == '__main__':
    news = BoChaNews()
    result = news.get_news("特斯拉")
    print(result)
