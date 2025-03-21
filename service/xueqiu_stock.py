import requests

from config import XUEQIU_COOKIE


class XueQiuStock:
    """
    雪球股票数据
    """

    def getStockList(self, page: int = 1, size: int = 90, market_type: str = "US") -> list:
        """
        从雪球获取所有股票信息
        https://finance.sina.com.cn/stock/usstock/sector.shtml
        page: 页码
        size: 每页数量, 20,40,60
        """
        url = f"https://stock.xueqiu.com/v5/stock/screener/quote/list.json?page={page}&size={size}&order=desc&order_by=percent"
        if market_type == "US":
            url += "&market=US&type=us"
        elif market_type == "A":
            url += "&market=CN&type=sh_sz"
        else:
            return []

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.",
            "cookie": XUEQIU_COOKIE
        }
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()  # 确保请求成功
            data_json = response.json()
            if data_json.get('error_code', 400) == 0:
                return data_json.get('data', {}).get('list', [])
        except Exception as e:
            print(f"获取或解析数据时发生错误: {e}")

        return []
