import json
import time

import requests

from config import XUEQIU_COOKIE


class XueQiuStock:
    """
    雪球股票数据
    """

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.",
            "cookie": XUEQIU_COOKIE
        }

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
        elif market_type == "HK":
            url += "&market=HK&type=hk"
        else:
            return []

        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()  # 确保请求成功
            data_json = response.json()
            if data_json.get('error_code', 400) == 0:
                return data_json.get('data', {}).get('list', [])
        except Exception as e:
            print(f"获取或解析数据时发生错误: {e}")

        return []

    def kline(self, symbol, days, period='day'):
        """
        获取K线数据
        """
        try:

            # 检查days类型是否是字符串
            if isinstance(days, str):
                # 解析 2day  5year 转为数字
                if days.endswith('d'):
                    days = int(days.replace('d', ''))
                elif days.endswith('y'):
                    days = int(days.replace('y', '')) * 365
                elif days.endswith('m'):
                    days = int(days.replace('m', '')) * 30
                else:
                    days = 100

            url = f"https://stock.xueqiu.com/v5/stock/chart/kline.json?symbol={symbol}&begin={int(time.time() * 1000)}&period={period}&type=before&count=-{days}&indicator=kline,pe,pb,ps,pcf,market_capital,agt,ggt,balance"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()  # 确保请求成功
            data_json = response.json()
            if data_json.get('error_code', 400) == 0:
                return data_json.get('data', {})
        except Exception as e:
            print(f"获取或解析数据时发生错误: {e}")

        return None


if __name__ == '__main__':
    service = XueQiuStock()
    print(json.dumps(service.kline('SZ300065', days=5)))
