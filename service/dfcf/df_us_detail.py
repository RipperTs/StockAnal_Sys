import pandas as pd
import requests


class DFUSDetail:
    """
    东方财富美股详情数据
    https://quote.eastmoney.com/us/KBSX.html
    """
    def __init__(self):
        pass

    def stock_us_individual_info_em(self,
            symbol: str = "105.KBSX", timeout: float = None
    ) -> pd.DataFrame:
        """
        东方财富-个股-股票信息
        https://quote.eastmoney.com/concept/sh603777.html?from=classic
        :param symbol: 股票代码
        :type symbol: str
        :param timeout: choice of None or a positive float number
        :type timeout: float
        :return: 股票信息
        :rtype: pandas.DataFrame
        """
        url = "https://push2.eastmoney.com/api/qt/stock/get"
        params = {
            "ut": "fa5fd1943c7b386f172d6893dbfba10b",
            "fltt": "2",
            "invt": "2",
            "fields": "f58,f107,f57,f43,f59,f169,f170,f152,f46,f60,f44,f45,f47,f48,f19,f532,f39,f161,f49,f171,f50,f86,f600,f601,f154,f84,f85,f168,f108,f116,f167,f164,f92,f71,f117,f292,f301",
            "secid": f"{symbol}",
            "_": "1742369391344",
        }
        r = requests.get(url, params=params, timeout=timeout)
        data_json = r.json()
        temp_df = pd.DataFrame(data_json)
        temp_df.reset_index(inplace=True)
        del temp_df["rc"]
        del temp_df["rt"]
        del temp_df["svr"]
        del temp_df["lt"]
        del temp_df["full"]
        code_name_map = {
            "f57": "股票代码",
            "f58": "股票简称",
            "f43": "最新价格",
            "f44": "最高价格",
            "f84": "总股本",
            "f85": "流通股",
            "f127": "行业",
            "f116": "总市值",
            "f117": "流通市值",
            "f189": "上市时间",
            "f171": "涨跌幅",
            "f164": "市盈率",
        }
        temp_df["index"] = temp_df["index"].map(code_name_map)
        temp_df = temp_df[pd.notna(temp_df["index"])]
        if "dlmkts" in temp_df.columns:
            del temp_df["dlmkts"]
        temp_df.columns = [
            "item",
            "value",
        ]
        temp_df.reset_index(inplace=True, drop=True)
        return temp_df


if __name__ == '__main__':
    # 测试查询
    df_us_detail = DFUSDetail()
    stock_us_individual_info_em_df = df_us_detail.stock_us_individual_info_em(symbol="105.KBSX")
    print(stock_us_individual_info_em_df)
