# -*- coding: utf-8 -*-
"""
智能分析系统（股票） - 股票市场数据分析系统
修改：熊猫大侠
版本：v2.1.0
"""
# us_stock_service.py
import akshare as ak
import pandas as pd
import logging
from datetime import datetime, timedelta


class USStockService:
    def __init__(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.data_cache = {}

    def search_us_stocks(self, keyword):
        """
        搜索美股代码
        :param keyword: 搜索关键词
        :return: 匹配的股票列表
        """
        try:
            # 获取美股数据
            df = ak.stock_us_spot_em()

            # 转换列名
            df = df.rename(columns={
                "序号": "index",
                "名称": "name",
                "最新价": "price",
                "涨跌额": "price_change",
                "涨跌幅": "price_change_percent",
                "开盘价": "open",
                "最高价": "high",
                "最低价": "low",
                "昨收价": "pre_close",
                "总市值": "market_value",
                "市盈率": "pe_ratio",
                "成交量": "volume",
                "成交额": "turnover",
                "振幅": "amplitude",
                "换手率": "turnover_rate",
                "代码": "symbol"
            })

            # 模糊匹配搜索
            mask = df['name'].str.contains(keyword, case=False, na=False)
            results = df[mask]

            # 格式化返回结果并处理 NaN 值
            formatted_results = []
            for _, row in results.iterrows():
                formatted_results.append({
                    'name': row['name'] if pd.notna(row['name']) else '',
                    'symbol': str(row['symbol']) if pd.notna(row['symbol']) else '',
                    'price': float(row['price']) if pd.notna(row['price']) else 0.0,
                    'market_value': float(row['market_value']) if pd.notna(row['market_value']) else 0.0
                })

            return formatted_results

        except Exception as e:
            self.logger.error(f"搜索美股代码时出错: {str(e)}")
            raise Exception(f"搜索美股代码失败: {str(e)}")
    
    def get_us_stock_data(self, stock_code, start_date=None, end_date=None):
        """
        获取美股历史数据
        :param stock_code: 美股代码 (例如: AAPL, MSFT)
        :param start_date: 开始日期 (格式: YYYYMMDD)
        :param end_date: 结束日期 (格式: YYYYMMDD)
        :return: 包含历史数据的DataFrame
        """
        try:
            self.logger.info(f"获取美股 {stock_code} 历史数据")
            
            # 缓存键
            cache_key = f"{stock_code}_{start_date}_{end_date}_us_price"
            if cache_key in self.data_cache:
                self.logger.info(f"使用缓存数据: {cache_key}")
                return self.data_cache[cache_key].copy()

            # 默认日期设置
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')

            # 使用akshare获取美股历史数据
            # 检查是否需要处理股票代码格式
            if not stock_code.startswith('105.'):
                # 如果不是东方财富格式的代码，尝试直接使用stock_us_daily
                try:
                    df = ak.stock_us_daily(symbol=stock_code, adjust="qfq")
                    self.logger.info(f"使用stock_us_daily获取数据成功: {len(df)}行")
                except Exception as e:
                    self.logger.warning(f"使用stock_us_daily获取数据失败: {str(e)}, 尝试使用stock_us_hist")
                    # 如果直接使用stock_code失败，尝试添加东方财富格式的前缀
                    stock_code_em = f"105.{stock_code}"
                    df = ak.stock_us_hist(
                        symbol=stock_code_em,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq"
                    )
            else:
                # 已经是东方财富格式的代码
                df = ak.stock_us_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                self.logger.info(f"使用stock_us_hist获取数据成功: {len(df)}行")

            # 检查数据格式并标准化
            if '日期' in df.columns:
                # 东方财富格式数据处理
                df = df.rename(columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount"
                })
            elif 'date' in df.columns and 'open' in df.columns:
                # stock_us_daily 已经有标准列名的情况
                pass
            else:
                # 未知格式，尝试通用映射
                column_mapping = {}
                for col in df.columns:
                    if '日期' in col or 'date' in col.lower():
                        column_mapping[col] = 'date'
                    elif '开' in col or 'open' in col.lower():
                        column_mapping[col] = 'open'
                    elif '收' in col or 'close' in col.lower():
                        column_mapping[col] = 'close'
                    elif '高' in col or 'high' in col.lower():
                        column_mapping[col] = 'high'
                    elif '低' in col or 'low' in col.lower():
                        column_mapping[col] = 'low'
                    elif '量' in col or 'volume' in col.lower():
                        column_mapping[col] = 'volume'
                    elif '额' in col or 'amount' in col.lower() or 'turnover' in col.lower():
                        column_mapping[col] = 'amount'
                
                df = df.rename(columns=column_mapping)

            # 确保关键列存在
            required_columns = ['date', 'open', 'close', 'high', 'low', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"数据缺少关键列: {col}")
                    raise ValueError(f"数据格式错误: 缺少{col}列")

            # 确保日期格式正确
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])

            # 数据类型转换
            numeric_columns = ['open', 'close', 'high', 'low', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 删除空值
            df = df.dropna(subset=['date', 'close'])

            # 按日期排序
            df = df.sort_values('date')

            # 缓存数据
            self.data_cache[cache_key] = df.copy()

            return df

        except Exception as e:
            self.logger.error(f"获取美股数据失败: {str(e)}")
            raise Exception(f"获取美股数据失败: {str(e)}")