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
            # 增加验证，确保不是A股科创板代码等
            if stock_code.startswith('688') or stock_code.startswith('300') or stock_code.startswith('301') or stock_code.startswith('003'):
                self.logger.error(f"股票代码 {stock_code} 不是美股代码，可能是A股代码")
                raise ValueError(f"股票代码 {stock_code} 不是美股代码，请使用正确的美股代码")
            
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

            # 处理不同格式的美股代码
            df = None
            
            # 提取干净的股票代码（移除前缀）
            clean_code = stock_code
            prefix = None
            if '.' in stock_code:
                parts = stock_code.split('.')
                if len(parts) == 2 and parts[0].isdigit():
                    prefix = parts[0]
                    clean_code = parts[1]
                    self.logger.info(f"从{stock_code}中提取干净代码: {clean_code}, 前缀: {prefix}")

            # 尝试多种方法获取数据
            methods_tried = []
            error_messages = []

            # 方法1: 如果原始代码以105.或106.开头，使用stock_us_hist
            if stock_code.startswith('105.') or stock_code.startswith('106.'):
                methods_tried.append("stock_us_hist with original code")
                try:
                    self.logger.info(f"尝试使用stock_us_hist获取数据: {stock_code}")
                    df = ak.stock_us_hist(
                        symbol=stock_code,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq"
                    )
                    if df is not None and not df.empty:
                        self.logger.info(f"使用stock_us_hist获取数据成功: {len(df)}行")
                except Exception as e:
                    error_msg = f"使用stock_us_hist获取{stock_code}数据失败: {str(e)}"
                    self.logger.warning(error_msg)
                    error_messages.append(error_msg)

            # 方法2: 使用干净的代码，直接用stock_us_daily
            if df is None or df.empty:
                methods_tried.append("stock_us_daily with clean code")
                try:
                    self.logger.info(f"尝试使用stock_us_daily获取数据: {clean_code}")
                    df = ak.stock_us_daily(symbol=clean_code, adjust="qfq")
                    if df is not None and not df.empty:
                        self.logger.info(f"使用stock_us_daily获取{clean_code}数据成功: {len(df)}行")
                except Exception as e:
                    error_msg = f"使用stock_us_daily获取{clean_code}数据失败: {str(e)}"
                    self.logger.warning(error_msg)
                    error_messages.append(error_msg)

            # 方法3: 尝试带105.前缀的stock_us_hist
            if df is None or df.empty:
                methods_tried.append("stock_us_hist with 105. prefix")
                try:
                    stock_code_em = f"105.{clean_code}"
                    self.logger.info(f"尝试使用stock_us_hist获取数据: {stock_code_em}")
                    df = ak.stock_us_hist(
                        symbol=stock_code_em,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq"
                    )
                    if df is not None and not df.empty:
                        self.logger.info(f"使用stock_us_hist获取{stock_code_em}数据成功: {len(df)}行")
                except Exception as e:
                    error_msg = f"使用stock_us_hist获取{stock_code_em}数据失败: {str(e)}"
                    self.logger.warning(error_msg)
                    error_messages.append(error_msg)

            # 方法4: 尝试带106.前缀的stock_us_hist
            if df is None or df.empty:
                methods_tried.append("stock_us_hist with 106. prefix")
                try:
                    stock_code_em = f"106.{clean_code}"
                    self.logger.info(f"尝试使用stock_us_hist获取数据: {stock_code_em}")
                    df = ak.stock_us_hist(
                        symbol=stock_code_em,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq"
                    )
                    if df is not None and not df.empty:
                        self.logger.info(f"使用stock_us_hist获取{stock_code_em}数据成功: {len(df)}行")
                except Exception as e:
                    error_msg = f"使用stock_us_hist获取{stock_code_em}数据失败: {str(e)}"
                    self.logger.warning(error_msg)
                    error_messages.append(error_msg)
            
            # 如果所有方法都失败，抛出汇总错误
            if df is None or df.empty:
                self.logger.error(f"所有方法获取{stock_code}数据均失败: {', '.join(methods_tried)}")
                raise ValueError(f"无法获取美股{stock_code}数据，尝试了以下方法: {', '.join(methods_tried)}。错误信息: {'; '.join(error_messages)}")

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
            
            # 记录成功获取的股票代码和方法，以便于调试
            self.logger.info(f"最终成功获取美股 {stock_code} 数据: {len(df)}行，使用方法: {methods_tried[-1]}")

            return df

        except Exception as e:
            self.logger.error(f"获取美股数据失败: {str(e)}")
            raise Exception(f"获取美股数据失败: {str(e)}")