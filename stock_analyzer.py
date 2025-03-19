# -*- coding: utf-8 -*-
"""
智能分析系统（股票） - 股票市场数据分析系统
修改：熊猫大侠
版本：v2.1.0
许可证：MIT License
"""
# stock_analyzer.py
import time
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import logging
import math
import json
import threading

from db.stock_info_dao import StockInfoDAO
from service.dfcf.df_us_detail import DFUSDetail

# Thread-local storage
thread_local = threading.local()


class StockAnalyzer:
    """
    股票分析器 - 原有API保持不变，内部实现增强
    """

    def __init__(self, initial_cash=1000000):
        # 设置日志
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # 加载环境变量
        load_dotenv()

        # 设置 OpenAI API (原来是Gemini API)
        self.openai_api_key = os.getenv('OPENAI_API_KEY', os.getenv('OPENAI_API_KEY'))
        self.openai_api_url = os.getenv('OPENAI_API_URL', 'https://api.openai.com/v1')
        self.openai_model = os.getenv('OPENAI_API_MODEL', 'gemini-2.0-pro-exp-02-05')
        self.news_model = os.getenv('NEWS_MODEL')

        # 配置参数
        self.params = {
            'ma_periods': {'short': 5, 'medium': 20, 'long': 60},
            'rsi_period': 14,
            'bollinger_period': 20,
            'bollinger_std': 2,
            'volume_ma_period': 20,
            'atr_period': 14
        }

        # 添加缓存初始化
        self.data_cache = {}

    def get_stock_data(self, stock_code, market_type='A', start_date=None, end_date=None):
        """获取股票数据"""
        import akshare as ak

        self.logger.info(f"开始获取股票 {stock_code} 数据，市场类型: {market_type}")

        cache_key = f"{stock_code}_{market_type}_{start_date}_{end_date}_price"
        if cache_key in self.data_cache:
            cached_df = self.data_cache[cache_key]
            # Create a copy to avoid modifying the cached data
            # and ensure date is datetime type for the copy
            result = cached_df.copy()
            # If 'date' column exists but is not datetime, convert it
            if 'date' in result.columns and not pd.api.types.is_datetime64_any_dtype(result['date']):
                try:
                    result['date'] = pd.to_datetime(result['date'])
                except Exception as e:
                    self.logger.warning(f"无法将日期列转换为datetime格式: {str(e)}")
            return result

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        try:
            # 根据市场类型获取数据
            if market_type == 'A':
                df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
            elif market_type == 'HK':
                df = ak.stock_hk_daily(
                    symbol=stock_code,
                    adjust="qfq"
                )
            elif market_type == 'US':
                # 使用专门的US股票服务获取美股数据
                from us_stock_service import USStockService
                us_service = USStockService()
                
                try:
                    # 尝试使用USStockService获取数据
                    df = us_service.get_us_stock_data(
                        stock_code=stock_code,
                        start_date=start_date,
                        end_date=end_date
                    )
                    self.logger.info(f"成功通过US股票服务获取 {stock_code} 数据: {len(df)}行")
                except Exception as us_error:
                    self.logger.warning(f"使用US股票服务获取数据失败: {str(us_error)}，尝试使用原始方法")
                    # 回退到原来的方法
                    df = ak.stock_us_hist(
                        symbol=stock_code,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq"
                    )
            else:
                raise ValueError(f"不支持的市场类型: {market_type}")

            # 重命名列名以匹配分析需求
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount"
            })

            # 确保日期格式正确
            df['date'] = pd.to_datetime(df['date'])

            # 数据类型转换
            numeric_columns = ['open', 'close', 'high', 'low', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 删除空值
            df = df.dropna()

            result = df.sort_values('date')

            # 缓存原始数据（包含datetime类型）
            self.data_cache[cache_key] = result.copy()

            return result

        except Exception as e:
            self.logger.error(f"获取股票数据失败: {e}")
            raise Exception(f"获取股票数据失败: {e}")

    def get_north_flow_history(self, stock_code, start_date=None, end_date=None):
        """获取单个股票的北向资金历史持股数据"""
        try:
            import akshare as ak

            # 获取历史持股数据
            if start_date is None and end_date is None:
                # 默认获取近90天数据
                north_hist_data = ak.stock_hsgt_hist_em(symbol=stock_code)
            else:
                north_hist_data = ak.stock_hsgt_hist_em(symbol=stock_code, start_date=start_date, end_date=end_date)

            if north_hist_data.empty:
                return {"history": []}

            # 转换为列表格式返回
            history = []
            for _, row in north_hist_data.iterrows():
                history.append({
                    "date": row.get('日期', ''),
                    "holding": float(row.get('持股数', 0)) if '持股数' in row else 0,
                    "ratio": float(row.get('持股比例', 0)) if '持股比例' in row else 0,
                    "change": float(row.get('持股变动', 0)) if '持股变动' in row else 0,
                    "market_value": float(row.get('持股市值', 0)) if '持股市值' in row else 0
                })

            return {"history": history}
        except Exception as e:
            self.logger.error(f"获取北向资金历史数据出错: {str(e)}")
            return {"history": []}

    def calculate_ema(self, series, period):
        """计算指数移动平均线"""
        return series.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, series, period):
        """计算RSI指标"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, series):
        """计算MACD指标"""
        exp1 = series.ewm(span=12, adjust=False).mean()
        exp2 = series.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def calculate_bollinger_bands(self, series, period, std_dev):
        """计算布林带"""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def calculate_atr(self, df, period):
        """计算ATR指标"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def format_indicator_data(self, df):
        """格式化指标数据，控制小数位数"""

        # 格式化价格数据 (2位小数)
        price_columns = ['open', 'close', 'high', 'low', 'MA5', 'MA20', 'MA60', 'BB_upper', 'BB_middle', 'BB_lower']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].round(2)

        # 格式化MACD相关指标 (3位小数)
        macd_columns = ['MACD', 'Signal', 'MACD_hist']
        for col in macd_columns:
            if col in df.columns:
                df[col] = df[col].round(3)

        # 格式化其他技术指标 (2位小数)
        other_columns = ['RSI', 'Volatility', 'ROC', 'Volume_Ratio']
        for col in other_columns:
            if col in df.columns:
                df[col] = df[col].round(2)

        return df

    def calculate_indicators(self, df):
        """计算技术指标"""

        try:
            # 计算移动平均线
            df['MA5'] = self.calculate_ema(df['close'], self.params['ma_periods']['short'])
            df['MA20'] = self.calculate_ema(df['close'], self.params['ma_periods']['medium'])
            df['MA60'] = self.calculate_ema(df['close'], self.params['ma_periods']['long'])

            # 计算RSI
            df['RSI'] = self.calculate_rsi(df['close'], self.params['rsi_period'])

            # 计算MACD
            df['MACD'], df['Signal'], df['MACD_hist'] = self.calculate_macd(df['close'])

            # 计算布林带
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.calculate_bollinger_bands(
                df['close'],
                self.params['bollinger_period'],
                self.params['bollinger_std']
            )

            # 成交量分析
            df['Volume_MA'] = df['volume'].rolling(window=self.params['volume_ma_period']).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_MA']

            # 计算ATR和波动率
            df['ATR'] = self.calculate_atr(df, self.params['atr_period'])
            df['Volatility'] = df['ATR'] / df['close'] * 100

            # 动量指标
            df['ROC'] = df['close'].pct_change(periods=10) * 100

            # 格式化数据
            df = self.format_indicator_data(df)

            return df

        except Exception as e:
            self.logger.error(f"计算技术指标时出错: {str(e)}")
            raise

    def calculate_score(self, df, market_type='A'):
        """
        计算股票综合评分 - 基于多维度分析框架
        根据不同市场特性调整评分权重和标准
        返回0-100的评分，其中:
        0-20: 强烈卖出信号
        21-40: 卖出信号
        41-60: 观望信号
        61-80: 买入信号
        81-100: 强烈买入信号
        """
        try:
            score = 0
            if len(df) < 30:  # 确保有足够的数据进行分析
                self.logger.warning(f"数据量不足，无法进行可靠评分: {len(df)}行")
                return 50  # 返回中性评分
                
            latest = df.iloc[-1]
            prev_days = min(30, len(df) - 1)
            
            # 基础权重配置 - 更加平衡的权重分配
            weights = {
                'trend': 0.20,       # 趋势分析
                'momentum': 0.20,    # 动量分析
                'technical': 0.25,   # 技术指标
                'volatility': 0.15,  # 波动性分析
                'volume': 0.10,      # 成交量分析
                'market': 0.10       # 市场环境
            }
            
            # 根据市场类型调整权重
            if market_type == 'US':
                # 美股更重视长期趋势和基本面
                weights['trend'] = 0.25
                weights['momentum'] = 0.15
                weights['market'] = 0.15
            elif market_type == 'HK':
                # 港股更受A股和国际市场影响
                weights['market'] = 0.15
                weights['momentum'] = 0.15
            
            # 1. 趋势分析 (0-100分)
            trend_score = 0
            
            # 均线系统评估 - 多周期分析
            # 短期趋势 (5日均线)
            if latest['MA5'] > latest['MA20']:
                trend_score += 20
            
            # 中期趋势 (20日均线)
            if latest['MA20'] > latest['MA60']:
                trend_score += 20
            
            # 多头排列 (黄金排列)
            if latest['MA5'] > latest['MA20'] and latest['MA20'] > latest['MA60']:
                trend_score += 20
            
            # 价格位置评估
            if latest['close'] > latest['MA5']:
                trend_score += 10
            if latest['close'] > latest['MA20']:
                trend_score += 10
            if latest['close'] > latest['MA60']:
                trend_score += 10
            
            # 趋势持续性评估 - 检查最近5天的均线方向
            ma5_direction = 0
            ma20_direction = 0
            
            for i in range(1, min(6, len(df))):
                if df.iloc[-i]['MA5'] > df.iloc[-i-1]['MA5']:
                    ma5_direction += 1
                if df.iloc[-i]['MA20'] > df.iloc[-i-1]['MA20']:
                    ma20_direction += 1
            
            # 均线向上的天数越多，评分越高
            trend_score += (ma5_direction / 5) * 5
            trend_score += (ma20_direction / 5) * 5
            
            # 确保最大分数限制
            trend_score = min(100, trend_score)
            
            # 2. 动量分析 (0-100分)
            momentum_score = 0
            
            # ROC动量指标评估
            roc = latest['ROC']
            if roc > 8:  # 强劲上涨动量
                momentum_score += 40
            elif 4 <= roc <= 8:  # 良好上涨动量
                momentum_score += 30
            elif 0 <= roc < 4:  # 弱上涨动量
                momentum_score += 20
            elif -4 <= roc < 0:  # 弱下跌动量
                momentum_score += 10
            elif -8 <= roc < -4:  # 中等下跌动量
                momentum_score += 5
            
            # 多周期动量比较 - 加速度分析
            try:
                if len(df) >= 21:
                    # 计算不同周期的价格变化率
                    price_change_5d = (latest['close'] / df.iloc[-6]['close'] - 1) * 100
                    price_change_10d = (latest['close'] / df.iloc[-11]['close'] - 1) * 100
                    price_change_20d = (latest['close'] / df.iloc[-21]['close'] - 1) * 100
                    
                    # 动量加速 - 短期动量强于长期动量
                    if price_change_5d > price_change_10d > price_change_20d and price_change_5d > 0:
                        # 完美的动量加速
                        momentum_score += 30
                    elif price_change_5d > price_change_10d and price_change_5d > 0:
                        # 短期加速
                        momentum_score += 20
                    elif price_change_10d > price_change_20d and price_change_10d > 0:
                        # 中期加速
                        momentum_score += 15
                    elif price_change_5d > 0 and price_change_10d > 0 and price_change_20d > 0:
                        # 全周期正动量
                        momentum_score += 10
            except Exception as e:
                self.logger.warning(f"计算多周期动量时出错: {str(e)}")
            
            # 价格突破评估
            try:
                # 计算20日高点和低点
                high_20d = max([df.iloc[-i]['high'] for i in range(1, min(21, len(df)))])
                low_20d = min([df.iloc[-i]['low'] for i in range(1, min(21, len(df)))])
                
                # 突破20日高点
                if latest['close'] > high_20d * 0.99 and latest['close'] < high_20d * 1.03:
                    momentum_score += 30
                # 跌破20日低点
                elif latest['close'] < low_20d * 1.01 and latest['close'] > low_20d * 0.97:
                    momentum_score -= 20
            except Exception as e:
                self.logger.warning(f"计算价格突破时出错: {str(e)}")
            
            # 确保分数在0-100范围内
            momentum_score = max(0, min(100, momentum_score))
            
            # 3. 技术指标分析 (0-100分)
            technical_score = 0
            
            # RSI指标评估 - 超买超卖与背离
            rsi = latest['RSI']
            if 40 <= rsi <= 60:  # 中性区域，稳定趋势
                technical_score += 15
            elif 30 <= rsi < 40:  # 接近超卖，潜在买入机会
                technical_score += 25
            elif 20 <= rsi < 30:  # 超卖区域，强烈买入信号
                technical_score += 30
            elif rsi < 20:  # 极度超卖，可能反转信号
                technical_score += 25
            elif 60 < rsi <= 70:  # 接近超买，潜在卖出机会
                technical_score += 10
            elif 70 < rsi <= 80:  # 超买区域，卖出信号
                technical_score += 5
            elif rsi > 80:  # 极度超买，强烈卖出信号
                technical_score += 0
            
            # RSI背离检测
            try:
                if len(df) >= 10:
                    # 检查价格新高而RSI未创新高 (顶背离)
                    price_new_high = latest['close'] > max([df.iloc[-i]['close'] for i in range(2, 10)])
                    rsi_not_new_high = latest['RSI'] < max([df.iloc[-i]['RSI'] for i in range(2, 10)])
                    
                    if price_new_high and rsi_not_new_high and rsi > 70:
                        technical_score -= 20  # 顶背离，看跌信号
                    
                    # 检查价格新低而RSI未创新低 (底背离)
                    price_new_low = latest['close'] < min([df.iloc[-i]['close'] for i in range(2, 10)])
                    rsi_not_new_low = latest['RSI'] > min([df.iloc[-i]['RSI'] for i in range(2, 10)])
                    
                    if price_new_low and rsi_not_new_low and rsi < 30:
                        technical_score += 20  # 底背离，看涨信号
            except Exception as e:
                self.logger.warning(f"检测RSI背离时出错: {str(e)}")
            
            # MACD指标评估
            if latest['MACD'] > latest['Signal'] and latest['MACD_hist'] > 0:
                # MACD金叉且柱状图为正
                technical_score += 25
            elif latest['MACD'] > latest['Signal']:
                # MACD金叉
                technical_score += 20
            elif latest['MACD_hist'] > 0 and latest['MACD_hist'] > df.iloc[-2]['MACD_hist']:
                # 柱状图为正且增加，上升趋势增强
                technical_score += 15
            elif latest['MACD_hist'] > df.iloc[-2]['MACD_hist']:
                # 柱状图增加，潜在反转信号
                technical_score += 10
            elif latest['MACD'] < latest['Signal'] and latest['MACD_hist'] < 0:
                # MACD死叉且柱状图为负
                technical_score += 0
            else:
                # 其他MACD情况
                technical_score += 5
            
            # 布林带位置评估
            bb_position = (latest['close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'])
            
            if bb_position < 0.2:  # 接近下轨，潜在超卖
                technical_score += 25
            elif 0.2 <= bb_position < 0.4:  # 下轨和中轨之间，潜在买入区域
                technical_score += 20
            elif 0.4 <= bb_position <= 0.6:  # 中轨附近，中性
                technical_score += 15
            elif 0.6 < bb_position <= 0.8:  # 中轨和上轨之间，潜在卖出区域
                technical_score += 5
            elif bb_position > 0.8:  # 接近上轨，潜在超买
                technical_score += 0
            
            # 布林带宽度评估 - 挤压和扩张
            try:
                bb_width = (latest['BB_upper'] - latest['BB_lower']) / latest['BB_middle']
                bb_width_prev = (df.iloc[-10]['BB_upper'] - df.iloc[-10]['BB_lower']) / df.iloc[-10]['BB_middle']
                
                if bb_width < bb_width_prev * 0.8:  # 布林带收窄，波动性降低，可能即将爆发
                    technical_score += 10
                elif bb_width > bb_width_prev * 1.5:  # 布林带扩张，波动性增加
                    technical_score += 5
            except Exception as e:
                self.logger.warning(f"计算布林带宽度时出错: {str(e)}")
            
            # 确保分数在0-100范围内
            technical_score = max(0, min(100, technical_score))
            
            # 4. 波动性分析 (0-100分)
            volatility_score = 0
            
            # 波动率评估 - 适中的波动率最佳
            volatility = latest['Volatility']
            if 0.8 <= volatility <= 2.5:  # 理想波动率范围
                volatility_score += 50
            elif 0.5 <= volatility < 0.8 or 2.5 < volatility <= 3.5:  # 次优波动率范围
                volatility_score += 40
            elif 0.3 <= volatility < 0.5 or 3.5 < volatility <= 5.0:  # 可接受波动率范围
                volatility_score += 30
            elif volatility < 0.3:  # 波动率过低，缺乏活力
                volatility_score += 20
            else:  # 波动率过高，风险较大
                volatility_score += 10
            
            # ATR趋势评估
            try:
                atr_5d_avg = sum([df.iloc[-i]['ATR'] for i in range(1, 6)]) / 5
                atr_20d_avg = sum([df.iloc[-i]['ATR'] for i in range(1, 21)]) / 20
                
                # ATR上升，波动性增加
                if latest['ATR'] > atr_5d_avg > atr_20d_avg:
                    volatility_score += 20
                # ATR下降，波动性减少
                elif latest['ATR'] < atr_5d_avg < atr_20d_avg:
                    volatility_score += 30
            except Exception as e:
                self.logger.warning(f"计算ATR趋势时出错: {str(e)}")
            
            # 价格波动范围评估
            try:
                # 计算最近10天的日内波动率
                intraday_volatility = [(df.iloc[-i]['high'] - df.iloc[-i]['low']) / df.iloc[-i]['low'] * 100 for i in range(1, 11)]
                avg_intraday_volatility = sum(intraday_volatility) / len(intraday_volatility)
                
                if 1.0 <= avg_intraday_volatility <= 3.0:  # 适中的日内波动
                    volatility_score += 20
                elif avg_intraday_volatility < 1.0:  # 日内波动过小
                    volatility_score += 10
                else:  # 日内波动过大
                    volatility_score += 0
            except Exception as e:
                self.logger.warning(f"计算日内波动率时出错: {str(e)}")
            
            # 确保分数在0-100范围内
            volatility_score = max(0, min(100, volatility_score))
            
            # 5. 成交量分析 (0-100分)
            volume_score = 0
            
            # 成交量趋势分析
            recent_vol_ratio = [df.iloc[-i]['Volume_Ratio'] for i in range(1, min(6, len(df)))]
            avg_vol_ratio = sum(recent_vol_ratio) / len(recent_vol_ratio)
            
            # 成交量与价格配合评估
            if avg_vol_ratio > 1.5 and latest['close'] > df.iloc[-2]['close']:
                # 放量上涨，强烈看涨信号
                volume_score += 50
            elif avg_vol_ratio > 1.2 and latest['close'] > df.iloc[-2]['close']:
                # 量价齐升，看涨信号
                volume_score += 40
            elif 0.8 <= avg_vol_ratio <= 1.2:
                # 正常成交量，市场稳定
                volume_score += 30
            elif avg_vol_ratio < 0.8 and latest['close'] < df.iloc[-2]['close']:
                # 缩量下跌，可能是健康回调
                volume_score += 20
            elif avg_vol_ratio > 1.2 and latest['close'] < df.iloc[-2]['close']:
                # 放量下跌，可能有较大卖压
                volume_score += 10
            else:
                # 其他情况
                volume_score += 20
            
            # 成交量变化趋势
            try:
                vol_trend = 0
                # 使用volume列而不是Volume列，确保与calculate_indicators方法中的列名一致
                for i in range(1, min(6, len(df))):
                    if df.iloc[-i]['volume'] > df.iloc[-i-1]['volume']:
                        vol_trend += 1
                
                # 成交量连续增加
                volume_score += (vol_trend / 5) * 20
            except Exception as e:
                self.logger.warning(f"计算成交量趋势时出错: {str(e)}")
                # 出错时不调整分数
            
            # 成交量突变检测
            try:
                # 使用volume列而不是Volume列，确保与calculate_indicators方法中的列名一致
                vol_avg_10d = sum([df.iloc[-i]['volume'] for i in range(2, 12)]) / 10
                
                if latest['volume'] > vol_avg_10d * 2:  # 成交量是10日均量的2倍以上
                    if latest['close'] > df.iloc[-2]['close']:  # 且价格上涨
                        volume_score += 30
                    else:  # 价格下跌
                        volume_score -= 10
            except Exception as e:
                self.logger.warning(f"检测成交量突变时出错: {str(e)}")
                # 出错时不调整分数
            
            # 确保分数在0-100范围内
            volume_score = max(0, min(100, volume_score))
            
            # 6. 市场环境分析 (0-100分)
            market_score = 50  # 默认中性评分
            
            # 获取对应市场指数
            index_code = None
            if market_type == 'A':
                index_code = 'sh000001'  # 上证指数
            elif market_type == 'HK':
                index_code = 'HSI'  # 恒生指数
            elif market_type == 'US':
                index_code = 'SPX'  # 标普500
            
            # 尝试获取指数数据并评估市场环境
            if index_code:
                try:
                    index_df = self._get_index_data(index_code, market_type)
                    
                    if index_df is not None and len(index_df) > 20:
                        # 计算指数趋势
                        try:
                            index_latest = index_df.iloc[-1]
                            
                            # 确保close列存在
                            if 'close' not in index_df.columns:
                                self.logger.warning(f"指数数据缺少close列，尝试寻找替代列")
                                # 尝试找到一个可能的替代列
                                possible_cols = [col for col in index_df.columns if col.lower() in ['close', 'closing', 'price', 'last', 'value']]
                                if possible_cols:
                                    index_df['close'] = index_df[possible_cols[0]]
                                else:
                                    raise ValueError("找不到可用作close的列")
                                    
                            # 计算移动平均线
                            index_ma5 = index_df['close'].rolling(window=5).mean().iloc[-1]
                            index_ma20 = index_df['close'].rolling(window=20).mean().iloc[-1]
                            
                            # 指数趋势评分
                            if index_latest['close'] > index_ma5 > index_ma20:
                                # 指数多头排列
                                market_score += 30
                            elif index_latest['close'] > index_ma5:
                                # 指数短期向上
                                market_score += 20
                            elif index_latest['close'] < index_ma5 < index_ma20:
                                # 指数空头排列
                                market_score -= 20
                            elif index_latest['close'] < index_ma5:
                                # 指数短期向下
                                market_score -= 10
                                
                            # 确保评分在0-100范围内
                            market_score = max(0, min(100, market_score))
                            
                        except Exception as trend_e:
                            self.logger.warning(f"计算指数趋势时出错: {str(trend_e)}")
                            # 保留默认的中性评分
                    else:
                        self.logger.warning(f"获取到的指数数据不足，使用默认市场评分")
                        
                        # 针对美股市场，如果无法获取SPX数据，降低市场评分在总分中的权重
                        if market_type == 'US':
                            self.logger.info("美股市场无指数数据，调整市场权重")
                            weights['market'] = 0.05  # 降低市场评分权重
                            # 将权重分配到其他因素
                            weights['trend'] += 0.03
                            weights['technical'] += 0.02
                            
                except Exception as idx_e:
                    self.logger.warning(f"处理指数数据时出错: {str(idx_e)}")
                    # 出错时保留默认的中性评分
                    
                    # 针对美股市场，如果无法获取SPX数据，降低市场评分在总分中的权重
                    if market_type == 'US':
                        self.logger.info("美股市场无法获取指数数据，调整权重")
                        weights['market'] = 0.05  # 降低市场评分权重
                        # 将权重分配到其他因素
                        weights['trend'] += 0.03
                        weights['technical'] += 0.02

            # 特殊市场调整
            if market_type == 'US':
                # 美股特殊调整 - 财报季
                is_earnings_season = self._is_earnings_season()
                if is_earnings_season:
                    # 财报季波动性更高，调整确定性
                    market_score = market_score * 0.9  # 降低确定性
            
            elif market_type == 'HK':
                # 港股特殊调整 - A股联动效应
                a_share_linkage = self._check_a_share_linkage(df)
                if a_share_linkage > 0.7:  # 高度联动
                    # 根据内地市场情绪调整
                    mainland_sentiment = self._get_mainland_market_sentiment()
                    market_score += mainland_sentiment * 10
            
            # 确保分数在0-100范围内
            market_score = max(0, min(100, market_score))

            # 计算加权总分
            final_score = (
                trend_score * weights['trend'] +
                momentum_score * weights['momentum'] +
                technical_score * weights['technical'] +
                volatility_score * weights['volatility'] +
                volume_score * weights['volume'] +
                market_score * weights['market']
            )
            
            # 确保最终分数在0-100范围内
            final_score = max(0, min(100, round(final_score)))
            
            # 存储各维度评分详情
            self.score_details = {
                'trend': trend_score,
                'momentum': momentum_score,
                'technical': technical_score,
                'volatility': volatility_score,
                'volume': volume_score,
                'market': market_score,
                'total': final_score
            }
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"计算评分时出错: {str(e)}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            # 出错时返回中性评分
            return 50

    def _get_index_data(self, index_code, market_type='A'):
        """
        获取指数数据的专用方法
        
        参数:
            index_code: 指数代码
            market_type: 市场类型(A/HK/US)
            
        返回:
            DataFrame: 指数历史数据，如果获取失败则返回None
        """
        try:
            import akshare as ak
            
            # 缓存键
            cache_key = f"{index_code}_index_data"
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]
            
            # 根据市场类型获取不同的指数数据
            if market_type == 'A':
                # 使用akshare获取A股指数数据
                df = ak.stock_zh_index_daily(symbol=index_code)
                # 确保列名标准化
                if 'close' not in df.columns and '收盘' in df.columns:
                    df = df.rename(columns={'收盘': 'close'})
                
            elif market_type == 'HK':
                # 获取港股指数数据
                df = ak.stock_hk_index_daily_em(symbol=index_code)
                # 确保列名标准化
                if 'close' not in df.columns and '收盘价' in df.columns:
                    df = df.rename(columns={'收盘价': 'close'})
                
            elif market_type == 'US':
                # 获取美股指数数据
                try:
                    # 首先尝试使用sina的API (推荐方法，更稳定)
                    self.logger.info(f"尝试使用新浪API获取美股指数 {index_code} 数据")
                    
                    # 映射标准指数代码到新浪代码
                    sina_index_map = {
                        'SPX': '.INX',  # 标普500
                        'DJI': '.DJI',  # 道琼斯工业
                        'IXIC': '.IXIC' # 纳斯达克
                    }
                    sina_symbol = sina_index_map.get(index_code, index_code)
                    
                    # 如果代码不是以"."开头，而且不在映射表中，可能需要添加"."前缀
                    if not sina_symbol.startswith('.') and sina_symbol not in sina_index_map.values():
                        sina_symbol = f".{sina_symbol}"
                    
                    self.logger.info(f"转换后的新浪指数代码: {sina_symbol}")
                    df = ak.index_us_stock_sina(symbol=sina_symbol)
                    
                    # 检查结果是否有效
                    if df is None or df.empty:
                        raise ValueError("返回的数据为空")
                    
                    # 确保列名标准化
                    if 'close' not in df.columns and '收盘价' in df.columns:
                        df = df.rename(columns={'收盘价': 'close'})
                        
                    self.logger.info(f"成功获取美股指数 {index_code} 数据: {len(df)}行")
                except Exception as us_e:
                    self.logger.warning(f"使用新浪API获取美股指数 {index_code} 数据失败: {str(us_e)}")
                    
                    try:
                        # 尝试使用stock_us_daily直接获取，这里使用原始代码，但使用正确的映射
                        self.logger.info(f"尝试使用stock_us_daily获取美股指数 {index_code} 数据")
                        
                        # 对于标普500指数，必须使用.INX代码
                        em_index_map = {
                            'SPX': '.INX',  # 标普500
                            'DJI': '.DJI',  # 道琼斯工业
                            'IXIC': '.IXIC'  # 纳斯达克
                        }
                        
                        em_code = em_index_map.get(index_code, index_code)
                        self.logger.info(f"使用stock_us_daily获取美股指数: {em_code}")
                        df = ak.stock_us_daily(symbol=em_code)
                        
                        # 检查结果是否有效
                        if df is None or df.empty:
                            raise ValueError("返回的数据为空")
                            
                        # 确保列名标准化
                        if 'close' not in df.columns:
                            for col_name in ['收盘', '收盘价', 'Close', 'close']:
                                if col_name in df.columns:
                                    df = df.rename(columns={col_name: 'close'})
                                    break
                                    
                        self.logger.info(f"成功使用stock_us_daily获取美股指数 {index_code} 数据: {len(df)}行")
                    except Exception as em_e:
                        self.logger.error(f"使用stock_us_daily获取美股指数 {index_code} 数据失败: {str(em_e)}")
                        
                        # 尝试使用其他备用方法 - 这里仅保留兼容性，但实际场景下可能不需要
                        self.logger.warning(f"无法获取美股指数 {index_code} 数据，调整市场评分权重")
                        
                        # 由于无法获取指数数据，我们调整打分时市场因素的权重
                        return None
            else:
                raise ValueError(f"不支持的市场类型: {market_type}")

            # 缓存数据
            if df is not None and not df.empty:
                # 确保有'date'列和'close'列
                if 'date' not in df.columns and df.index.name == 'date':
                    df = df.reset_index()
                
                # 确保日期格式标准化
                if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'])
                    
                # 对所有数值列进行类型转换
                for col in df.columns:
                    if col != 'date' and df[col].dtype != 'object':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                self.data_cache[cache_key] = df
                return df
            else:
                self.logger.warning(f"获取 {market_type} 市场指数 {index_code} 的数据为空")
                return None
                
        except Exception as e:
            self.logger.warning(f"获取指数 {index_code} 数据失败: {str(e)}")
            return None

    def calculate_position_size(self, stock_code, risk_percent=2.0, stop_loss_percent=5.0):
        """
        Calculate optimal position size based on risk management principles
        Implements the "Position Sizing Formula" from Time-Space Resonance System

        Parameters:
            stock_code: Stock code to analyze
            risk_percent: Percentage of total capital to risk on this trade (default 2%)
            stop_loss_percent: Stop loss percentage from entry point (default 5%)

        Returns:
            Position size as percentage of total capital
        """
        try:
            # Get stock data
            df = self.get_stock_data(stock_code)
            df = self.calculate_indicators(df)

            # Get volatility factor (from dimension 3: Energy Conservation)
            latest = df.iloc[-1]
            volatility = latest['Volatility']

            # Calculate volatility adjustment factor (higher volatility = smaller position)
            volatility_factor = 1.0
            if volatility > 4.0:
                volatility_factor = 0.6  # Reduce position for high volatility stocks
            elif volatility > 2.5:
                volatility_factor = 0.8  # Slightly reduce position
            elif volatility < 1.0:
                volatility_factor = 1.2  # Can increase position for low volatility stocks

            # Calculate position size using risk formula
            # Formula: position_size = (risk_amount) / (stop_loss * volatility_factor)
            position_size = (risk_percent) / (stop_loss_percent * volatility_factor)

            # Limit maximum position to 25% for diversification
            position_size = min(position_size, 25.0)

            return position_size

        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            # Return conservative default position size on error
            return 5.0

    def get_recommendation(self, score, market_type='A', technical_data=None, news_data=None):
        """
        Generate investment recommendation based on score and additional information
        Enhanced with Time-Space Resonance Trading System strategies
        """
        try:
            # 1. Base recommendation logic - 大幅调整阈值，匹配新的评分系统
            if score >= 75:  # 降低强烈买入阈值
                base_recommendation = '强烈建议买入'
                confidence = 'high'
                action = 'strong_buy'
            elif score >= 60:  # 降低买入阈值
                base_recommendation = '建议买入'
                confidence = 'medium_high'
                action = 'buy'
            elif score >= 45:  # 降低谨慎买入阈值
                base_recommendation = '谨慎买入'
                confidence = 'medium'
                action = 'cautious_buy'
            elif score >= 35:  # 调整观望阈值
                base_recommendation = '持观望态度'
                confidence = 'medium'
                action = 'hold'
            elif score >= 25:  # 调整谨慎持有阈值
                base_recommendation = '谨慎持有'
                confidence = 'medium'
                action = 'cautious_hold'
            elif score >= 15:  # 调整减仓阈值
                base_recommendation = '建议减仓'
                confidence = 'medium_high'
                action = 'reduce'
            else:  # 卖出阈值不变
                base_recommendation = '建议卖出'
                confidence = 'high'
                action = 'sell'

            # 2. Market type adjustment
            market_adjustment = ""
            if market_type == 'HK':
                market_adjustment = "(针对港股市场)"
            elif market_type == 'US':
                market_adjustment = "(针对美股市场)"
                
                # 美股市场交易习惯与A股不同，调整建议
                if action in ['cautious_buy', 'buy', 'strong_buy']:
                    market_adjustment = "(美股市场买入机会)"
                elif action in ['cautious_hold', 'hold']:
                    market_adjustment = "(美股市场观望)"
                elif action in ['reduce', 'sell']:
                    market_adjustment = "(美股市场卖出信号)"

            # 3. Technical indicators adjustment
            technical_adjustment = ""
            if technical_data:
                # Process technical indicators to adjust recommendation
                pass  # 代码省略，保持原有逻辑

            # 4. Adjust action based on technical data
            if technical_data:
                # 获取价格趋势
                price_trend = technical_data.get('price_trend', 0)
                
                # 强价格趋势可能会调整建议
                if price_trend > 3 and action in ['hold', 'cautious_hold']:  # 价格强势上涨
                    action = 'cautious_buy'  # 升级为谨慎买入
                    technical_adjustment += "价格呈上涨趋势，"
                elif price_trend < -3 and action not in ['sell', 'reduce']:  # 价格强势下跌
                    action = 'cautious_hold'  # 降级为谨慎持有
                    technical_adjustment += "价格呈下跌趋势，"

            # 5. Convert adjusted action to final recommendation
            action_to_recommendation = {
                'strong_buy': '强烈建议买入',
                'buy': '建议买入',
                'cautious_buy': '谨慎买入',
                'hold': '持观望态度',
                'cautious_hold': '谨慎持有',
                'reduce': '建议减仓',
                'sell': '建议卖出'
            }

            final_recommendation = action_to_recommendation.get(action, base_recommendation)

            # 6. Combine all adjustment factors
            adjustments = " ".join(filter(None, [market_adjustment, technical_adjustment]))

            if adjustments:
                return {
                    'action': final_recommendation,
                    'details': adjustments,
                    'market_type': market_type
                }
            else:
                return {
                    'action': final_recommendation,
                    'details': '',
                    'market_type': market_type
                }

        except Exception as e:
            self.logger.error(f"Error generating investment recommendation: {str(e)}")
            # Return safe default recommendation on error
            return {
                'action': "无法提供明确建议，请结合多种因素谨慎决策",
                'details': '',
                'market_type': market_type
            }

    def check_consecutive_losses(self, trade_history, max_consecutive_losses=3):
        """
        Implement the "Refractory Period Risk Control" - stop trading after consecutive losses

        Parameters:
            trade_history: List of recent trade results (True for profit, False for loss)
            max_consecutive_losses: Maximum allowed consecutive losses

        Returns:
            Boolean: True if trading should be paused, False if trading can continue
        """
        consecutive_losses = 0

        # Count consecutive losses from most recent trades
        for trade in reversed(trade_history):
            if not trade:  # If trade is a loss
                consecutive_losses += 1
            else:
                break  # Break on first profitable trade

        # Return True if we've hit max consecutive losses
        return consecutive_losses >= max_consecutive_losses

    def check_profit_taking(self, current_profit_percent, threshold=20.0):
        """
        Implement profit-taking mechanism when returns exceed threshold
        Part of "Energy Conservation Dimension"

        Parameters:
            current_profit_percent: Current profit percentage
            threshold: Profit percentage threshold for taking profits

        Returns:
            Float: Percentage of position to reduce (0.0-1.0)
        """
        if current_profit_percent >= threshold:
            # If profit exceeds threshold, suggest reducing position by 50%
            return 0.5

        return 0.0  # No position reduction recommended

    def _is_earnings_season(self):
        """检查当前是否处于财报季(辅助函数)"""
        from datetime import datetime
        current_month = datetime.now().month
        # 美股财报季大致在1月、4月、7月和10月
        return current_month in [1, 4, 7, 10]

    def _check_a_share_linkage(self, df, window=20):
        """检查港股与A股的联动性(辅助函数)"""
        # 该函数需要获取对应的A股指数数据
        # 简化版实现:
        try:
            # 获取恒生指数与上证指数的相关系数
            # 实际实现中需要获取真实数据
            correlation = 0.6  # 示例值
            return correlation
        except:
            return 0.5  # 默认中等关联度

    def _get_mainland_market_sentiment(self):
        """获取中国大陆市场情绪(辅助函数)"""
        # 实际实现中需要分析上证指数、北向资金等因素
        try:
            # 简化版实现，返回-1到1之间的值，1表示积极情绪
            sentiment = 0.2  # 示例值
            return sentiment
        except:
            return 0  # 默认中性情绪

    def get_stock_news(self, stock_code, market_type='A', limit=5):
        """
        获取股票相关新闻和实时信息，通过OpenAI API调用news模型获取
        参数:
            stock_code: 股票代码
            market_type: 市场类型 (A/HK/US)
            limit: 返回的新闻条数上限
        返回:
            包含新闻和公告的字典
        """
        try:
            self.logger.info(f"获取股票 {stock_code} 的相关新闻和信息")

            # 缓存键
            cache_key = f"{stock_code}_{market_type}_news"
            if cache_key in self.data_cache and (
                    datetime.now() - self.data_cache[cache_key]['timestamp']).seconds < 3600:
                # 缓存1小时内的数据
                return self.data_cache[cache_key]['data']

            # 获取股票基本信息
            stock_info = self.get_stock_info(stock_code)
            stock_name = stock_info.get('股票名称', '未知')
            industry = stock_info.get('行业', '未知')

            # 构建新闻查询的prompt
            market_name = "A股" if market_type == 'A' else "港股" if market_type == 'HK' else "美股"
            query = f"""请提供以下股票的最新相关新闻和信息:
            股票名称: {stock_name}
            股票代码: {stock_code}
            市场: {market_name}
            行业: {industry}

            请返回以下格式的JSON数据:
            {{
                "news": [
                    {{"title": "新闻标题", "date": "YYYY-MM-DD", "source": "新闻来源", "summary": "新闻摘要"}},
                    ...
                ],
                "announcements": [
                    {{"title": "公告标题", "date": "YYYY-MM-DD", "type": "公告类型"}},
                    ...
                ],
                "industry_news": [
                    {{"title": "行业新闻标题", "date": "YYYY-MM-DD", "summary": "新闻摘要"}},
                    ...
                ],
                "market_sentiment": "市场情绪(bullish/slightly_bullish/neutral/slightly_bearish/bearish)"
            }}

            每个类别最多返回{limit}条。如果无法获取实际新闻，请基于行业知识生成合理的示例数据。
            """

            messages = [{"role": "user", "content": query}]

            # 使用线程和队列添加超时控制
            import queue
            import threading
            import json
            import openai

            result_queue = queue.Queue()

            def call_api():
                try:
                    # 使用OpenAI API调用news模型
                    response = openai.ChatCompletion.create(
                        model=self.news_model,  # 使用news模型
                        messages=messages,
                        temperature=0.7,
                        max_tokens=4000,
                        stream=False,
                        timeout=240
                    )
                    result_queue.put(response)
                except Exception as e:
                    result_queue.put(e)

            # 启动API调用线程
            api_thread = threading.Thread(target=call_api)
            api_thread.daemon = True
            api_thread.start()

            # 等待结果，最多等待20秒
            try:
                result = result_queue.get(timeout=240)

                # 检查结果是否为异常
                if isinstance(result, Exception):
                    self.logger.error(f"获取新闻API调用失败: {str(result)}")
                    raise result

                # 提取回复内容
                content = result["choices"][0]["message"]["content"].strip()

                # 解析JSON
                try:
                    # 尝试直接解析JSON
                    news_data = json.loads(content)
                except json.JSONDecodeError:
                    # 如果直接解析失败，尝试提取JSON部分
                    import re
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                    if json_match:
                        json_str = json_match.group(1)
                        news_data = json.loads(json_str)
                    else:
                        raise ValueError("无法从响应中提取JSON数据")

                # 添加时间戳
                news_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # 缓存结果
                self.data_cache[cache_key] = {
                    'data': news_data,
                    'timestamp': datetime.now()
                }

                return news_data

            except queue.Empty:
                self.logger.warning("获取新闻API调用超时")
                return {
                    'news': [],
                    'announcements': [],
                    'industry_news': [],
                    'market_sentiment': 'neutral',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            except Exception as e:
                self.logger.error(f"处理新闻数据时出错: {str(e)}")
                return {
                    'news': [],
                    'announcements': [],
                    'industry_news': [],
                    'market_sentiment': 'neutral',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

        except Exception as e:
            self.logger.error(f"获取股票新闻时出错: {str(e)}")
            # 出错时返回空结果
            return {
                'news': [],
                'announcements': [],
                'industry_news': [],
                'market_sentiment': 'neutral',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def get_ai_analysis(self, df, stock_code, market_type='A'):
        """
        使用AI进行增强分析
        结合技术指标、实时新闻和行业信息

        参数:
            df: 股票历史数据DataFrame
            stock_code: 股票代码
            market_type: 市场类型(A/HK/US)

        返回:
            AI生成的分析报告文本
        """
        try:
            import openai
            import threading
            import queue

            # 设置API密钥和基础URL
            openai.api_key = self.openai_api_key
            openai.api_base = self.openai_api_url

            # 1. 获取最近K线数据
            recent_data = df.tail(20).to_dict('records')

            # 2. 计算技术指标摘要
            technical_summary = {
                'trend': 'upward' if df.iloc[-1]['MA5'] > df.iloc[-1]['MA20'] else 'downward',
                'volatility': f"{df.iloc[-1]['Volatility']:.2f}%",
                'volume_trend': 'increasing' if df.iloc[-1]['Volume_Ratio'] > 1 else 'decreasing',
                'rsi_level': df.iloc[-1]['RSI'],
                'macd_signal': 'bullish' if df.iloc[-1]['MACD'] > df.iloc[-1]['Signal'] else 'bearish',
                'bb_position': self._calculate_bb_position(df)
            }

            # 3. 获取支撑压力位
            sr_levels = self.identify_support_resistance(df)

            # 4. 获取股票基本信息
            stock_info = self.get_stock_info(stock_code)
            stock_name = stock_info.get('股票名称', '未知')
            industry = stock_info.get('行业', '未知')

            # 5. 获取相关新闻和实时信息 - 整合get_stock_news
            self.logger.info(f"获取 {stock_code} 的相关新闻和市场信息")
            news_data = self.get_stock_news(stock_code, market_type)

            # 6. 评分分解
            score = self.calculate_score(df, market_type)
            score_details = getattr(self, 'score_details', {'total': score})

            # 7. 获取投资建议
            # 传递技术指标和新闻数据给get_recommendation函数
            tech_data = {
                'RSI': technical_summary['rsi_level'],
                'MACD_signal': technical_summary['macd_signal'],
                'Volatility': df.iloc[-1]['Volatility']
            }
            recommendation = self.get_recommendation(score, market_type, tech_data, news_data)

            # 8. 构建更全面的prompt
            prompt = f"""作为专业的股票分析师，请对{stock_name}({stock_code})进行全面分析:

    1. 基本信息:
       - 股票名称: {stock_name}
       - 股票代码: {stock_code}
       - 行业: {industry}
       - 市场类型: {"A股" if market_type == 'A' else "港股" if market_type == 'HK' else "美股"}
       - 交易货币: {"美元($)" if market_type == 'US' else "港元(HK$)" if market_type == 'HK' else "人民币(¥)"}

    2. 技术指标摘要:
       - 趋势: {technical_summary['trend']}
       - 波动率: {technical_summary['volatility']}
       - 成交量趋势: {technical_summary['volume_trend']}
       - RSI: {technical_summary['rsi_level']:.2f}
       - MACD信号: {technical_summary['macd_signal']}
       - 布林带位置: {technical_summary['bb_position']}

    3. 支撑与压力位:
       - 短期支撑位: {', '.join([str(level) for level in sr_levels['support_levels']['short_term']])}
       - 中期支撑位: {', '.join([str(level) for level in sr_levels['support_levels']['medium_term']])}
       - 短期压力位: {', '.join([str(level) for level in sr_levels['resistance_levels']['short_term']])}
       - 中期压力位: {', '.join([str(level) for level in sr_levels['resistance_levels']['medium_term']])}

    4. 综合评分: {score_details['total']}分
       - 趋势评分: {score_details.get('trend', 0)}
       - 波动率评分: {score_details.get('volatility', 0)}
       - 技术指标评分: {score_details.get('technical', 0)}
       - 成交量评分: {score_details.get('volume', 0)}
       - 动量评分: {score_details.get('momentum', 0)}

    5. 投资建议: {recommendation['action']} {recommendation['details']}

    6. 近期相关新闻:
    {self._format_news_for_prompt(news_data.get('news', []))}

    7. 公司公告:
    {self._format_announcements_for_prompt(news_data.get('announcements', []))}

    8. 行业动态:
    {self._format_news_for_prompt(news_data.get('industry_news', []))}

    9. 市场情绪: {news_data.get('market_sentiment', 'neutral')}

    请提供以下内容:
    1. 技术面分析 - 详细分析价格走势、支撑压力位、主要技术指标的信号
    2. 行业和市场环境 - 结合新闻和行业动态分析公司所处环境
    3. 风险因素 - 识别潜在风险点
    4. 具体交易策略 - 给出明确的买入/卖出建议，包括入场点、止损位和目标价位
    5. 短期(1周)、中期(1-3个月)和长期(半年)展望

    请基于数据给出客观分析，不要过度乐观或悲观。分析应该包含具体数据和百分比，避免模糊表述。
    """

            messages = [{"role": "user", "content": prompt}]

            # 使用线程和队列添加超时控制
            result_queue = queue.Queue()

            def call_api():
                try:
                    response = openai.ChatCompletion.create(
                        model=self.openai_model,
                        messages=messages,
                        temperature=0.8,
                        max_tokens=4000,
                        stream=False,
                        timeout=180
                    )
                    result_queue.put(response)
                except Exception as e:
                    result_queue.put(e)

            # 启动API调用线程
            api_thread = threading.Thread(target=call_api)
            api_thread.daemon = True
            api_thread.start()

            # 等待结果，最多等待30秒
            try:
                result = result_queue.get(timeout=30)

                # 检查结果是否为异常
                if isinstance(result, Exception):
                    raise result

                # 提取助理回复
                assistant_reply = result["choices"][0]["message"]["content"].strip()
                return assistant_reply

            except queue.Empty:
                return "AI分析超时，无法获取分析结果。请稍后再试。"
            except Exception as e:
                return f"AI分析过程中发生错误: {str(e)}"

        except Exception as e:
            self.logger.error(f"AI分析发生错误: {str(e)}")
            return f"AI分析过程中发生错误，请稍后再试。错误信息: {str(e)}"

    def _calculate_bb_position(self, df):
        """计算价格在布林带中的位置"""
        latest = df.iloc[-1]
        bb_width = latest['BB_upper'] - latest['BB_lower']
        if bb_width == 0:
            return "middle"

        position = (latest['close'] - latest['BB_lower']) / bb_width

        if position < 0.2:
            return "near lower band (potential oversold)"
        elif position < 0.4:
            return "below middle band"
        elif position < 0.6:
            return "near middle band"
        elif position < 0.8:
            return "above middle band"
        else:
            return "near upper band (potential overbought)"

    def _format_news_for_prompt(self, news_list):
        """格式化新闻列表为prompt字符串"""
        if not news_list:
            return "   无最新相关新闻"

        formatted = ""
        for i, news in enumerate(news_list[:3]):  # 最多显示3条
            date = news.get('date', '')
            title = news.get('title', '')
            source = news.get('source', '')
            formatted += f"   {i + 1}. [{date}] {title} (来源: {source})\n"

        return formatted

    def _format_announcements_for_prompt(self, announcements):
        """格式化公告列表为prompt字符串"""
        if not announcements:
            return "   无最新公告"

        formatted = ""
        for i, ann in enumerate(announcements[:3]):  # 最多显示3条
            date = ann.get('date', '')
            title = ann.get('title', '')
            type_ = ann.get('type', '')
            formatted += f"   {i + 1}. [{date}] {title} (类型: {type_})\n"

        return formatted

    def analyze_stock(self, stock_code, market_type='A'):
        """分析单个股票"""
        try:
            # 获取股票数据
            df = self.get_stock_data(stock_code, market_type)
            
            # 计算技术指标
            df = self.calculate_indicators(df)
            
            # 计算评分
            score = self.calculate_score(df, market_type)
            
            # 获取最新数据
            latest = df.iloc[-1]
            
            # 计算价格趋势 - 添加错误处理
            recent_price_change = 0
            try:
                if len(df) >= 6:
                    recent_price_change = (latest['close'] / df.iloc[-6]['close'] - 1) * 100
            except Exception as e:
                self.logger.warning(f"计算价格趋势时出错: {str(e)}")
            
            # 准备技术数据
            technical_data = {
                'RSI': latest['RSI'],
                'MACD_signal': 'bullish' if latest['MACD'] > latest['Signal'] else 'bearish',
                'Volatility': latest['Volatility'],
                'BB_position': (latest['close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']),
                'price_trend': recent_price_change  # 价格趋势数据
            }
            
            # 获取新闻数据
            news_data = None
            try:
                news = self.get_stock_news(stock_code, market_type)
                if news and isinstance(news, list) and len(news) > 0:
                    # 简单情感分析
                    sentiment_score = sum([n.get('sentiment', 0) for n in news]) / len(news)
                    news_data = {
                        'news': news,
                        'market_sentiment': 'bullish' if sentiment_score > 0.3 else 
                                           'bearish' if sentiment_score < -0.3 else 'neutral'
                    }
            except Exception as e:
                self.logger.warning(f"获取新闻数据时出错: {str(e)}")
            
            # 获取投资建议
            recommendation = self.get_recommendation(score, market_type, technical_data, news_data)
            
            # 识别支撑位和阻力位
            support_resistance = self.identify_support_resistance(df)
            
            # 获取AI分析
            ai_analysis = None
            try:
                ai_analysis = self.get_ai_analysis(df, stock_code, market_type)
            except Exception as e:
                self.logger.error(f"获取AI分析时出错: {str(e)}")
                ai_analysis = "无法获取AI分析"
            
            # 构建分析报告
            report = {
                'stock_code': stock_code,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'price': float(latest['close']),
                'score': score,
                'score_details': self.score_details,
                'recommendation': recommendation,
                'technical_indicators': {
                    'MA5': float(latest['MA5']),
                    'MA20': float(latest['MA20']),
                    'MA60': float(latest['MA60']),
                    'RSI': float(latest['RSI']),
                    'MACD': float(latest['MACD']),
                    'Signal': float(latest['Signal']),
                    'MACD_hist': float(latest['MACD_hist']),
                    'Volatility': float(latest['Volatility']),
                    'Volume_Ratio': float(latest['Volume_Ratio']),
                    'BB_upper': float(latest['BB_upper']),
                    'BB_middle': float(latest['BB_middle']),
                    'BB_lower': float(latest['BB_lower'])
                },
                'support_resistance': support_resistance,
                'ai_analysis': ai_analysis
            }
            
            # 验证并修复报告中的无效值
            report = self._validate_and_fix_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"分析股票 {stock_code} 时出错: {str(e)}")
            raise

    def scan_market(self, stock_list, min_score=60, market_type='A'):
        """扫描市场，寻找符合条件的股票"""
        try:
            results = []
            recommendations = []
            
            # 调整最低评分阈值，使其与新的评分系统匹配
            adjusted_min_score = min_score
            if min_score > 80:  # 如果用户设置了很高的阈值，适当调整
                adjusted_min_score = 75
            
            for stock_code in stock_list:
                try:
                    # 快速分析股票
                    report = self.quick_analyze_stock(stock_code, market_type)
                    
                    # 检查评分是否达到最低要求
                    if report['score'] >= adjusted_min_score:
                        results.append(report)
                        
                        # 添加到推荐列表
                        recommendations.append({
                            'stock_code': stock_code,
                            'stock_name': report.get('stock_name', '未知'),
                            'score': report['score'],
                            'price': report['price'],
                            'recommendation': report['recommendation']
                        })
                except Exception as e:
                    self.logger.error(f"扫描股票 {stock_code} 时出错: {str(e)}")
                    continue
            
            # 按评分排序
            recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
            
            return recommendations
        except Exception as e:
            self.logger.error(f"市场扫描时出错: {str(e)}")
            raise

    def quick_analyze_stock(self, stock_code, market_type='A'):
        """快速分析股票，用于市场扫描"""
        try:
            # 获取股票数据
            df = self.get_stock_data(stock_code, market_type)

            # 计算技术指标
            df = self.calculate_indicators(df)

            # 计算评分
            score = self.calculate_score(df, market_type)

            # 获取最新数据
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # 计算价格趋势 - 添加错误处理
            recent_price_change = 0
            try:
                if len(df) >= 6:
                    recent_price_change = (latest['close'] / df.iloc[-6]['close'] - 1) * 100
            except Exception as e:
                self.logger.warning(f"计算价格趋势时出错: {str(e)}")

            # 准备技术数据
            technical_data = {
                'RSI': latest['RSI'],
                'MACD_signal': 'bullish' if latest['MACD'] > latest['Signal'] else 'bearish',
                'Volatility': latest['Volatility'],
                'price_trend': recent_price_change
            }

            # 尝试获取股票名称和行业
            stock_name = '未知'
            industry = '未知'
            
            try:
                # 美股特殊处理
                if market_type == 'US':
                    # 提前设置默认值以防获取失败
                    stock_name = f"{stock_code} (US)"
                    industry = "美股"
                    
                    # 尝试获取详细信息
                    stock_info = self.get_stock_info(stock_code)
                    if stock_info and '股票名称' in stock_info and stock_info['股票名称'] != '未知':
                        stock_name = stock_info['股票名称']
                        # 确保美股名称包含(US)标识
                        if not stock_name.endswith('(US)'):
                            stock_name += ' (US)'
                    if stock_info and '行业' in stock_info and stock_info['行业'] != '未知':
                        industry = stock_info['行业']
                else:
                    # A股和港股处理
                    stock_info = self.get_stock_info(stock_code)
                    stock_name = stock_info.get('股票名称', '未知')
                    industry = stock_info.get('行业', '未知')
            except Exception as e:
                self.logger.warning(f"获取股票信息时出错: {str(e)}")
                # 对于美股，确保至少有一个默认名称
                if market_type == 'US':
                    stock_name = f"{stock_code} (US)"
                    industry = "美股"

            # 获取投资建议
            recommendation = self.get_recommendation(score, market_type, technical_data)

            # 生成简化报告
            report = {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'industry': industry,
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'score': score,
                'price': float(latest['close']),
                'price_change': float((latest['close'] - prev['close']) / prev['close'] * 100),
                'ma_trend': 'UP' if latest['MA5'] > latest['MA20'] else 'DOWN',
                'rsi': float(latest['RSI']),
                'macd_signal': 'BUY' if latest['MACD'] > latest['Signal'] else 'SELL',
                'volume_status': '放量' if latest['Volume_Ratio'] > 1.5 else '平量',
                'recommendation': recommendation['action']  # 只使用recommendation的action字段
            }

            return report
        except Exception as e:
            self.logger.error(f"快速分析股票 {stock_code} 时出错: {str(e)}")
            raise

    def get_stock_info(self, stock_code):
        """获取股票基本信息"""
        import akshare as ak

        cache_key = f"{stock_code}_info"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        try:
            # 检查是否是美股代码 (通常包含字母或者以105.开头)
            is_us_stock = False
            if stock_code.startswith('105.') or stock_code.startswith('106.') or (any(c.isalpha() for c in stock_code) and not stock_code.startswith('0') and not stock_code.startswith('3') and not stock_code.startswith('6')):
                is_us_stock = True
                self.logger.info(f"检测到美股代码: {stock_code}")
            
            info_dict = {}
            
            if is_us_stock:
                # 美股特殊处理
                try:
                    # 去掉可能的105.前缀
                    clean_code = stock_code.replace('105.', '').replace('106.', '')
                    stock_info = StockInfoDAO.find_by_code('US', clean_code.upper())

                    info_dict['股票名称'] = stock_info.stock_name
                    info_dict['行业'] = stock_info.industry
                    info_dict['地区'] = "美国"
                except Exception as us_e:
                    self.logger.error(f"获取美股信息时出错: {str(us_e)}")
                    # 提供美股默认信息，避免出错
                    info_dict['股票名称'] = f"{stock_code} (US)"
                    info_dict['行业'] = "美股"
                    info_dict['地区'] = "美国"
            else:
                # A股信息获取
                try:
                    # 获取A股股票基本信息
                    stock_info = ak.stock_individual_info_em(symbol=stock_code)

                    # 使用列名而不是索引访问数据
                    for _, row in stock_info.iterrows():
                        # 使用iloc安全地获取数据
                        if len(row) >= 2:  # 确保有至少两列
                            info_dict[row.iloc[0]] = row.iloc[1]

                    # 获取股票名称
                    try:
                        stock_name = ak.stock_info_a_code_name()

                        # 检查数据框是否包含预期的列
                        if '代码' in stock_name.columns and '名称' in stock_name.columns:
                            # 尝试找到匹配的股票代码
                            matched_stocks = stock_name[stock_name['代码'] == stock_code]
                            if not matched_stocks.empty:
                                name = matched_stocks['名称'].values[0]
                            else:
                                self.logger.warning(f"未找到股票代码 {stock_code} 的名称信息")
                                name = "未知"
                        else:
                            # 尝试使用不同的列名
                            possible_code_columns = ['代码', 'code', 'symbol', '股票代码', 'stock_code']
                            possible_name_columns = ['名称', 'name', '股票名称', 'stock_name']

                            code_col = next((col for col in possible_code_columns if col in stock_name.columns), None)
                            name_col = next((col for col in possible_name_columns if col in stock_name.columns), None)

                            if code_col and name_col:
                                matched_stocks = stock_name[stock_name[code_col] == stock_code]
                                if not matched_stocks.empty:
                                    name = matched_stocks[name_col].values[0]
                                else:
                                    name = "未知"
                            else:
                                self.logger.warning(f"股票信息DataFrame结构不符合预期: {stock_name.columns.tolist()}")
                                name = "未知"
                    except Exception as e:
                        self.logger.error(f"获取股票名称时出错: {str(e)}")
                        name = "未知"

                    info_dict['股票名称'] = name
                except Exception as a_e:
                    self.logger.error(f"获取A股信息时出错: {str(a_e)}")
                    # 提供默认信息
                    info_dict['股票名称'] = "未知"
                    info_dict['行业'] = "未知"
                    info_dict['地区'] = "未知"

            # 确保基本字段存在
            if '行业' not in info_dict:
                info_dict['行业'] = "未知"
            if '地区' not in info_dict:
                info_dict['地区'] = "未知"

            # 增加更多日志来调试问题
            self.logger.info(f"获取到股票信息: 名称={info_dict.get('股票名称', '未知')}, 行业={info_dict.get('行业', '未知')}")

            self.data_cache[cache_key] = info_dict
            return info_dict
        except Exception as e:
            self.logger.error(f"获取股票信息失败: {str(e)}")
            return {"股票名称": "未知", "行业": "未知", "地区": "未知"}

    def identify_support_resistance(self, df):
        """识别支撑位和压力位"""
        latest_price = df['close'].iloc[-1]

        # 使用布林带作为支撑压力参考
        support_levels = [df['BB_lower'].iloc[-1]]
        resistance_levels = [df['BB_upper'].iloc[-1]]

        # 添加主要均线作为支撑压力
        if latest_price < df['MA5'].iloc[-1]:
            resistance_levels.append(df['MA5'].iloc[-1])
        else:
            support_levels.append(df['MA5'].iloc[-1])

        if latest_price < df['MA20'].iloc[-1]:
            resistance_levels.append(df['MA20'].iloc[-1])
        else:
            support_levels.append(df['MA20'].iloc[-1])

        # 添加整数关口
        price_digits = len(str(int(latest_price)))
        base = 10 ** (price_digits - 1)

        lower_integer = math.floor(latest_price / base) * base
        upper_integer = math.ceil(latest_price / base) * base

        if lower_integer < latest_price:
            support_levels.append(lower_integer)
        if upper_integer > latest_price:
            resistance_levels.append(upper_integer)

        # 排序并格式化
        support_levels = sorted(set([round(x, 2) for x in support_levels if x < latest_price]), reverse=True)
        resistance_levels = sorted(set([round(x, 2) for x in resistance_levels if x > latest_price]))

        # 分类为短期和中期
        short_term_support = support_levels[:1] if support_levels else []
        medium_term_support = support_levels[1:2] if len(support_levels) > 1 else []
        short_term_resistance = resistance_levels[:1] if resistance_levels else []
        medium_term_resistance = resistance_levels[1:2] if len(resistance_levels) > 1 else []

        return {
            'support_levels': {
                'short_term': short_term_support,
                'medium_term': medium_term_support
            },
            'resistance_levels': {
                'short_term': short_term_resistance,
                'medium_term': medium_term_resistance
            }
        }

    def calculate_technical_score(self, df):
        """计算技术面评分 (0-40分，标准化为0-100分)"""
        try:
            score = 0
            # 确保有足够的数据
            if len(df) < 2:
                self.logger.warning("数据不足，无法计算技术面评分")
                return {'total': 0, 'trend': 0, 'indicators': 0, 'support_resistance': 0, 'volatility_volume': 0}

            latest = df.iloc[-1]
            prev = df.iloc[-2]  # 获取前一个时间点的数据
            prev_close = prev['close']

            # 1. 趋势分析 (0-10分)
            trend_score = 0

            # 均线排列情况
            if latest['MA5'] > latest['MA20'] > latest['MA60']:  # 多头排列
                trend_score += 5
            elif latest['MA5'] < latest['MA20'] < latest['MA60']:  # 空头排列
                trend_score = 0
            else:  # 交叉状态
                if latest['MA5'] > latest['MA20']:
                    trend_score += 3
                if latest['MA20'] > latest['MA60']:
                    trend_score += 2

            # 价格与均线关系
            if latest['close'] > latest['MA5']:
                trend_score += 3
            elif latest['close'] > latest['MA20']:
                trend_score += 2

            # 限制最大值
            trend_score = min(trend_score, 10)
            score += trend_score

            # 2. 技术指标分析 (0-10分)
            indicator_score = 0

            # RSI
            if 40 <= latest['RSI'] <= 60:  # 中性
                indicator_score += 2
            elif 30 <= latest['RSI'] < 40 or 60 < latest['RSI'] <= 70:  # 边缘区域
                indicator_score += 4
            elif latest['RSI'] < 30:  # 超卖
                indicator_score += 5
            elif latest['RSI'] > 70:  # 超买
                indicator_score += 0

            # MACD
            if latest['MACD'] > latest['Signal']:  # MACD金叉或在零轴上方
                indicator_score += 3
            else:
                # 修复：比较当前和前一个时间点的MACD柱状图值
                if latest['MACD_hist'] > prev['MACD_hist']:  # 柱状图上升
                    indicator_score += 1

            # 限制最大值和最小值
            indicator_score = max(0, min(indicator_score, 10))
            score += indicator_score

            # 3. 支撑压力位分析 (0-10分)
            sr_score = 0

            # 识别支撑位和压力位
            middle_price = latest['close']
            upper_band = latest['BB_upper']
            lower_band = latest['BB_lower']

            # 距离布林带上下轨的距离
            upper_distance = (upper_band - middle_price) / middle_price * 100
            lower_distance = (middle_price - lower_band) / middle_price * 100

            if lower_distance < 2:  # 接近下轨
                sr_score += 5
            elif lower_distance < 5:
                sr_score += 3

            if upper_distance > 5:  # 距上轨较远
                sr_score += 5
            elif upper_distance > 2:
                sr_score += 2

            # 限制最大值
            sr_score = min(sr_score, 10)
            score += sr_score

            # 4. 波动性和成交量分析 (0-10分)
            vol_score = 0

            # 波动率分析
            if latest['Volatility'] < 2:  # 低波动率
                vol_score += 3
            elif latest['Volatility'] < 4:  # 中等波动率
                vol_score += 2

            # 成交量分析
            if 'Volume_Ratio' in df.columns:
                if latest['Volume_Ratio'] > 1.5 and latest['close'] > prev_close:  # 放量上涨
                    vol_score += 4
                elif latest['Volume_Ratio'] < 0.8 and latest['close'] < prev_close:  # 缩量下跌
                    vol_score += 3
                elif latest['Volume_Ratio'] > 1 and latest['close'] > prev_close:  # 普通放量上涨
                    vol_score += 2

            # 限制最大值
            vol_score = min(vol_score, 10)
            score += vol_score

            # 将评分标准化为0-100的范围 (原评分最高40分)
            normalized_score = int(score * 2.5)  # 将40分制转换为100分制
            
            # 保存各个维度的分数 (也同样标准化)
            technical_scores = {
                'total': normalized_score,
                'trend': int(trend_score * 2.5),
                'indicators': int(indicator_score * 2.5),
                'support_resistance': int(sr_score * 2.5),
                'volatility_volume': int(vol_score * 2.5)
            }

            return technical_scores

        except Exception as e:
            self.logger.error(f"计算技术面评分时出错: {str(e)}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return {'total': 0, 'trend': 0, 'indicators': 0, 'support_resistance': 0, 'volatility_volume': 0}

    def perform_enhanced_analysis(self, stock_code, market_type='A'):
        """执行增强版分析"""
        try:
            # 记录开始时间，便于性能分析
            start_time = time.time()
            self.logger.info(f"开始执行股票 {stock_code} 的增强分析")

            # 获取股票数据
            df = self.get_stock_data(stock_code, market_type)
            data_time = time.time()
            self.logger.info(f"获取股票数据耗时: {data_time - start_time:.2f}秒")

            # 计算技术指标
            df = self.calculate_indicators(df)
            indicator_time = time.time()
            self.logger.info(f"计算技术指标耗时: {indicator_time - data_time:.2f}秒")

            # 获取最新数据
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest

            # 获取支撑压力位
            sr_levels = self.identify_support_resistance(df)

            # 计算技术面评分
            technical_score = self.calculate_technical_score(df)

            # 获取股票信息
            stock_info = self.get_stock_info(stock_code)

            # 确保technical_score包含必要的字段
            if 'total' not in technical_score:
                technical_score['total'] = 0

            # 生成增强版报告
            enhanced_report = {
                'basic_info': {
                    'stock_code': stock_code,
                    'stock_name': stock_info.get('股票名称', '未知'),
                    'industry': stock_info.get('行业', '未知'),
                    'market_type': market_type,  # 添加市场类型
                    'analysis_date': datetime.now().strftime('%Y-%m-%d')
                },
                'price_data': {
                    'current_price': float(latest['close']),  # 确保是Python原生类型
                    'price_change': float((latest['close'] - prev['close']) / prev['close'] * 100),
                    'price_change_value': float(latest['close'] - prev['close'])
                },
                'technical_analysis': {
                    'trend': {
                        'ma_trend': 'UP' if latest['MA5'] > latest['MA20'] else 'DOWN',
                        'ma_status': "多头排列" if latest['MA5'] > latest['MA20'] > latest['MA60'] else
                        "空头排列" if latest['MA5'] < latest['MA20'] < latest['MA60'] else
                        "交叉状态",
                        'ma_values': {
                            'ma5': float(latest['MA5']),
                            'ma20': float(latest['MA20']),
                            'ma60': float(latest['MA60'])
                        }
                    },
                    'indicators': {
                        # 确保所有指标都存在并是原生类型
                        'rsi': float(latest['RSI']) if 'RSI' in latest else 50.0,
                        'macd': float(latest['MACD']) if 'MACD' in latest else 0.0,
                        'macd_signal': float(latest['Signal']) if 'Signal' in latest else 0.0,
                        'macd_histogram': float(latest['MACD_hist']) if 'MACD_hist' in latest else 0.0,
                        'volatility': float(latest['Volatility']) if 'Volatility' in latest else 0.0
                    },
                    'volume': {
                        'current_volume': float(latest['volume']) if 'volume' in latest else 0.0,
                        'volume_ratio': float(latest['Volume_Ratio']) if 'Volume_Ratio' in latest else 1.0,
                        'volume_status': '放量' if 'Volume_Ratio' in latest and latest['Volume_Ratio'] > 1.5 else '平量'
                    },
                    'support_resistance': sr_levels
                },
                'scores': technical_score,
                'recommendation': {
                    'action': self.get_recommendation(technical_score['total'], market_type)['action'],
                    'key_points': []
                },
                'ai_analysis': self.get_ai_analysis(df, stock_code, market_type),
                'support_resistance': self.identify_support_resistance(df)
            }

            # 最后检查并修复报告结构
            self._validate_and_fix_report(enhanced_report)

            # 在函数结束时记录总耗时
            end_time = time.time()
            self.logger.info(f"执行增强分析总耗时: {end_time - start_time:.2f}秒")

            return enhanced_report

        except Exception as e:
            self.logger.error(f"执行增强版分析时出错: {str(e)}")
            self.logger.error(traceback.format_exc())

            # 返回基础错误报告
            return {
                'basic_info': {
                    'stock_code': stock_code,
                    'stock_name': '分析失败',
                    'industry': '未知',
                    'market_type': market_type,
                    'analysis_date': datetime.now().strftime('%Y-%m-%d')
                },
                'price_data': {
                    'current_price': 0.0,
                    'price_change': 0.0,
                    'price_change_value': 0.0
                },
                'technical_analysis': {
                    'trend': {
                        'ma_trend': 'UNKNOWN',
                        'ma_status': '未知',
                        'ma_values': {'ma5': 0.0, 'ma20': 0.0, 'ma60': 0.0}
                    },
                    'indicators': {
                        'rsi': 50.0,
                        'macd': 0.0,
                        'macd_signal': 0.0,
                        'macd_histogram': 0.0,
                        'volatility': 0.0
                    },
                    'volume': {
                        'current_volume': 0.0,
                        'volume_ratio': 0.0,
                        'volume_status': 'NORMAL'
                    },
                    'support_resistance': {
                        'support_levels': {'short_term': [], 'medium_term': []},
                        'resistance_levels': {'short_term': [], 'medium_term': []}
                    }
                },
                'scores': {'total': 0},
                'recommendation': {'action': '分析出错，无法提供建议'},
                'ai_analysis': f"分析过程中出错: {str(e)}",
                'support_resistance': {
                    'support_levels': {'short_term': [], 'medium_term': []},
                    'resistance_levels': {'short_term': [], 'medium_term': []}
                }
            }

            return error_report

    # 添加一个辅助方法确保报告结构完整
    def _validate_and_fix_report(self, report):
        """确保分析报告结构完整"""
        # 检查必要的顶级字段
        required_sections = ['basic_info', 'price_data', 'technical_analysis', 'scores', 'recommendation',
                             'ai_analysis']
        for section in required_sections:
            if section not in report:
                self.logger.warning(f"报告缺少 {section} 部分，添加空对象")
                report[section] = {}

        # 检查technical_analysis的结构
        if 'technical_analysis' in report:
            tech = report['technical_analysis']
            if not isinstance(tech, dict):
                report['technical_analysis'] = {}
                tech = report['technical_analysis']

            # 检查indicators部分
            if 'indicators' not in tech or not isinstance(tech['indicators'], dict):
                tech['indicators'] = {
                    'rsi': 50.0,
                    'macd': 0.0,
                    'macd_signal': 0.0,
                    'macd_histogram': 0.0,
                    'volatility': 0.0
                }

            # 转换所有指标为原生Python类型
            for key, value in tech['indicators'].items():
                try:
                    tech['indicators'][key] = float(value)
                except (TypeError, ValueError):
                    tech['indicators'][key] = 0.0