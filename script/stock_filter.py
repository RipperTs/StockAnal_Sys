import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from db.stock_info_dao import StockInfoDAO


def filter_by_pe_ratio(min_pe=5, max_pe=50, market_type="US"):
    """
    根据市盈率筛选股票
    
    Args:
        min_pe (float): 最小市盈率
        max_pe (float): 最大市盈率
        market_type (str): 市场类型，默认为美股
        
    Returns:
        list: 符合条件的股票信息列表
    """
    stocks = StockInfoDAO.find_by_market_type(market_type)
    filtered_stocks = []
    
    for stock in stocks:
        try:
            pe_ratio = float(stock.pe_ratio)
            if min_pe <= pe_ratio <= max_pe:
                filtered_stocks.append(stock)
        except (ValueError, TypeError):
            # 跳过无效的市盈率数据
            continue
    
    return filtered_stocks


def get_stock_history(stock_code, period="5y"):
    """
    使用yfinance获取股票的历史价格和交易量数据
    
    Args:
        stock_code (str): 股票代码
        period (str): 获取数据的时间范围，默认为5年
            可选值: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        
    Returns:
        pandas.DataFrame: 股票历史数据
    """
    try:
        import yfinance as yf
        
        # 可能需要处理一下股票代码格式
        # 美股代码通常不需要额外处理
        symbol = stock_code
        
        # 创建Ticker对象
        ticker = yf.Ticker(symbol)
        
        # 配置获取历史数据的参数
        kwargs = {
            'period': period,      # 时间范围
            'interval': '1d',      # 数据粒度：日数据
            'auto_adjust': True,   # 自动调整价格
            'actions': False       # 不包含分红和拆分信息
        }
        
        # 获取历史数据
        history = ticker.history(**kwargs)
        
        # 检查是否成功获取到数据
        if history is None or history.empty:
            print(f"没有获取到股票 {stock_code} 的历史数据")
            return None
            
        # 确保包含必要的列
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in history.columns]
        if missing_columns:
            print(f"股票 {stock_code} 数据中缺少以下列: {missing_columns}")
            return None
            
        # 检查数据是否足够
        if len(history) < 20:  # 至少需要20个交易日的数据
            print(f"股票 {stock_code} 的历史数据不足 (仅有 {len(history)} 个交易日)")
            return None
            
        return history
        
    except ImportError:
        print("请先安装yfinance库: pip install yfinance")
        return None
    except Exception as e:
        print(f"获取股票 {stock_code} 的历史数据时出错: {e}")
        return None


def filter_by_volume_surge(stock_info, min_volume_ratio=4.0):
    """
    策略1: 历史底部20%上下附近忽然放量，日交易量比前一日放大4倍以上
    
    Args:
        stock_info (StockInfo): 股票信息
        min_volume_ratio (float): 最小成交量比率
        
    Returns:
        bool: 是否符合条件
    """
    # 获取完整历史数据
    history = get_stock_history(stock_info.stock_code)
    if history is None or len(history) < 60:  # 至少需要60个交易日的数据
        return False
    
    try:
        # 计算价格的历史百分位
        prices = history['Close'].values
        
        # 检查价格数据是否有效
        if len(prices) == 0 or np.isnan(prices).any():
            return False
            
        min_price = np.min(prices)
        max_price = np.max(prices)
        price_range = max_price - min_price
        
        if price_range == 0:  # 避免除以零
            return False
        
        # 计算每个价格点的百分位
        percentiles = [(price - min_price) / price_range for price in prices]
        
        # 检查成交量变化
        volumes = history['Volume'].values
        
        # 检查成交量数据是否有效
        if len(volumes) == 0 or np.isnan(volumes).any():
            return False
            
        # 只关注最近一年的交易日进行放量判断
        recent_trading_days = min(252, len(history))  # 约一年的交易日
        
        for i in range(1, recent_trading_days):
            # 检查是否在历史底部20%附近
            if percentiles[-i] <= 0.2:  # 从最近的数据开始检查
                # 检查成交量是否比前一日放大4倍以上
                volume_ratio = volumes[-i] / volumes[-i-1] if volumes[-i-1] > 0 else 0
                if volume_ratio >= min_volume_ratio:
                    return True
        
        return False
    except Exception as e:
        print(f"分析股票 {stock_info.stock_code} 的成交量变化时出错: {e}")
        return False


def filter_by_pullback_50_percent(stock_info):
    """
    策略2: 起涨翻倍后最高点下来50%
    
    Args:
        stock_info (StockInfo): 股票信息
        
    Returns:
        bool: 是否符合条件
    """
    # 获取完整历史数据
    history = get_stock_history(stock_info.stock_code)
    if history is None or len(history) < 120:  # 至少需要120个交易日的数据
        return False
    
    try:
        prices = history['Close'].values
        
        # 检查价格数据是否有效
        if len(prices) == 0 or np.isnan(prices).any():
            return False
        
        # 找到历史上所有的主要低点
        low_points = []
        for i in range(1, len(prices) - 1):
            if prices[i-1] > prices[i] < prices[i+1]:
                low_points.append((i, prices[i]))
        
        # 按价格从低到高排序
        low_points.sort(key=lambda x: x[1])
        
        # 检查每个低点是否满足条件
        for start_idx, start_price in low_points[:3]:  # 考虑3个最低点
            # 从低点寻找随后的最高点
            max_idx = start_idx
            max_price = start_price
            
            for i in range(start_idx + 1, len(prices)):
                if prices[i] > max_price:
                    max_price = prices[i]
                    max_idx = i
            
            # 如果价格已经翻倍
            if max_price >= start_price * 2:
                # 检查最高点之后是否有回调50%
                for i in range(max_idx + 1, len(prices)):
                    pullback = (max_price - prices[i]) / (max_price - start_price)
                    if pullback >= 0.5:  # 回调幅度超过50%
                        return True
        
        return False
    except Exception as e:
        print(f"分析股票 {stock_info.stock_code} 的价格回调时出错: {e}")
        return False


def filter_by_initial_pullback_20_percent(stock_info):
    """
    策略3: 底部起涨初升段回测20%
    
    Args:
        stock_info (StockInfo): 股票信息
        
    Returns:
        bool: 是否符合条件
    """
    # 获取完整历史数据
    history = get_stock_history(stock_info.stock_code)
    if history is None or len(history) < 60:  # 至少需要60个交易日的数据
        return False
    
    try:
        prices = history['Close'].values
        
        # 检查价格数据是否有效
        if len(prices) == 0 or np.isnan(prices).any():
            return False
        
        # 寻找历史底部
        min_idx = np.argmin(prices)
        min_price = prices[min_idx]
        
        # 如果底部是最近的价格，不符合我们的策略
        if min_idx >= len(prices) - 20:
            return False
        
        # 寻找底部之后的初升段高点
        max_idx = min_idx
        max_price = min_price
        
        # 在底部后的30个交易日内寻找初升段高点
        search_end = min(min_idx + 30, len(prices))
        for i in range(min_idx + 1, search_end):
            if prices[i] > max_price:
                max_price = prices[i]
                max_idx = i
        
        # 初升段幅度
        initial_rise = max_price - min_price
        if initial_rise <= 0:
            return False
        
        # 检查之后是否有回测20%
        for i in range(max_idx + 1, len(prices)):
            pullback = (max_price - prices[i]) / initial_rise
            if 0.18 <= pullback <= 0.22:  # 回测幅度接近20%
                return True
        
        return False
    except Exception as e:
        print(f"分析股票 {stock_info.stock_code} 的初升段回测时出错: {e}")
        return False


def filter_stocks(min_pe=5, max_pe=50, market_type="US", strategies=None):
    """
    根据多种策略筛选股票
    
    Args:
        min_pe (float): 最小市盈率
        max_pe (float): 最大市盈率
        market_type (str): 市场类型
        strategies (list): 筛选策略列表，默认为全部策略
        
    Returns:
        dict: 按策略分类的股票列表
    """
    if strategies is None:
        strategies = ["volume_surge", "pullback_50", "initial_pullback_20"]
    
    # 首先按市盈率筛选
    pe_filtered_stocks = filter_by_pe_ratio(min_pe, max_pe, market_type)
    print(f"市盈率筛选后的股票数量: {len(pe_filtered_stocks)}")
    
    result = {
        "volume_surge": [],
        "pullback_50": [],
        "initial_pullback_20": [],
        "all_strategies": []
    }
    
    total_stocks = len(pe_filtered_stocks)
    for i, stock in enumerate(pe_filtered_stocks):
        print(f"处理进度: {i+1}/{total_stocks} - 当前股票: {stock.stock_code} ({stock.stock_name})")
        
        matched_strategies = []
        
        if "volume_surge" in strategies and filter_by_volume_surge(stock):
            result["volume_surge"].append(stock)
            matched_strategies.append("volume_surge")
        
        if "pullback_50" in strategies and filter_by_pullback_50_percent(stock):
            result["pullback_50"].append(stock)
            matched_strategies.append("pullback_50")
        
        if "initial_pullback_20" in strategies and filter_by_initial_pullback_20_percent(stock):
            result["initial_pullback_20"].append(stock)
            matched_strategies.append("initial_pullback_20")
        
        # 如果符合所有指定的策略，添加到all_strategies列表
        if len(matched_strategies) == len(strategies):
            result["all_strategies"].append(stock)
    
    return result


if __name__ == "__main__":
    # 确保已安装yfinance库
    try:
        import yfinance
    except ImportError:
        print("请先安装yfinance库: pip install yfinance")
        sys.exit(1)
    
    print("开始筛选股票...")
    
    # 测试单个股票的数据获取 - 用于调试
    test_mode = False
    if test_mode:
        try:
            # 使用苹果股票作为测试
            test_stock_code = "AAPL"
            print(f"测试获取 {test_stock_code} 的历史数据...")
            history = get_stock_history(test_stock_code)
            if history is not None:
                print(f"成功获取到 {len(history)} 条历史数据")
                print(history.head())
                
                # 显示历史最高价和最低价
                min_price = history['Close'].min()
                min_date = history['Close'].idxmin().strftime('%Y-%m-%d')
                max_price = history['Close'].max()
                max_date = history['Close'].idxmax().strftime('%Y-%m-%d')
                
                print(f"历史最低收盘价: ${min_price:.2f} ({min_date})")
                print(f"历史最高收盘价: ${max_price:.2f} ({max_date})")
                print(f"历史价格范围: ${min_price:.2f} - ${max_price:.2f}")
            else:
                print("获取历史数据失败")
            sys.exit(0)
        except Exception as e:
            print(f"测试时出错: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # 筛选符合所有策略的股票
    try:
        filtered_stocks = filter_stocks(min_pe=5, max_pe=15)
        
        # 打印筛选结果
        print("\n符合策略1(历史底部放量)的股票:")
        for stock in filtered_stocks["volume_surge"]:
            print(f"{stock.stock_code} ({stock.stock_name}) - PE: {stock.pe_ratio}")
        
        print("\n符合策略2(翻倍后回调50%)的股票:")
        for stock in filtered_stocks["pullback_50"]:
            print(f"{stock.stock_code} ({stock.stock_name}) - PE: {stock.pe_ratio}")
        
        print("\n符合策略3(初升段回测20%)的股票:")
        for stock in filtered_stocks["initial_pullback_20"]:
            print(f"{stock.stock_code} ({stock.stock_name}) - PE: {stock.pe_ratio}")
        
        print("\n符合所有策略的股票:")
        for stock in filtered_stocks["all_strategies"]:
            print(f"{stock.stock_code} ({stock.stock_name}) - PE: {stock.pe_ratio}")
    except Exception as e:
        print(f"筛选股票时出错: {e}")
        import traceback
        traceback.print_exc()
