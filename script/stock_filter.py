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


def filter_by_volume_surge(stock_info, min_volume_ratio=2.5, min_price=5.0):
    """
    策略1: 历史底部区域附近出现持续放量
    检测特征:
    1. 处于历史底部区域(30%以内)
    2. 最近两周(10个交易日)出现明显持续放量
    3. 股价形态开始走稳或有小幅上涨趋势
    4. 当前股价大于5美元
    
    Args:
        stock_info (StockInfo): 股票信息
        min_volume_ratio (float): 放量倍数(相对于之前均值)
        min_price (float): 最小股价(美元)
        
    Returns:
        bool: 是否符合条件
    """
    # 获取历史数据(5年)用于确定历史底部
    history = get_stock_history(stock_info.stock_code)
    if history is None or len(history) < 60:  # 至少需要60个交易日的数据
        return False
    
    try:
        # 计算价格的历史百分位
        prices = history['Close'].values
        
        # 检查价格数据是否有效
        if len(prices) == 0 or np.isnan(prices).any():
            return False
        
        # 检查当前股价是否大于最小要求
        current_price = prices[-1]
        if current_price < min_price:
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
        
        # 只关注最近两周的交易日(约10个交易日)
        lookback_days = min(10, len(history))
        
        # 检查当前是否处于历史底部区域(30%以内)
        if percentiles[-1] > 0.3:
            return False
        
        # 计算最近10个交易日的平均成交量
        recent_avg_volume = np.mean(volumes[-lookback_days:])
        # 计算之前30个交易日的平均成交量
        previous_avg_volume = np.mean(volumes[-40:-lookback_days])
        
        # 检查近期平均成交量是否显著高于之前
        volume_increase_ratio = recent_avg_volume / previous_avg_volume if previous_avg_volume > 0 else 0
        
        # 检查是否有连续放量
        high_volume_days = 0
        for i in range(1, lookback_days+1):  # 检查最近10天
            if volumes[-i] > previous_avg_volume * min_volume_ratio:
                high_volume_days += 1
                
        # 计算最近价格稳定性和趋势
        recent_prices = prices[-lookback_days:]
        price_std = np.std(recent_prices) / np.mean(recent_prices)  # 价格波动率
        price_trend = prices[-1] / prices[-lookback_days] - 1 if prices[-lookback_days] > 0 else 0
        
        # 判断条件 (三选一):
        # 1. 最近平均成交量是之前的1.8倍以上
        # 2. 10天内有2天以上成交量是之前均值的2.5倍以上
        # 3. 价格处于底部区域且有任何明显放量(单日2倍)，同时价格开始走稳
        condition1 = volume_increase_ratio >= 1.8
        condition2 = high_volume_days >= 2
        condition3 = (percentiles[-1] <= 0.2 and max(volumes[-lookback_days:]) > previous_avg_volume * 2.0 and 
                      price_std < 0.05 and price_trend >= -0.02)  # 非常低的位置有放量且价格稳定
        
        if condition1 or condition2 or condition3:
            return True
                    
        return False
    except Exception as e:
        print(f"分析股票 {stock_info.stock_code} 的成交量变化时出错: {e}")
        return False


def filter_by_pullback_50_percent(stock_info):
    """
    策略2: 起涨翻倍后最高点下来50%
    重点关注近期6个月内完成的回调
    
    Args:
        stock_info (StockInfo): 股票信息
        
    Returns:
        bool: 是否符合条件
    """
    # 获取历史数据(5年)
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
                        # 确保回调完成点在最近6个月内
                        if i >= (len(prices) - 126):  # 约6个月的交易日
                            return True
        
        return False
    except Exception as e:
        print(f"分析股票 {stock_info.stock_code} 的价格回调时出错: {e}")
        return False


def filter_by_initial_pullback_20_percent(stock_info):
    """
    策略3: 底部起涨初升段回测20%
    重点关注近期完成的回测
    
    Args:
        stock_info (StockInfo): 股票信息
        
    Returns:
        bool: 是否符合条件
    """
    # 获取历史数据(5年)
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
        pullback_point = None
        for i in range(max_idx + 1, len(prices)):
            pullback = (max_price - prices[i]) / initial_rise
            if 0.18 <= pullback <= 0.22:  # 回测幅度接近20%
                pullback_point = i
                break  # 找到第一个满足条件的回测点
        
        # 如果找到了回测点，检查是否在近期(4个月内)
        if pullback_point is not None and pullback_point >= (len(prices) - 84):  # 约4个月的交易日
            return True
        
        return False
    except Exception as e:
        print(f"分析股票 {stock_info.stock_code} 的初升段回测时出错: {e}")
        return False


def filter_stocks(min_pe=5, max_pe=50, market_type="US", strategies=None, output_dir=None):
    """
    根据多种策略筛选股票
    
    Args:
        min_pe (float): 最小市盈率
        max_pe (float): 最大市盈率
        market_type (str): 市场类型
        strategies (list): 筛选策略列表，默认为全部策略
        output_dir (str): 输出文件目录，如果提供则将结果实时写入文件
        
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
    
    # 如果提供了输出目录，则准备输出文件
    file_handlers = {}
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 创建时间戳，用于文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 为每个策略创建一个文件
        file_handlers["volume_surge"] = open(os.path.join(output_dir, f"strategy1_volume_surge_{timestamp}.txt"), "w", encoding="utf-8")
        file_handlers["pullback_50"] = open(os.path.join(output_dir, f"strategy2_pullback_50_{timestamp}.txt"), "w", encoding="utf-8")
        file_handlers["initial_pullback_20"] = open(os.path.join(output_dir, f"strategy3_initial_pullback_20_{timestamp}.txt"), "w", encoding="utf-8")
        file_handlers["all_strategies"] = open(os.path.join(output_dir, f"all_strategies_{timestamp}.txt"), "w", encoding="utf-8")
        
        # 写入文件头
        for key, file in file_handlers.items():
            file.write(f"筛选条件: PE介于{min_pe}和{max_pe}之间的{market_type}股票\n")
            file.write("股票代码, 股票名称, PE\n")
            file.flush()
    
    total_stocks = len(pe_filtered_stocks)
    for i, stock in enumerate(pe_filtered_stocks):
        print(f"处理进度: {i+1}/{total_stocks} - 当前股票: {stock.stock_code} ({stock.stock_name})")
        
        matched_strategies = []
        
        if "volume_surge" in strategies and filter_by_volume_surge(stock):
            result["volume_surge"].append(stock)
            matched_strategies.append("volume_surge")
            
            # 如果有文件输出，则立即写入
            if output_dir and "volume_surge" in file_handlers:
                file_handlers["volume_surge"].write(f"{stock.stock_code}, {stock.stock_name}, {stock.pe_ratio}\n")
                file_handlers["volume_surge"].flush()  # 确保立即写入磁盘
        
        if "pullback_50" in strategies and filter_by_pullback_50_percent(stock):
            result["pullback_50"].append(stock)
            matched_strategies.append("pullback_50")
            
            # 如果有文件输出，则立即写入
            if output_dir and "pullback_50" in file_handlers:
                file_handlers["pullback_50"].write(f"{stock.stock_code}, {stock.stock_name}, {stock.pe_ratio}\n")
                file_handlers["pullback_50"].flush()  # 确保立即写入磁盘
        
        if "initial_pullback_20" in strategies and filter_by_initial_pullback_20_percent(stock):
            result["initial_pullback_20"].append(stock)
            matched_strategies.append("initial_pullback_20")
            
            # 如果有文件输出，则立即写入
            if output_dir and "initial_pullback_20" in file_handlers:
                file_handlers["initial_pullback_20"].write(f"{stock.stock_code}, {stock.stock_name}, {stock.pe_ratio}\n")
                file_handlers["initial_pullback_20"].flush()  # 确保立即写入磁盘
        
        # 如果符合所有指定的策略，添加到all_strategies列表
        if len(matched_strategies) == len(strategies):
            result["all_strategies"].append(stock)
            
            # 如果有文件输出，则立即写入
            if output_dir and "all_strategies" in file_handlers:
                file_handlers["all_strategies"].write(f"{stock.stock_code}, {stock.stock_name}, {stock.pe_ratio}\n")
                file_handlers["all_strategies"].flush()  # 确保立即写入磁盘
    
    # 关闭所有文件
    if output_dir:
        for file in file_handlers.values():
            file.close()
        
        print(f"\n筛选结果已保存到目录: {output_dir}")
    
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
    test_mode = True
    if test_mode:
        try:
            # 测试图中的股票
            test_stocks = ["VSCO", "KSS", "AAPL", "XP", "ABLV"]
            for test_stock_code in test_stocks:
                print(f"\n测试获取 {test_stock_code} 的历史数据...")
                history = get_stock_history(test_stock_code)
                if history is not None:
                    print(f"成功获取到 {len(history)} 条历史数据")
                    
                    # 显示历史最高价和最低价
                    min_price = history['Close'].min()
                    min_date = history['Close'].idxmin().strftime('%Y-%m-%d')
                    max_price = history['Close'].max()
                    max_date = history['Close'].idxmax().strftime('%Y-%m-%d')
                    
                    print(f"历史最低收盘价: ${min_price:.2f} ({min_date})")
                    print(f"历史最高收盘价: ${max_price:.2f} ({max_date})")
                    
                    # 显示最近价格
                    latest_price = history['Close'].iloc[-1]
                    latest_date = history.index[-1].strftime('%Y-%m-%d')
                    print(f"最近收盘价: ${latest_price:.2f} ({latest_date})")
                    print(f"股价是否大于5美元: {latest_price > 5.0}")
                    
                    # 计算当前价格在历史区间的位置
                    price_percentile = (latest_price - min_price) / (max_price - min_price) * 100
                    print(f"当前价格位于历史区间的: {price_percentile:.2f}%")
                    
                    # 计算最近10个交易日的平均成交量
                    recent_avg_volume = np.mean(history['Volume'].iloc[-10:])
                    # 计算之前30个交易日的平均成交量
                    previous_avg_volume = np.mean(history['Volume'].iloc[-40:-10]) 
                    volume_increase_ratio = recent_avg_volume / previous_avg_volume
                    print(f"最近10天平均成交量是之前的 {volume_increase_ratio:.2f} 倍")
                    
                    # 检查连续高成交量
                    volumes = history['Volume'].values
                    previous_avg = np.mean(volumes[-40:-10])
                    high_volume_days = 0
                    for i in range(1, 11):  # 检查最近10天
                        if volumes[-i] > previous_avg * 2.5:
                            high_volume_days += 1
                    print(f"最近10天中有 {high_volume_days} 天成交量是之前均值的2.5倍以上")
                    
                    # 查看价格趋势
                    price_trend = history['Close'].iloc[-1] / history['Close'].iloc[-10] - 1
                    print(f"最近10天价格变化: {price_trend:.2%}")
                    
                    # 模拟筛选
                    class MockStock:
                        def __init__(self, code):
                            self.stock_code = code
                            self.stock_name = code
                            self.pe_ratio = 10.0
                    
                    mock_stock = MockStock(test_stock_code)
                    is_matched = filter_by_volume_surge(mock_stock)
                    print(f"是否符合放量策略: {is_matched}")
                else:
                    print("获取历史数据失败")
            sys.exit(0)
        except Exception as e:
            print(f"测试时出错: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # 创建输出目录
    output_dir = os.path.join(parent_dir, "stock_filter_results")
    
    # 筛选符合所有策略的股票，并将结果实时写入文件
    try:
        filtered_stocks = filter_stocks(min_pe=5, max_pe=15, output_dir=output_dir)
        
        # 控制台也打印一下结果概要
        print("\n筛选结果概要:")
        print(f"符合策略1(历史底部放量)的股票: {len(filtered_stocks['volume_surge'])}支")
        print(f"符合策略2(翻倍后回调50%)的股票: {len(filtered_stocks['pullback_50'])}支")
        print(f"符合策略3(初升段回测20%)的股票: {len(filtered_stocks['initial_pullback_20'])}支")
        print(f"符合所有策略的股票: {len(filtered_stocks['all_strategies'])}支")
    except Exception as e:
        print(f"筛选股票时出错: {e}")
        import traceback
        traceback.print_exc()
