import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv  # 导入csv模块

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


def get_stock_history(stock_code, period="5y", use_cache=True, cache_dir=None):
    """
    使用yfinance获取股票的历史价格和交易量数据，支持本地缓存
    
    Args:
        stock_code (str): 股票代码
        period (str): 获取数据的时间范围，默认为5年
            可选值: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        use_cache (bool): 是否使用缓存
        cache_dir (str): 缓存目录，默认为项目根目录下的'data/stock_history'
        
    Returns:
        pandas.DataFrame: 股票历史数据
    """
    try:
        import yfinance as yf
        import os
        import pandas as pd
        from datetime import datetime, timedelta
        
        # 设置缓存目录
        if cache_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            cache_dir = os.path.join(parent_dir, 'data', 'stock_history')
        
        # 确保缓存目录存在
        if use_cache and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # 缓存文件路径
        cache_file = os.path.join(cache_dir, f"{stock_code}_{period}.csv") if use_cache else None
        
        # 检查缓存是否存在且未过期（24小时内）
        if use_cache and os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(hours=24):
                # 从缓存加载数据
                try:
                    cached_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    if not cached_data.empty:
                        return cached_data
                except Exception as e:
                    print(f"读取缓存数据失败: {e}，将重新获取")
        
        # 创建Ticker对象
        ticker = yf.Ticker(stock_code)
        
        # 配置获取历史数据的参数 - 移除不支持的参数
        kwargs = {
            'period': period,      # 时间范围
            'interval': '1d',      # 数据粒度：日数据
            'auto_adjust': True,   # 自动调整价格
            'actions': False       # 不包含分红和拆分信息
        }
        
        # 获取历史数据
        try:
            history = ticker.history(**kwargs)
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                # 尝试移除可能导致问题的参数
                problematic_params = ['progress', 'proxy', 'timeout', 'raise_errors']
                for param in problematic_params:
                    if param in kwargs:
                        del kwargs[param]
                # 重试
                history = ticker.history(**kwargs)
            else:
                raise
        
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
        
        # 数据清洗：处理缺失值和异常值
        # 填充缺失值
        history = history.ffill()
        
        # 处理异常值：成交量为0的数据用前一个交易日的值填充
        history.loc[history['Volume'] <= 0, 'Volume'] = history['Volume'].shift(1)
        
        # 保存到缓存
        if use_cache:
            history.to_csv(cache_file)
            
        return history
        
    except ImportError:
        print("请先安装yfinance库: pip install yfinance")
        return None
    except Exception as e:
        print(f"获取股票 {stock_code} 的历史数据时出错: {e}")
        return None


def filter_by_volume_surge(stock_info, min_volume_ratio=2.5, min_price=5.0):
    """
    策略1: 历史底部区域附近出现持续放量并形成反转形态
    检测特征:
    1. 处于历史底部区域(30%以内)
    2. 最近一周至两周内出现明显持续放量
    3. 放量同时价格形成反转形态，从下跌到企稳/上涨
    4. 当前股价大于5美元
    5. 排除长期成交量太低的股票(流动性差)
    
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
        import numpy as np
        
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
        
        # 检查当前是否处于历史底部区域(30%以内)
        if percentiles[-1] > 0.3:
            return False
        
        # 确定是否经历了长期下跌后到达底部
        # 检查最近6个月和3个月的趋势
        lookback_120 = min(120, len(prices)-1)  # 约6个月
        lookback_60 = min(60, len(prices)-1)   # 约3个月
        
        price_trend_120d = prices[-1] / prices[-lookback_120] - 1
        price_trend_60d = prices[-1] / prices[-lookback_60] - 1
        
        # 确认是处于下跌后的底部区域，而不是上升或横盘区域
        is_after_decline = price_trend_120d < -0.15 or price_trend_60d < -0.1
        if not is_after_decline:
            return False
        
        # 计算基本面数据 - 用于检查交易活跃度
        historical_volumes = volumes[-252:] if len(volumes) >= 252 else volumes  # 使用过去一年数据
        
        # 1. 检查历史成交量的有效交易日比例 - 排除长期无交易的股票
        # 有效交易的定义：成交量不为零或不低于某个最小阈值
        min_volume_threshold = 10000  # 最小成交量阈值，根据实际情况调整
        valid_trading_days = sum(1 for vol in historical_volumes if vol > min_volume_threshold)
        valid_trading_ratio = valid_trading_days / len(historical_volumes)
        
        # 要求至少80%的交易日有足够的成交量
        if valid_trading_ratio < 0.8:
            return False
        
        # 2. 计算历史成交量的平均和中位数
        avg_historical_volume = np.mean(historical_volumes)
        median_historical_volume = np.median(historical_volumes)
        
        # 3. 检查是否是流动性极差的股票 - 平均成交量过低
        # 这里使用美股市场的经验值，可根据不同市场调整
        if avg_historical_volume < 50000:  # 日均成交量太低
            return False
        
        # 4. 检查成交量的稳定性和连续性
        # 计算成交量的变异系数(CV) - 标准差/均值，较高的CV表示成交量波动大
        volume_std = np.std(historical_volumes)
        volume_cv = volume_std / avg_historical_volume if avg_historical_volume > 0 else float('inf')
        
        # 如果变异系数过高(>5)，说明成交量非常不稳定，常见于流动性差的股票
        if volume_cv > 5:
            # 进一步检查高CV是否由少数几天的异常成交量导致
            sorted_hist_volumes = sorted(historical_volumes)
            # 去除最高10%的成交量后重新计算
            trimmed_volumes = sorted_hist_volumes[:int(len(sorted_hist_volumes) * 0.9)]
            trimmed_mean = np.mean(trimmed_volumes)
            
            # 检查修剪后的均值是否仍然足够高
            if trimmed_mean < 30000:
                return False
        
        # 获取基准期的成交量
        base_volumes = volumes[-60:-10]  # 排除最近10天
        
        # 使用中位数代替均值作为基准，避免异常值影响
        base_volume_median = np.median(base_volumes)
        if base_volume_median <= 0:
            return False
        
        # 计算最近10个交易日的成交量
        recent_volumes = volumes[-10:]
        
        # 计算最近5个交易日的成交量
        recent_5d_volumes = volumes[-5:]
        
        # 计算连续的高成交量日
        consecutive_high_vol_days = 0
        max_consecutive = 0
        
        for vol in recent_volumes:
            if vol > base_volume_median * 1.5:
                consecutive_high_vol_days += 1
                max_consecutive = max(max_consecutive, consecutive_high_vol_days)
            else:
                consecutive_high_vol_days = 0
        
        # 计算高成交量日的数量
        high_volume_days = sum(1 for vol in recent_volumes if vol > base_volume_median * min_volume_ratio)
        high_volume_days_5d = sum(1 for vol in recent_5d_volumes if vol > base_volume_median * min_volume_ratio)
        
        # 确认持续放量 - 至少有连续3天成交量超过基准的1.5倍，或者有3天以上超过基准的2.5倍
        has_sustained_volume = max_consecutive >= 3 or high_volume_days >= 3
        
        # 如果没有持续放量特征，直接返回False
        if not has_sustained_volume:
            return False
        
        # 确认价格反转形态
        # 最近10天和5天价格走势
        price_trend_10d = prices[-1] / prices[-10] - 1 if len(prices) >= 10 else 0
        price_trend_5d = prices[-1] / prices[-5] - 1 if len(prices) >= 5 else 0
        
        # 计算出现放量的具体日期并检查对应的价格走势
        volume_surge_days = []
        for i in range(1, 11):
            if i < len(volumes) and volumes[-i] > base_volume_median * min_volume_ratio:
                volume_surge_days.append(-i)
        
        # 检查放量日后的价格走势
        price_improves_after_surge = False
        if volume_surge_days:
            # 对于每个放量日，检查之后3天的价格走势
            for day in volume_surge_days:
                if day+3 < 0:  # 确保有足够数据检查后续3天
                    max_price_after = max(prices[day:day+3])
                    surge_day_price = prices[day]
                    if max_price_after > surge_day_price * 1.02:  # 放量后价格上涨超过2%
                        price_improves_after_surge = True
                        break
        
        # 确认最近的底部形态 - 检查最近10天内的最低价是否出现在前5天
        recent_10d_prices = prices[-10:]
        min_price_idx = np.argmin(recent_10d_prices)
        bottom_forms_recently = min_price_idx < 5  # 最低点在前5天
        
        # V型反转：价格先跌后涨，且最近几天持续上涨
        v_reversal = False
        if len(prices) >= 10:
            # 价格见底后连续上涨
            if min_price_idx < 5 and price_trend_5d > 0.03:
                v_reversal = True
                
        # 判断条件组合：
        
        # 1. 基本条件：在底部区域 + 有持续放量 + 价格开始好转
        basic_condition = (percentiles[-1] <= 0.3 and 
                           has_sustained_volume and 
                           price_trend_5d >= 0.01)
        
        # 2. 强烈持续放量特征：连续多日高于基准的放量 + 价格明确上涨
        strong_volume_surge = (max_consecutive >= 3 and 
                              high_volume_days >= 2 and 
                              price_trend_5d > 0.02)
        
        # 3. 典型的底部放量反转：在历史底部有明显放量 + 形成V型反转
        typical_reversal = (percentiles[-1] <= 0.25 and 
                           high_volume_days >= 2 and 
                           v_reversal)
        
        # 4. 放量后价格明显改善：在底部区域 + 持续放量后价格上涨
        post_surge_improvement = (percentiles[-1] <= 0.3 and 
                                 high_volume_days >= 3 and
                                 price_improves_after_surge)
        
        # 综合判断
        if basic_condition and (strong_volume_surge or typical_reversal or post_surge_improvement):
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
        import numpy as np
        
        # 尝试导入scipy的argrelextrema
        try:
            from scipy.signal import argrelextrema
            has_scipy = True
        except ImportError:
            has_scipy = False
        
        # 获取价格数据
        prices = history['Close'].values
        
        # 检查价格数据是否有效
        if len(prices) == 0 or np.isnan(prices).any():
            return False
        
        # 寻找局部最小值和最大值
        if has_scipy:
            # 使用更先进的极值点检测方法 - scipy的argrelextrema函数
            # 检测局部最小值点（谷底）
            order = 10  # 窗口大小，用于确定局部极值
            local_min_indices = argrelextrema(prices, np.less, order=order)[0]
            
            # 检测局部最大值点（峰顶）
            local_max_indices = argrelextrema(prices, np.greater, order=order)[0]
            
            # 过滤小波动的峰谷，保留显著的波动
            significant_mins = []
            for idx in local_min_indices:
                # 前后order个点的平均值
                surrounding_avg = np.mean(prices[max(0, idx-order):min(len(prices), idx+order+1)])
                # 如果波谷比周围平均值低5%以上，认为是显著波谷
                if prices[idx] < surrounding_avg * 0.95:
                    significant_mins.append(idx)
            
            significant_maxs = []
            for idx in local_max_indices:
                # 前后order个点的平均值
                surrounding_avg = np.mean(prices[max(0, idx-order):min(len(prices), idx+order+1)])
                # 如果波峰比周围平均值高5%以上，认为是显著波峰
                if prices[idx] > surrounding_avg * 1.05:
                    significant_maxs.append(idx)
        else:
            # 使用简化的极值点检测方法
            significant_mins = []
            significant_maxs = []
            
            # 只检查局部最小值和最大值
            for i in range(1, len(prices) - 1):
                if prices[i-1] > prices[i] < prices[i+1]:  # 局部最小值
                    significant_mins.append(i)
                elif prices[i-1] < prices[i] > prices[i+1]:  # 局部最大值
                    significant_maxs.append(i)
        
        # 按时间排序，找出波谷-波峰-回调的组合
        for bottom_idx in significant_mins:
            # 寻找底部之后的峰顶
            potential_tops = [idx for idx in significant_maxs if idx > bottom_idx]
            if not potential_tops:
                continue
                
            for top_idx in potential_tops:
                bottom_price = prices[bottom_idx]
                top_price = prices[top_idx]
                
                # 判断是否满足"起涨翻倍"条件
                if top_price >= bottom_price * 2:
                    # 找出峰顶之后的所有价格点
                    pullback_indices = [i for i in range(top_idx + 1, len(prices))]
                    
                    # 检查峰顶之后是否有回调满足条件（回调50%）
                    for pullback_idx in pullback_indices:
                        pullback_price = prices[pullback_idx]
                        pullback_ratio = (top_price - pullback_price) / (top_price - bottom_price)
                        
                        # 判断是否满足回调条件
                        if 0.5 <= pullback_ratio <= 0.6:  # 0.6是为了避免回调过深
                            # 确保回调完成点在最近6个月内（约126个交易日）
                            if pullback_idx >= (len(prices) - 126):
                                
                                # 进一步确认回调企稳
                                # 检查回调后的10个交易日是否形成底部企稳
                                if pullback_idx + 10 < len(prices):
                                    after_pullback = prices[pullback_idx:pullback_idx+10]
                                    # 企稳条件：最低价不低于回调价格的2%，最高价不高于回调价格的5%
                                    min_after = min(after_pullback)
                                    max_after = max(after_pullback)
                                    if min_after >= pullback_price * 0.98 and max_after <= pullback_price * 1.05:
                                        # 使用RSI判断是否超卖
                                        rsi = get_rsi(prices, period=14)[pullback_idx]
                                        if rsi < 40:  # RSI低于40表示超卖
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
        import numpy as np
        
        # 尝试导入scipy的argrelextrema
        try:
            from scipy.signal import argrelextrema
            has_scipy = True
        except ImportError:
            has_scipy = False
        
        # 获取价格数据
        prices = history['Close'].values
        volumes = history['Volume'].values
        dates = history.index
        
        # 检查价格数据是否有效
        if len(prices) == 0 or np.isnan(prices).any():
            return False
        
        # 使用技术分析找出明显的底部
        # 1. 使用价格和成交量的综合判断
        # 2. 使用RSI判断超卖区域
        # 3. 使用MACD判断底背离
        
        if has_scipy:
            # 使用局部最小值检测可能的底部
            order = 5  # 窗口大小，用于确定局部极值
            local_min_indices = argrelextrema(prices, np.less, order=order)[0]
        else:
            # 使用简化的方法检测局部最小值
            local_min_indices = []
            for i in range(2, len(prices) - 2):
                if (prices[i-2] > prices[i-1] > prices[i] < prices[i+1] < prices[i+2]):
                    local_min_indices.append(i)
        
        # 过滤出在过去3-12个月内形成的底部
        recent_bottoms = []
        for idx in local_min_indices:
            if len(prices) - 252 <= idx <= len(prices) - 63:  # 约3-12个月的交易日
                # 计算底部前后的价格变化率，确认是真正的低点
                before_bottom = prices[max(0, idx-20):idx]
                after_bottom = prices[idx:min(len(prices), idx+20)]
                
                if len(before_bottom) > 0 and len(after_bottom) > 5:
                    # 底部前有下跌，底部后有上涨
                    before_change = (before_bottom[0] - prices[idx]) / before_bottom[0] if before_bottom[0] > 0 else 0
                    after_change = (after_bottom[-1] - prices[idx]) / prices[idx] if prices[idx] > 0 else 0
                    
                    # 底部前下跌超过10%，底部后上涨超过10%
                    if before_change > 0.1 and after_change > 0.1:
                        recent_bottoms.append(idx)
        
        # 对每个底部检查初升段和回测
        for bottom_idx in recent_bottoms:
            bottom_price = prices[bottom_idx]
            
            # 寻找初升段高点
            initial_rise_end = min(bottom_idx + 30, len(prices))
            rise_segment = prices[bottom_idx:initial_rise_end]
            
            if len(rise_segment) < 10:  # 需要足够的数据
                continue
                
            # 找到初升段的最高点
            peak_idx = bottom_idx + np.argmax(rise_segment)
            peak_price = prices[peak_idx]
            
            # 计算初升段的上涨幅度
            rise_percent = (peak_price / bottom_price - 1) * 100
            
            # 初升段上涨超过15%
            if rise_percent < 15:
                continue
                
            # 找出初升段之后的回测点
            pullback_start = peak_idx + 1
            pullback_end = min(peak_idx + 20, len(prices))  # 回测通常发生在短期内
            
            # 如果没有足够的回测数据，跳过
            if pullback_end <= pullback_start:
                continue
                
            # 在回测区间寻找最低点
            pullback_segment = prices[pullback_start:pullback_end]
            pullback_idx = pullback_start + np.argmin(pullback_segment)
            pullback_price = prices[pullback_idx]
            
            # 计算回测幅度
            pullback_percent = (peak_price - pullback_price) / (peak_price - bottom_price) * 100
            
            # 判断是否满足回测约20%的条件
            if 18 <= pullback_percent <= 25:
                # 检查回测完成点是否在近期(4个月内)
                if pullback_idx >= (len(prices) - 84):
                    
                    # 检查回测后的企稳和上涨
                    if pullback_idx + 10 < len(prices):
                        after_pullback = prices[pullback_idx:pullback_idx+10]
                        # 回测后没有继续下跌
                        if min(after_pullback) >= pullback_price * 0.97:
                            # 回测后有企稳迹象
                            last_5_days = prices[pullback_idx+5:pullback_idx+10] if pullback_idx+10 < len(prices) else []
                            if len(last_5_days) > 0 and np.mean(last_5_days) > pullback_price:
                                # 使用技术指标确认企稳
                                # 计算回测点的RSI
                                rsi = get_rsi(prices, period=14)[pullback_idx]
                                # 计算回测点的MACD
                                macd, signal, hist = get_macd(prices)
                                
                                # RSI回升或MACD柱状图回升，认为有企稳迹象
                                if (rsi > 40 or
                                    (pullback_idx < len(hist) and hist[pullback_idx] > hist[pullback_idx-1])):
                                    return True
        
        return False
    except Exception as e:
        print(f"分析股票 {stock_info.stock_code} 的初升段回测时出错: {e}")
        return False


def filter_stocks(min_pe=5, max_pe=50, market_type="US", strategies=None, output_dir=None, parallel=False):
    """
    根据多种策略筛选股票
    
    Args:
        min_pe (float): 最小市盈率
        max_pe (float): 最大市盈率
        market_type (str): 市场类型
        strategies (list): 筛选策略列表，默认为全部策略
        output_dir (str): 输出文件目录，如果提供则将结果实时写入文件
        parallel (bool): 是否使用并行处理提高效率
        
    Returns:
        dict: 按策略分类的股票列表
    """
    import os
    import csv
    import concurrent.futures
    import traceback
    from datetime import datetime
    
    # 安全导入TA-Lib
    has_talib = False
    try:
        import talib
        has_talib = True
        print("已加载TA-Lib技术分析库")
    except ImportError:
        print("未找到TA-Lib库，将使用简化版技术分析")
    
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
    csv_writers = {}
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 创建时间戳，用于文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 为每个策略创建一个CSV文件
        file_handlers["volume_surge"] = open(os.path.join(output_dir, f"strategy1_volume_surge_{timestamp}.csv"), "w", newline='', encoding="utf-8")
        file_handlers["pullback_50"] = open(os.path.join(output_dir, f"strategy2_pullback_50_{timestamp}.csv"), "w", newline='', encoding="utf-8")
        file_handlers["initial_pullback_20"] = open(os.path.join(output_dir, f"strategy3_initial_pullback_20_{timestamp}.csv"), "w", newline='', encoding="utf-8")
        file_handlers["all_strategies"] = open(os.path.join(output_dir, f"all_strategies_{timestamp}.csv"), "w", newline='', encoding="utf-8")
        
        # 创建CSV写入器
        for key, file in file_handlers.items():
            csv_writers[key] = csv.writer(file)
            
            # 写入表头行
            if key == "all_strategies":
                # 先写入筛选条件说明
                csv_writers[key].writerow([f"筛选条件: PE介于{min_pe}和{max_pe}之间的{market_type}股票"])
                # 写入列标题
                csv_writers[key].writerow(["股票代码", "股票名称", "PE", "策略1(放量)", "策略2(回调)", "策略3(回测)"])
            else:
                # 先写入筛选条件说明
                csv_writers[key].writerow([f"筛选条件: PE介于{min_pe}和{max_pe}之间的{market_type}股票"])
                # 写入列标题
                csv_writers[key].writerow(["股票代码", "股票名称", "PE"])
            
            # 立即刷新
            file.flush()
    
    def process_stock(stock, stock_index, total_stocks):
        """处理单个股票的函数，用于并行处理"""
        try:
            print(f"处理进度: {stock_index+1}/{total_stocks} - 当前股票: {stock.stock_code} ({stock.stock_name})")
            
            # 检查每个策略
            strategy_results = {
                "volume_surge": False,
                "pullback_50": False,
                "initial_pullback_20": False
            }
            
            matched_strategies = []
            
            if "volume_surge" in strategies and filter_by_volume_surge(stock):
                strategy_results["volume_surge"] = True
                matched_strategies.append("volume_surge")
            
            if "pullback_50" in strategies and filter_by_pullback_50_percent(stock):
                strategy_results["pullback_50"] = True
                matched_strategies.append("pullback_50")
            
            if "initial_pullback_20" in strategies and filter_by_initial_pullback_20_percent(stock):
                strategy_results["initial_pullback_20"] = True
                matched_strategies.append("initial_pullback_20")
            
            return stock, strategy_results, matched_strategies
        except Exception as e:
            print(f"处理股票 {stock.stock_code} 时出错: {e}")
            traceback.print_exc()
            return stock, {"volume_surge": False, "pullback_50": False, "initial_pullback_20": False}, []
    
    total_stocks = len(pe_filtered_stocks)
    
    # 并行处理
    if parallel and total_stocks > 10:
        chunk_size = min(10, max(1, total_stocks // 5))  # 动态调整批次大小
        print(f"使用并行处理，批次大小: {chunk_size}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i, stock in enumerate(pe_filtered_stocks):
                future = executor.submit(process_stock, stock, i, total_stocks)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                stock, strategy_results, matched_strategies = future.result()
                
                # 更新结果集合
                for strategy_name, matched in strategy_results.items():
                    if matched:
                        result[strategy_name].append(stock)
                        
                        # 如果有文件输出，则立即写入
                        if output_dir and strategy_name in csv_writers:
                            csv_writers[strategy_name].writerow([stock.stock_code, stock.stock_name, stock.pe_ratio])
                            file_handlers[strategy_name].flush()
                
                # 如果符合所有指定的策略，添加到all_strategies列表
                if len(matched_strategies) == len(strategies):
                    result["all_strategies"].append(stock)
                    
                    # 如果有文件输出，则立即写入
                    if output_dir and "all_strategies" in csv_writers:
                        csv_writers["all_strategies"].writerow([
                            stock.stock_code, 
                            stock.stock_name, 
                            stock.pe_ratio,
                            strategy_results["volume_surge"],
                            strategy_results["pullback_50"],
                            strategy_results["initial_pullback_20"]
                        ])
                        file_handlers["all_strategies"].flush()
    else:
        # 串行处理
        for i, stock in enumerate(pe_filtered_stocks):
            stock, strategy_results, matched_strategies = process_stock(stock, i, total_stocks)
            
            # 更新结果集合
            for strategy_name, matched in strategy_results.items():
                if matched:
                    result[strategy_name].append(stock)
                    
                    # 如果有文件输出，则立即写入
                    if output_dir and strategy_name in csv_writers:
                        csv_writers[strategy_name].writerow([stock.stock_code, stock.stock_name, stock.pe_ratio])
                        file_handlers[strategy_name].flush()
            
            # 如果符合所有指定的策略，添加到all_strategies列表
            if len(matched_strategies) == len(strategies):
                result["all_strategies"].append(stock)
                
                # 如果有文件输出，则立即写入
                if output_dir and "all_strategies" in csv_writers:
                    csv_writers["all_strategies"].writerow([
                        stock.stock_code, 
                        stock.stock_name, 
                        stock.pe_ratio,
                        strategy_results["volume_surge"],
                        strategy_results["pullback_50"],
                        strategy_results["initial_pullback_20"]
                    ])
                    file_handlers["all_strategies"].flush()
    
    # 关闭所有文件
    if output_dir:
        for file in file_handlers.values():
            file.close()
        
        print(f"\n筛选结果已保存到目录: {output_dir}")
    
    return result


def calculate_rsi(prices, period=14):
    """
    计算相对强弱指数(RSI)，TA-Lib的替代方案
    
    Args:
        prices (numpy.array): 价格数组
        period (int): 周期，默认14天
        
    Returns:
        numpy.array: RSI值数组
    """
    import numpy as np
    
    # 确保输入是numpy数组
    prices = np.array(prices)
    
    # 计算价格变化
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    
    # 计算增长和下跌
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    
    if down == 0:  # 避免除以零
        return np.ones_like(prices) * 100
    
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[period] = 100. - (100. / (1. + rs))
    
    # 计算剩余的RSI
    for i in range(period + 1, len(prices)):
        delta = deltas[i - 1]  # 当前价格变化
        
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
            
        # 使用平滑移动平均
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        
        rs = up / down if down != 0 else float('inf')
        rsi[i] = 100. - (100. / (1. + rs))
        
    # 填充前period个值
    rsi[:period] = rsi[period]
    
    return rsi


def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    计算MACD，TA-Lib的替代方案
    
    Args:
        prices (numpy.array): 价格数组
        fast_period (int): 快线周期
        slow_period (int): 慢线周期
        signal_period (int): 信号线周期
        
    Returns:
        tuple: (macd, signal, histogram)
    """
    import numpy as np
    
    # 确保输入是numpy数组
    prices = np.array(prices)
    
    # 计算EMA
    def ema(values, period):
        alpha = 2.0 / (period + 1)
        result = np.zeros_like(values)
        # 初始化第一个值
        result[0] = values[0]
        
        # 计算EMA
        for i in range(1, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
            
        return result
    
    # 计算快线和慢线EMA
    ema_fast = ema(prices, fast_period)
    ema_slow = ema(prices, slow_period)
    
    # 计算MACD线
    macd_line = ema_fast - ema_slow
    
    # 计算信号线 (MACD的EMA)
    signal_line = ema(macd_line, signal_period)
    
    # 计算柱状图
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bbands(prices, period=20, num_std_dev=2):
    """
    计算布林带，TA-Lib的替代方案
    
    Args:
        prices (numpy.array): 价格数组
        period (int): 周期
        num_std_dev (float): 标准差倍数
        
    Returns:
        tuple: (upper, middle, lower)
    """
    import numpy as np
    
    # 确保输入是numpy数组
    prices = np.array(prices)
    
    # 简单移动平均
    def sma(values, period):
        return np.convolve(values, np.ones(period)/period, mode='valid')
    
    # 计算中轨(SMA)
    middle_band = np.zeros_like(prices)
    for i in range(period-1, len(prices)):
        middle_band[i] = np.mean(prices[i-(period-1):i+1])
    
    # 填充前period-1个值
    middle_band[:period-1] = middle_band[period-1]
    
    # 计算标准差
    std_dev = np.zeros_like(prices)
    for i in range(period-1, len(prices)):
        std_dev[i] = np.std(prices[i-(period-1):i+1], ddof=1)
    
    # 填充前period-1个值
    std_dev[:period-1] = std_dev[period-1]
    
    # 计算上轨和下轨
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    
    return upper_band, middle_band, lower_band


def have_talib():
    """检查是否安装了TA-Lib"""
    try:
        import talib
        return True
    except ImportError:
        return False


def get_rsi(prices, period=14):
    """获取RSI，优先使用TA-Lib，否则使用替代实现"""
    if have_talib():
        import talib
        return talib.RSI(prices, timeperiod=period)
    else:
        return calculate_rsi(prices, period)


def get_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """获取MACD，优先使用TA-Lib，否则使用替代实现"""
    if have_talib():
        import talib
        return talib.MACD(prices, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
    else:
        return calculate_macd(prices, fast_period, slow_period, signal_period)


def get_bbands(prices, period=20, num_std_dev=2):
    """获取布林带，优先使用TA-Lib，否则使用替代实现"""
    if have_talib():
        import talib
        return talib.BBANDS(prices, timeperiod=period, nbdevup=num_std_dev, nbdevdn=num_std_dev)
    else:
        return calculate_bbands(prices, period, num_std_dev)


if __name__ == "__main__":
    # 确保已安装必要的库
    try:
        import yfinance
    except ImportError:
        print("请先安装yfinance库: pip install yfinance")
        sys.exit(1)
    
    # 尝试导入scipy，用于更精确的算法
    try:
        import scipy
        print("已加载scipy库用于高级数据分析")
    except ImportError:
        print("警告: 未找到scipy库，将使用简化版算法。建议安装: pip install scipy")
    
    # 尝试导入TA-Lib
    try:
        import talib
        print("已加载TA-Lib技术分析库")
    except ImportError:
        print("警告: 未找到TA-Lib库，将使用简化版技术分析。请考虑安装TA-Lib以获得更准确的结果")
        print("安装指南: https://github.com/mrjbq7/ta-lib")
    
    print("开始筛选股票...")
    
    # 命令行参数解析
    import argparse
    
    parser = argparse.ArgumentParser(description="股票筛选工具")
    parser.add_argument("--min_pe", type=float, default=5, help="最小市盈率")
    parser.add_argument("--max_pe", type=float, default=15, help="最大市盈率")
    parser.add_argument("--market", type=str, default="US", help="市场类型：US(美股)、HK(港股)、CN(A股)")
    parser.add_argument("--parallel", action="store_true", help="启用并行处理提高效率")
    parser.add_argument("--test", action="store_true", help="测试模式，只处理少量样本股票")
    parser.add_argument("--strategies", type=str, default="all", 
                        help="指定要使用的策略(逗号分隔): volume_surge,pullback_50,initial_pullback_20，或使用'all'表示全部")
    
    args = parser.parse_args()
    
    # 解析策略
    selected_strategies = None  # None表示使用默认值(全部策略)
    if args.strategies != "all":
        selected_strategies = args.strategies.split(",")
    
    # 测试单个股票的数据获取 - 用于调试
    if args.test:
        try:
            # 测试图中的股票
            test_stocks = ["AMRC", "KODK", "MSFT", "VSCO", "MOMO"]
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
                    
                    # 计算最近成交量数据
                    volumes = history['Volume'].values
                    
                    # 计算移动平均成交量
                    ma_volume_10 = np.mean(history['Volume'].iloc[-10:])    # 10日均量
                    ma_volume_20 = np.mean(history['Volume'].iloc[-20:])    # 20日均量
                    ma_volume_50 = np.mean(history['Volume'].iloc[-50:])    # 50日均量
                    
                    # 计算成交量比例
                    volume_ratio_10_50 = ma_volume_10 / ma_volume_50 if ma_volume_50 > 0 else 0
                    volume_ratio_20_50 = ma_volume_20 / ma_volume_50 if ma_volume_50 > 0 else 0
                    
                    print(f"最近10天平均成交量是50日均量的 {volume_ratio_10_50:.2f} 倍")
                    print(f"最近20天平均成交量是50日均量的 {volume_ratio_20_50:.2f} 倍")
                    
                    # 技术指标
                    try:
                        import talib
                        # 计算RSI
                        rsi14 = talib.RSI(history['Close'].values, timeperiod=14)[-1]
                        print(f"当前14日RSI: {rsi14:.2f}")
                        
                        # 计算MACD
                        macd, macdsignal, macdhist = talib.MACD(history['Close'].values)
                        print(f"当前MACD: {macd[-1]:.4f}, 信号线: {macdsignal[-1]:.4f}, 柱状图: {macdhist[-1]:.4f}")
                        
                        # 计算布林带
                        upper, middle, lower = talib.BBANDS(history['Close'].values)
                        bb_width = (upper[-1] - lower[-1]) / middle[-1]
                        print(f"布林带宽度: {bb_width:.4f}")
                        
                    except ImportError:
                        print("未安装TA-Lib库，无法计算技术指标")
                    
                    # 模拟筛选
                    class MockStock:
                        def __init__(self, code):
                            self.stock_code = code
                            self.stock_name = code
                            self.pe_ratio = 10.0
                    
                    mock_stock = MockStock(test_stock_code)
                    
                    # 检查是否符合各个策略
                    is_volume_surge = filter_by_volume_surge(mock_stock)
                    is_pullback_50 = filter_by_pullback_50_percent(mock_stock)
                    is_initial_pullback_20 = filter_by_initial_pullback_20_percent(mock_stock)
                    
                    print(f"符合策略1(历史底部放量): {is_volume_surge}")
                    print(f"符合策略2(翻倍回调50%): {is_pullback_50}")
                    print(f"符合策略3(初升段回测20%): {is_initial_pullback_20}")
                else:
                    print("获取历史数据失败")
            sys.exit(0)
        except Exception as e:
            print(f"测试时出错: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_filter_results")
    
    # 筛选符合所有策略的股票，并将结果实时写入文件
    try:
        filtered_stocks = filter_stocks(
            min_pe=args.min_pe, 
            max_pe=args.max_pe, 
            market_type=args.market, 
            strategies=selected_strategies, 
            output_dir=output_dir, 
            parallel=args.parallel
        )
        
        # 控制台也打印一下结果概要
        print("\n筛选结果概要:")
        print(f"符合策略1(历史底部放量)的股票: {len(filtered_stocks['volume_surge'])}支")
        print(f"符合策略2(翻倍后回调50%)的股票: {len(filtered_stocks['pullback_50'])}支")
        print(f"符合策略3(初升段回测20%)的股票: {len(filtered_stocks['initial_pullback_20'])}支")
        print(f"符合所有策略的股票: {len(filtered_stocks['all_strategies'])}支")
        
        # 如果找到符合所有策略的股票，显示详细信息
        if filtered_stocks['all_strategies']:
            print("\n符合所有策略的股票:")
            for i, stock in enumerate(filtered_stocks['all_strategies']):
                print(f"{i+1}. {stock.stock_code} - {stock.stock_name} (PE: {stock.pe_ratio})")
    except Exception as e:
        print(f"筛选股票时出错: {e}")
        import traceback
        traceback.print_exc()
