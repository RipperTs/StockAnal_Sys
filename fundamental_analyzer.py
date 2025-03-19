import akshare as ak
import pandas as pd


class FundamentalAnalyzer:
    """
    基本面分析
    """
    def __init__(self):
        self.data_cache = {}

    def get_financial_indicators(self, stock_code, market_type='A'):
        """获取财务指标"""
        try:
            # 检查市场类型，使用不同的处理方法
            if market_type == 'US' or (not market_type and self._is_us_stock(stock_code)):
                return self.get_us_financial_indicators(stock_code)
            
            # A股处理
            # 获取基本财务指标
            financial_data = ak.stock_financial_analysis_indicator(symbol=stock_code,start_year="2022")

            # 获取最新估值指标
            valuation = ak.stock_value_em(symbol=stock_code)

            # 整合数据
            indicators = {
                'stock_name': self._get_stock_name(stock_code),
                'industry': self._get_industry(stock_code),
                'pe_ttm': float(valuation['PE(TTM)'].iloc[0]),
                'pb': float(valuation['市净率'].iloc[0]),
                'ps_ttm': float(valuation['市销率'].iloc[0]),
                'roe': float(financial_data['加权净资产收益率(%)'].iloc[0]),
                'gross_margin': float(financial_data['销售毛利率(%)'].iloc[0]),
                'net_profit_margin': float(financial_data['总资产净利润率(%)'].iloc[0]),
                'debt_ratio': float(financial_data['资产负债率(%)'].iloc[0])
            }

            return indicators
        except Exception as e:
            print(f"获取财务指标出错: {str(e)}")
            # 返回默认值而不是空字典，避免访问None时出错
            return {
                'stock_name': self._get_stock_name(stock_code) or stock_code,
                'industry': '未知',
                'pe_ttm': 0,
                'pb': 0,
                'ps_ttm': 0,
                'roe': 0,
                'gross_margin': 0,
                'net_profit_margin': 0,
                'debt_ratio': 0
            }

    def get_us_financial_indicators(self, stock_code):
        """获取美股财务指标"""
        try:
            # 清理股票代码，确保格式正确
            clean_code = self._clean_us_stock_code(stock_code)
            
            # 获取美股基本面财务指标数据
            fin_indicator = ak.stock_financial_us_analysis_indicator_em(symbol=clean_code, indicator="年报")
            
            # 默认值
            stock_name = clean_code
            industry = "未知"
            pe_ttm = 0
            pb = 0
            roe = 0
            gross_margin = 0
            net_margin = 0
            debt_ratio = 0
            
            # 从财务指标中提取数据
            if not fin_indicator.empty:
                try:
                    # 获取第一行数据
                    row = fin_indicator.iloc[0]
                    
                    # 公司名称
                    if 'SECURITY_NAME_ABBR' in fin_indicator.columns:
                        stock_name = row['SECURITY_NAME_ABBR'] if not pd.isna(row['SECURITY_NAME_ABBR']) else clean_code
                    
                    # ROE (净资产收益率)
                    if 'ROE_AVG' in fin_indicator.columns:
                        roe = float(row['ROE_AVG']) if not pd.isna(row['ROE_AVG']) else 0
                    
                    # 毛利率
                    if 'GROSS_PROFIT_RATIO' in fin_indicator.columns:
                        gross_margin = float(row['GROSS_PROFIT_RATIO']) if not pd.isna(row['GROSS_PROFIT_RATIO']) else 0
                    
                    # 净利率
                    if 'NET_PROFIT_RATIO' in fin_indicator.columns:
                        net_margin = float(row['NET_PROFIT_RATIO']) if not pd.isna(row['NET_PROFIT_RATIO']) else 0
                    
                    # 资产负债率
                    if 'DEBT_ASSET_RATIO' in fin_indicator.columns:
                        debt_ratio = float(row['DEBT_ASSET_RATIO']) if not pd.isna(row['DEBT_ASSET_RATIO']) else 0
                    
                    # 尝试计算PE (TTM)，基于每股收益
                    if 'BASIC_EPS' in fin_indicator.columns and float(row['BASIC_EPS']) > 0:
                        # 由于没有实时股价，使用基本每股收益和行业平均PE估算PE
                        pe_ttm = 20  # 使用默认的适中PE值
                        
                    # 其他信息（行业等）暂时无法从此API获取
                except Exception as e:
                    print(f"解析美股财务指标出错: {str(e)}")
            
            # 构建返回结果
            indicators = {
                'stock_name': f"{stock_name} (US)",
                'industry': industry,
                'pe_ttm': pe_ttm,
                'pb': pb,
                'ps_ttm': 0,  # 美股数据源可能没有市销率，设为0
                'roe': roe,
                'gross_margin': gross_margin,
                'net_profit_margin': net_margin,
                'debt_ratio': debt_ratio
            }
            
            return indicators
        except Exception as e:
            print(f"获取美股财务指标出错: {str(e)}")
            # 返回默认值而不是空字典
            return {
                'stock_name': f"{stock_code} (US)",
                'industry': '未知',
                'pe_ttm': 0,
                'pb': 0,
                'ps_ttm': 0,
                'roe': 0,
                'gross_margin': 0,
                'net_profit_margin': 0,
                'debt_ratio': 0
            }

    def get_growth_data(self, stock_code, market_type='A'):
        """获取成长性数据"""
        try:
            # 检查是否为美股
            if market_type == 'US' or (not market_type and self._is_us_stock(stock_code)):
                return self.get_us_growth_data(stock_code)
                
            # A股处理
            # 获取历年财务数据
            financial_data = ak.stock_financial_abstract(symbol=stock_code)

            # 检查数据结构
            if financial_data.empty:
                raise ValueError(f"未找到股票 {stock_code} 的财务数据")
                
            # 新的API数据结构处理
            # 如果数据包含"指标"列，则使用新格式处理方式
            if '指标' in financial_data.columns:
                # 找到"营业总收入"或"营业收入"行
                revenue_row = financial_data[financial_data['指标'].str.contains('营业总收入|营业收入', na=False)]
                
                if revenue_row.empty:
                    raise ValueError(f"未找到营业收入数据")
                    
                # 找到"净利润"行
                profit_row = financial_data[financial_data['指标'].str.contains('净利润$', na=False)]
                if profit_row.empty:
                    # 尝试使用"归母净利润"
                    profit_row = financial_data[financial_data['指标'].str.contains('归母净利润', na=False)]
                
                if profit_row.empty:
                    raise ValueError(f"未找到净利润数据")
                
                # 提取日期列（去掉"选项"和"指标"列）
                date_cols = [col for col in financial_data.columns if col not in ['选项', '指标']]
                date_cols.sort(reverse=True)  # 按日期降序排序
                
                # 提取收入和利润数据
                revenue_series = pd.Series(dtype=float)
                profit_series = pd.Series(dtype=float)
                
                # 使用所有日期列获取数据
                for date in date_cols:
                    try:
                        if pd.notna(revenue_row[date].iloc[0]) and revenue_row[date].iloc[0] != '':
                            rev_value = float(revenue_row[date].iloc[0])
                            if rev_value > 0:
                                # 只使用年度数据（以1231结尾）
                                if date.endswith('1231'):
                                    year = date[:4]
                                    revenue_series[year] = rev_value
                    except (ValueError, TypeError):
                        continue
                    
                    try:
                        if pd.notna(profit_row[date].iloc[0]) and profit_row[date].iloc[0] != '':
                            profit_value = float(profit_row[date].iloc[0])
                            if profit_value != 0:
                                # 只使用年度数据（以1231结尾）
                                if date.endswith('1231'):
                                    year = date[:4]
                                    profit_series[year] = profit_value
                    except (ValueError, TypeError):
                        continue
            
            # 旧的API数据结构处理（保留兼容性）
            else:
                # 尝试直接从列名读取（旧方式）
                try:
                    revenue = financial_data['营业收入'].astype(float)
                    net_profit = financial_data['净利润'].astype(float)
                except KeyError:
                    # 尝试其他可能的列名
                    try:
                        revenue = financial_data['营业总收入'].astype(float) if '营业总收入' in financial_data.columns else None
                        if revenue is None:
                            revenue = financial_data.filter(like='收入').iloc[:, 0].astype(float)
                        
                        net_profit = financial_data['归母净利润'].astype(float) if '归母净利润' in financial_data.columns else None
                        if net_profit is None:
                            net_profit = financial_data.filter(like='利润').iloc[:, 0].astype(float)
                    except:
                        raise ValueError("无法识别财务数据列名")
                
                revenue_series = revenue
                profit_series = net_profit

            # 使用所有年份数据计算增长率
            if len(revenue_series) >= 2 or len(profit_series) >= 2:
                revenue_years = sorted(revenue_series.index, reverse=True)
                profit_years = sorted(profit_series.index, reverse=True)
                
                # 计算各项成长率
                growth = {
                    'revenue_growth_3y': self._calculate_annual_growth(revenue_series) if len(revenue_years) >= 2 else 0,
                    'profit_growth_3y': self._calculate_annual_growth(profit_series) if len(profit_years) >= 2 else 0,
                    'revenue_growth_5y': self._calculate_annual_growth(revenue_series, 5) if len(revenue_years) >= 2 else 0,
                    'profit_growth_5y': self._calculate_annual_growth(profit_series, 5) if len(profit_years) >= 2 else 0
                }
            else:
                growth = {
                    'revenue_growth_3y': 0,
                    'profit_growth_3y': 0,
                    'revenue_growth_5y': 0,
                    'profit_growth_5y': 0
                }

            return growth
        except Exception as e:
            print(f"获取成长数据出错: {str(e)}")
            # 返回默认值
            return {
                'revenue_growth_3y': 0,
                'profit_growth_3y': 0,
                'revenue_growth_5y': 0,
                'profit_growth_5y': 0
            }
            
    def _calculate_annual_growth(self, series, max_years=3):
        """计算年均增长率"""
        if len(series) < 2:
            return 0
            
        try:
            # 按年份降序排序
            sorted_series = series.sort_index(ascending=False)
            
            # 获取最早和最晚的数据
            latest_year = sorted_series.index[0]
            # 如果有足够的数据，使用max_years前的数据；否则使用最早的数据
            if len(sorted_series) > max_years:
                earliest_idx = max_years - 1
            else:
                earliest_idx = len(sorted_series) - 1
                
            earliest_year = sorted_series.index[earliest_idx]
            
            latest_value = sorted_series[latest_year]
            earliest_value = sorted_series[earliest_year]
            
            # 计算实际年数
            actual_years = float(latest_year) - float(earliest_year)
            
            if actual_years <= 0 or earliest_value <= 0:
                return 0
                
            # 计算复合年增长率
            cagr = ((latest_value / earliest_value) ** (1 / actual_years) - 1) * 100
            return cagr
        except Exception as e:
            print(f"计算年均增长率出错: {str(e)}")
            return 0

    def get_us_growth_data(self, stock_code):
        """获取美股成长性数据"""
        try:
            # 清理股票代码
            clean_code = self._clean_us_stock_code(stock_code)
            
            # 获取美股财务指标数据
            try:
                # 尝试使用综合损益表获取收入和利润数据
                income_annual = ak.stock_financial_us_report_em(stock=clean_code, symbol="综合损益表", indicator="年报")
                
                # 检查是否获取到数据
                if not income_annual.empty:
                    # 创建用于存储收入和利润数据的字典
                    revenues = {}
                    profits = {}
                    
                    # 遍历数据提取收入和利润信息
                    for _, row in income_annual.iterrows():
                        if "营业收入" in str(row['ITEM_NAME']) or "Revenue" in str(row['ITEM_NAME']):
                            report_date = str(row['REPORT_DATE'])[:4]  # 提取年份
                            amount = float(row['AMOUNT']) if not pd.isna(row['AMOUNT']) else 0
                            revenues[report_date] = amount
                        
                        if "净利润" in str(row['ITEM_NAME']) or "Net Income" in str(row['ITEM_NAME']) or "Net Profit" in str(row['ITEM_NAME']):
                            report_date = str(row['REPORT_DATE'])[:4]  # 提取年份
                            amount = float(row['AMOUNT']) if not pd.isna(row['AMOUNT']) else 0
                            profits[report_date] = amount
                    
                    # 将字典转换为Series以计算增长率
                    revenue_series = pd.Series(revenues).sort_index(ascending=False)
                    profit_series = pd.Series(profits).sort_index(ascending=False)
                    
                    # 计算成长率
                    return {
                        'revenue_growth_3y': self._calculate_cagr(revenue_series, 3) if len(revenue_series) >= 3 else 0,
                        'profit_growth_3y': self._calculate_cagr(profit_series, 3) if len(profit_series) >= 3 else 0,
                        'revenue_growth_5y': self._calculate_cagr(revenue_series, 5) if len(revenue_series) >= 5 else 0,
                        'profit_growth_5y': self._calculate_cagr(profit_series, 5) if len(profit_series) >= 5 else 0
                    }
            except Exception as e:
                print(f"获取美股成长数据出错: {str(e)}")
            
            # 如果上面的方法失败，返回默认值
            return {
                'revenue_growth_3y': 0,
                'profit_growth_3y': 0,
                'revenue_growth_5y': 0,
                'profit_growth_5y': 0
            }
        except Exception as e:
            print(f"获取美股成长数据出错: {str(e)}")
            return {
                'revenue_growth_3y': 0,
                'profit_growth_3y': 0,
                'revenue_growth_5y': 0,
                'profit_growth_5y': 0
            }

    def _calculate_cagr(self, series, years):
        """计算复合年增长率"""
        if len(series) < years:
            return 0  # 改为返回0而不是None
        
        try:
            latest = series.iloc[0]
            earlier = series.iloc[min(years, len(series) - 1)]
            
            if earlier <= 0:
                return 0  # 改为返回0
            
            return ((latest / earlier) ** (1 / years) - 1) * 100
        except Exception as e:
            print(f"计算CAGR出错: {str(e)}")
            return 0  # 错误时返回0

    def _clean_us_stock_code(self, stock_code):
        """清理美股代码格式"""
        # 移除可能存在的前缀，如105.等
        if '.' in stock_code:
            parts = stock_code.split('.')
            if len(parts) >= 2 and parts[0].isdigit():
                return parts[1]
        return stock_code

    def _is_us_stock(self, stock_code):
        """判断是否为美股代码"""
        # 美股代码通常是字母或带点的格式
        if stock_code.startswith('105.'):  # 东方财富美股代码前缀
            return True
        
        # 纯字母代码可能是美股（简单判断）
        if stock_code.isalpha():
            return True
            
        return False

    def _get_stock_name(self, stock_code):
        """获取股票名称"""
        try:
            stock_info = ak.stock_individual_info_em(symbol=stock_code)
            if not stock_info.empty:
                stock_name = stock_info.iloc[0, 0]
                return stock_name
        except:
            pass
        return None

    def _get_industry(self, stock_code):
        """获取股票所属行业"""
        try:
            stock_info = ak.stock_individual_info_em(symbol=stock_code)
            if not stock_info.empty and len(stock_info) > 10:
                industry = stock_info.iloc[10, 0]
                return industry
        except:
            pass
        return '未知'

    def calculate_fundamental_score(self, stock_code, market_type='A'):
        """计算基本面综合评分"""
        # 判断市场类型
        if not market_type:
            market_type = 'US' if self._is_us_stock(stock_code) else 'A'
            
        indicators = self.get_financial_indicators(stock_code, market_type)
        growth = self.get_growth_data(stock_code, market_type)

        # 估值评分 (30分)
        valuation_score = 0
        if 'pe_ttm' in indicators and indicators['pe_ttm'] > 0:
            pe = indicators['pe_ttm']
            if pe < 15:
                valuation_score += 25
            elif pe < 25:
                valuation_score += 20
            elif pe < 35:
                valuation_score += 15
            elif pe < 50:
                valuation_score += 10
            else:
                valuation_score += 5

        # 财务健康评分 (40分)
        financial_score = 0
        if 'roe' in indicators and indicators['roe'] > 0:
            roe = indicators['roe']
            if roe > 20:
                financial_score += 15
            elif roe > 15:
                financial_score += 12
            elif roe > 10:
                financial_score += 8
            elif roe > 5:
                financial_score += 4

        if 'debt_ratio' in indicators:
            debt_ratio = indicators['debt_ratio']
            if debt_ratio < 30:
                financial_score += 15
            elif debt_ratio < 50:
                financial_score += 10
            elif debt_ratio < 70:
                financial_score += 5

        # 成长性评分 (30分)
        growth_score = 0
        if 'revenue_growth_3y' in growth and growth['revenue_growth_3y']:
            rev_growth = growth['revenue_growth_3y']
            if rev_growth > 30:
                growth_score += 15
            elif rev_growth > 20:
                growth_score += 12
            elif rev_growth > 10:
                growth_score += 8
            elif rev_growth > 0:
                growth_score += 4

        if 'profit_growth_3y' in growth and growth['profit_growth_3y']:
            profit_growth = growth['profit_growth_3y']
            if profit_growth > 30:
                growth_score += 15
            elif profit_growth > 20:
                growth_score += 12
            elif profit_growth > 10:
                growth_score += 8
            elif profit_growth > 0:
                growth_score += 4

        # 计算总分
        total_score = valuation_score + financial_score + growth_score

        return {
            'total': total_score,
            'valuation': valuation_score,
            'financial_health': financial_score,
            'growth': growth_score,
            'details': {
                'indicators': indicators,
                'growth': growth
            }
        }