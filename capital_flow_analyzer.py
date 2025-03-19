# capital_flow_analyzer.py
import logging
import traceback
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class CapitalFlowAnalyzer:
    def __init__(self):
        self.data_cache = {}

        # Set up logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def get_concept_fund_flow(self, period="10日排行"):
        """Get concept/sector fund flow data"""
        try:
            self.logger.info(f"Getting concept fund flow for period: {period}")

            # Check cache
            cache_key = f"concept_fund_flow_{period}"
            if cache_key in self.data_cache:
                cache_time, cached_data = self.data_cache[cache_key]
                # If cached within the last hour, return cached data
                if (datetime.now() - cache_time).total_seconds() < 3600:
                    return cached_data

            # Get data from akshare
            concept_data = ak.stock_fund_flow_concept(symbol=period)

            # Process data
            result = []
            for _, row in concept_data.iterrows():
                try:
                    # The column names may vary, so we use a flexible approach
                    item = {
                        "rank": int(row.get("序号", 0)),
                        "sector": row.get("行业", ""),
                        "company_count": int(row.get("公司家数", 0)),
                        "sector_index": float(row.get("行业指数", 0)),
                        "change_percent": self._parse_percent(row.get("阶段涨跌幅", "0%")),
                        "inflow": float(row.get("流入资金", 0)),
                        "outflow": float(row.get("流出资金", 0)),
                        "net_flow": float(row.get("净额", 0))
                    }
                    result.append(item)
                except Exception as e:
                    # self.logger.warning(f"Error processing row in concept fund flow: {str(e)}")
                    continue

            # Cache the result
            self.data_cache[cache_key] = (datetime.now(), result)

            return result
        except Exception as e:
            self.logger.error(f"Error getting concept fund flow: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Return mock data if API fails
            return self._generate_mock_concept_fund_flow(period)

    def get_individual_fund_flow_rank(self, period="10日"):
        """Get individual stock fund flow ranking"""
        try:
            self.logger.info(f"Getting individual fund flow ranking for period: {period}")

            # Check cache
            cache_key = f"individual_fund_flow_rank_{period}"
            if cache_key in self.data_cache:
                cache_time, cached_data = self.data_cache[cache_key]
                # If cached within the last hour, return cached data
                if (datetime.now() - cache_time).total_seconds() < 3600:
                    return cached_data

            # Get data from akshare
            stock_data = ak.stock_individual_fund_flow_rank(indicator=period)

            # Process data
            result = []
            for _, row in stock_data.iterrows():
                try:
                    # Column names vary based on the period
                    period_prefix = "" if period == "今日" else f"{period}"

                    item = {
                        "rank": int(row.get("序号", 0)),
                        "code": row.get("代码", ""),
                        "name": row.get("名称", ""),
                        "price": float(row.get("最新价", 0)),
                        "change_percent": float(row.get(f"{period_prefix}涨跌幅", 0)),
                        "main_net_inflow": float(row.get(f"{period_prefix}主力净流入-净额", 0)),
                        "main_net_inflow_percent": float(row.get(f"{period_prefix}主力净流入-净占比", 0)),
                        "super_large_net_inflow": float(row.get(f"{period_prefix}超大单净流入-净额", 0)),
                        "super_large_net_inflow_percent": float(row.get(f"{period_prefix}超大单净流入-净占比", 0)),
                        "large_net_inflow": float(row.get(f"{period_prefix}大单净流入-净额", 0)),
                        "large_net_inflow_percent": float(row.get(f"{period_prefix}大单净流入-净占比", 0)),
                        "medium_net_inflow": float(row.get(f"{period_prefix}中单净流入-净额", 0)),
                        "medium_net_inflow_percent": float(row.get(f"{period_prefix}中单净流入-净占比", 0)),
                        "small_net_inflow": float(row.get(f"{period_prefix}小单净流入-净额", 0)),
                        "small_net_inflow_percent": float(row.get(f"{period_prefix}小单净流入-净占比", 0))
                    }
                    result.append(item)
                except Exception as e:
                    self.logger.warning(f"Error processing row in individual fund flow rank: {str(e)}")
                    continue

            # Cache the result
            self.data_cache[cache_key] = (datetime.now(), result)

            return result
        except Exception as e:
            self.logger.error(f"Error getting individual fund flow ranking: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Return mock data if API fails
            return self._generate_mock_individual_fund_flow_rank(period)

    def get_individual_fund_flow(self, stock_code, market_type="", re_date="10日"):
        """Get fund flow data for individual stock"""
        try:
            self.logger.info(f"Getting fund flow for stock: {stock_code}, market: {market_type}")

            # Check cache
            cache_key = f"individual_fund_flow_{stock_code}_{market_type}"
            if cache_key in self.data_cache:
                cache_time, cached_data = self.data_cache[cache_key]
                # If cached within the last hour, return cached data
                if (datetime.now() - cache_time).total_seconds() < 3600:
                    return cached_data

            # Determine the market type if not provided
            if not market_type:
                if stock_code.startswith('6'):
                    market_type = "sh"
                elif stock_code.startswith('0') or stock_code.startswith('3'):
                    market_type = "sz"
                else:
                    market_type = "sh"  # Default to Shanghai

            # Get data from akshare
            flow_data = ak.stock_individual_fund_flow(stock=stock_code, market=market_type)

            # Process data
            result = {
                "stock_code": stock_code,
                "data": []
            }

            for _, row in flow_data.iterrows():
                try:
                    item = {
                        "date": row.get("日期", ""),
                        "price": float(row.get("收盘价", 0)),
                        "change_percent": float(row.get("涨跌幅", 0)),
                        "main_net_inflow": float(row.get("主力净流入-净额", 0)),
                        "main_net_inflow_percent": float(row.get("主力净流入-净占比", 0)),
                        "super_large_net_inflow": float(row.get("超大单净流入-净额", 0)),
                        "super_large_net_inflow_percent": float(row.get("超大单净流入-净占比", 0)),
                        "large_net_inflow": float(row.get("大单净流入-净额", 0)),
                        "large_net_inflow_percent": float(row.get("大单净流入-净占比", 0)),
                        "medium_net_inflow": float(row.get("中单净流入-净额", 0)),
                        "medium_net_inflow_percent": float(row.get("中单净流入-净占比", 0)),
                        "small_net_inflow": float(row.get("小单净流入-净额", 0)),
                        "small_net_inflow_percent": float(row.get("小单净流入-净占比", 0))
                    }
                    result["data"].append(item)
                except Exception as e:
                    self.logger.warning(f"Error processing row in individual fund flow: {str(e)}")
                    continue

            # Calculate summary statistics
            if result["data"]:
                # Recent data (last 10 days)
                recent_data = result["data"][:min(10, len(result["data"]))]



                result["summary"] = {
                    "recent_days": len(recent_data),
                    "total_main_net_inflow": sum(item["main_net_inflow"] for item in recent_data),
                    "avg_main_net_inflow_percent": np.mean([item["main_net_inflow_percent"] for item in recent_data]),
                    "positive_days": sum(1 for item in recent_data if item["main_net_inflow"] > 0),
                    "negative_days": sum(1 for item in recent_data if item["main_net_inflow"] <= 0)
                }

            # Cache the result
            self.data_cache[cache_key] = (datetime.now(), result)

            return result
        except Exception as e:
            self.logger.error(f"Error getting individual fund flow: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Return mock data if API fails
            return self._generate_mock_individual_fund_flow(stock_code, market_type)

    def get_sector_stocks(self, sector):
        """Get stocks in a specific sector"""
        try:
            self.logger.info(f"Getting stocks for sector: {sector}")

            # Try to get data from akshare
            try:
                # For industry sectors (using 东方财富 interface)
                stocks = ak.stock_board_industry_cons_em(symbol=sector)

                # Extract stock list
                if not stocks.empty and '代码' in stocks.columns:
                    result = []
                    for _, row in stocks.iterrows():
                        try:
                            item = {
                                "code": row.get("代码", ""),
                                "name": row.get("名称", ""),
                                "price": float(row.get("最新价", 0)),
                                "change_percent": float(row.get("涨跌幅", 0)) if "涨跌幅" in row else 0,
                                "main_net_inflow": 0,  # We'll get this data separately if needed
                                "main_net_inflow_percent": 0  # We'll get this data separately if needed
                            }
                            result.append(item)
                        except Exception as e:
                            # self.logger.warning(f"Error processing row in sector stocks: {str(e)}")
                            continue

                    # Cache the result
                    return result
            except Exception as e:
                self.logger.warning(f"Failed to get sector stocks from API: {str(e)}")
                # Fall through to mock data

            return []

        except Exception as e:
            self.logger.error(f"Error getting sector stocks: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Return mock data if API fails
            return []

    def calculate_capital_flow_score(self, stock_code, market_type=""):
        """Calculate capital flow score for a stock"""
        try:
            self.logger.info(f"Calculating capital flow score for stock: {stock_code}")

            # Get individual fund flow data
            fund_flow = self.get_individual_fund_flow(stock_code, market_type)

            if not fund_flow or not fund_flow.get("data") or not fund_flow.get("summary"):
                return {
                    "total": 0,
                    "main_force": 0,
                    "large_order": 0,
                    "small_order": 0,
                    "details": {}
                }

            # Extract summary statistics
            summary = fund_flow["summary"]
            recent_days = summary["recent_days"]
            total_main_net_inflow = summary["total_main_net_inflow"]
            avg_main_net_inflow_percent = summary["avg_main_net_inflow_percent"]
            positive_days = summary["positive_days"]

            # Calculate main force score (0-40)
            main_force_score = 0

            # Score based on net inflow percentage
            if avg_main_net_inflow_percent > 3:
                main_force_score += 20
            elif avg_main_net_inflow_percent > 1:
                main_force_score += 15
            elif avg_main_net_inflow_percent > 0:
                main_force_score += 10

            # Score based on positive days
            positive_ratio = positive_days / recent_days if recent_days > 0 else 0
            if positive_ratio > 0.7:
                main_force_score += 20
            elif positive_ratio > 0.5:
                main_force_score += 15
            elif positive_ratio > 0.3:
                main_force_score += 10

            # Calculate large order score (0-30)
            large_order_score = 0

            # Analyze super large and large orders
            recent_super_large = [item["super_large_net_inflow"] for item in fund_flow["data"][:recent_days]]
            recent_large = [item["large_net_inflow"] for item in fund_flow["data"][:recent_days]]

            super_large_positive = sum(1 for x in recent_super_large if x > 0)
            large_positive = sum(1 for x in recent_large if x > 0)

            # Score based on super large orders
            super_large_ratio = super_large_positive / recent_days if recent_days > 0 else 0
            if super_large_ratio > 0.7:
                large_order_score += 15
            elif super_large_ratio > 0.5:
                large_order_score += 10
            elif super_large_ratio > 0.3:
                large_order_score += 5

            # Score based on large orders
            large_ratio = large_positive / recent_days if recent_days > 0 else 0
            if large_ratio > 0.7:
                large_order_score += 15
            elif large_ratio > 0.5:
                large_order_score += 10
            elif large_ratio > 0.3:
                large_order_score += 5

            # Calculate small order score (0-30)
            small_order_score = 0

            # Analyze medium and small orders
            recent_medium = [item["medium_net_inflow"] for item in fund_flow["data"][:recent_days]]
            recent_small = [item["small_net_inflow"] for item in fund_flow["data"][:recent_days]]

            medium_positive = sum(1 for x in recent_medium if x > 0)
            small_positive = sum(1 for x in recent_small if x > 0)

            # Score based on medium orders
            medium_ratio = medium_positive / recent_days if recent_days > 0 else 0
            if medium_ratio > 0.7:
                small_order_score += 15
            elif medium_ratio > 0.5:
                small_order_score += 10
            elif medium_ratio > 0.3:
                small_order_score += 5

            # Score based on small orders
            small_ratio = small_positive / recent_days if recent_days > 0 else 0
            if small_ratio > 0.7:
                small_order_score += 15
            elif small_ratio > 0.5:
                small_order_score += 10
            elif small_ratio > 0.3:
                small_order_score += 5

            # Calculate total score
            total_score = main_force_score + large_order_score + small_order_score

            return {
                "total": total_score,
                "main_force": main_force_score,
                "large_order": large_order_score,
                "small_order": small_order_score,
                "details": fund_flow
            }
        except Exception as e:
            self.logger.error(f"Error calculating capital flow score: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                "total": 0,
                "main_force": 0,
                "large_order": 0,
                "small_order": 0,
                "details": {},
                "error": str(e)
            }

    def _parse_percent(self, percent_str):
        """Parse percentage string to float"""
        try:
            if isinstance(percent_str, str) and '%' in percent_str:
                return float(percent_str.replace('%', ''))
            return float(percent_str)
        except (ValueError, TypeError):
            return 0.0

    def _generate_mock_concept_fund_flow(self, period):
        """Generate mock concept fund flow data"""
        # self.logger.warning(f"Generating mock concept fund flow data for period: {period}")

        sectors = [
            "新能源", "医药", "半导体", "芯片", "人工智能", "大数据", "云计算", "5G",
            "汽车", "消费", "金融", "互联网", "游戏", "农业", "化工", "建筑", "军工",
            "钢铁", "有色金属", "煤炭", "石油"
        ]

        result = []
        for i, sector in enumerate(sectors):
            # Random data - positive for top half, negative for bottom half
            is_positive = i < len(sectors) // 2

            inflow = round(np.random.uniform(10, 50), 2) if is_positive else round(np.random.uniform(5, 20), 2)
            outflow = round(np.random.uniform(5, 20), 2) if is_positive else round(np.random.uniform(10, 50), 2)
            net_flow = round(inflow - outflow, 2)

            change_percent = round(np.random.uniform(0, 5), 2) if is_positive else round(np.random.uniform(-5, 0), 2)

            item = {
                "rank": i + 1,
                "sector": sector,
                "company_count": np.random.randint(10, 100),
                "sector_index": round(np.random.uniform(1000, 5000), 2),
                "change_percent": change_percent,
                "inflow": inflow,
                "outflow": outflow,
                "net_flow": net_flow
            }
            result.append(item)

        # Sort by net flow (descending)
        return sorted(result, key=lambda x: x["net_flow"], reverse=True)

    def _generate_mock_individual_fund_flow_rank(self, period):
        """Generate mock individual stock fund flow ranking data"""
        # self.logger.warning(f"Generating mock individual fund flow ranking data for period: {period}")

        # Sample stock data
        stocks = [
            {"code": "600000", "name": "浦发银行"}, {"code": "600036", "name": "招商银行"},
            {"code": "601318", "name": "中国平安"}, {"code": "600519", "name": "贵州茅台"},
            {"code": "000858", "name": "五粮液"}, {"code": "000333", "name": "美的集团"},
            {"code": "600276", "name": "恒瑞医药"}, {"code": "601888", "name": "中国中免"},
            {"code": "600030", "name": "中信证券"}, {"code": "601166", "name": "兴业银行"},
            {"code": "600887", "name": "伊利股份"}, {"code": "601398", "name": "工商银行"},
            {"code": "600028", "name": "中国石化"}, {"code": "601988", "name": "中国银行"},
            {"code": "601857", "name": "中国石油"}, {"code": "600019", "name": "宝钢股份"},
            {"code": "600050", "name": "中国联通"}, {"code": "601328", "name": "交通银行"},
            {"code": "601668", "name": "中国建筑"}, {"code": "601288", "name": "农业银行"}
        ]

        result = []
        for i, stock in enumerate(stocks):
            # Random data - positive for top half, negative for bottom half
            is_positive = i < len(stocks) // 2

            main_net_inflow = round(np.random.uniform(1e6, 5e7), 2) if is_positive else round(
                np.random.uniform(-5e7, -1e6), 2)
            main_net_inflow_percent = round(np.random.uniform(1, 10), 2) if is_positive else round(
                np.random.uniform(-10, -1), 2)

            super_large_net_inflow = round(main_net_inflow * np.random.uniform(0.3, 0.5), 2)
            super_large_net_inflow_percent = round(main_net_inflow_percent * np.random.uniform(0.3, 0.5), 2)

            large_net_inflow = round(main_net_inflow * np.random.uniform(0.3, 0.5), 2)
            large_net_inflow_percent = round(main_net_inflow_percent * np.random.uniform(0.3, 0.5), 2)

            medium_net_inflow = round(np.random.uniform(-1e6, 1e6), 2)
            medium_net_inflow_percent = round(np.random.uniform(-2, 2), 2)

            small_net_inflow = round(np.random.uniform(-1e6, 1e6), 2)
            small_net_inflow_percent = round(np.random.uniform(-2, 2), 2)

            change_percent = round(np.random.uniform(0, 5), 2) if is_positive else round(np.random.uniform(-5, 0), 2)

            item = {
                "rank": i + 1,
                "code": stock["code"],
                "name": stock["name"],
                "price": round(np.random.uniform(10, 100), 2),
                "change_percent": change_percent,
                "main_net_inflow": main_net_inflow,
                "main_net_inflow_percent": main_net_inflow_percent,
                "super_large_net_inflow": super_large_net_inflow,
                "super_large_net_inflow_percent": super_large_net_inflow_percent,
                "large_net_inflow": large_net_inflow,
                "large_net_inflow_percent": large_net_inflow_percent,
                "medium_net_inflow": medium_net_inflow,
                "medium_net_inflow_percent": medium_net_inflow_percent,
                "small_net_inflow": small_net_inflow,
                "small_net_inflow_percent": small_net_inflow_percent
            }
            result.append(item)

        # Sort by main net inflow (descending)
        return sorted(result, key=lambda x: x["main_net_inflow"], reverse=True)

    def _generate_mock_individual_fund_flow(self, stock_code, market_type):
        """Generate mock individual stock fund flow data"""
        # self.logger.warning(f"Generating mock individual fund flow data for stock: {stock_code}")

        # Generate 30 days of mock data
        end_date = datetime.now()

        result = {
            "stock_code": stock_code,
            "data": []
        }

        # Create mock price trend (reasonable random walk)
        base_price = np.random.uniform(10, 100)
        current_price = base_price

        for i in range(30):
            date = (end_date - timedelta(days=i)).strftime('%Y-%m-%d')

            # Random price change (-2% to +2%)
            change_percent = np.random.uniform(-2, 2)
            price = round(current_price * (1 + change_percent / 100), 2)
            current_price = price

            # Random fund flow data, somewhat correlated with price change
            is_positive = change_percent > 0

            main_net_inflow = round(np.random.uniform(1e5, 5e6), 2) if is_positive else round(
                np.random.uniform(-5e6, -1e5), 2)
            main_net_inflow_percent = round(np.random.uniform(1, 5), 2) if is_positive else round(
                np.random.uniform(-5, -1), 2)

            super_large_net_inflow = round(main_net_inflow * np.random.uniform(0.3, 0.5), 2)
            super_large_net_inflow_percent = round(main_net_inflow_percent * np.random.uniform(0.3, 0.5), 2)

            large_net_inflow = round(main_net_inflow * np.random.uniform(0.3, 0.5), 2)
            large_net_inflow_percent = round(main_net_inflow_percent * np.random.uniform(0.3, 0.5), 2)

            medium_net_inflow = round(np.random.uniform(-1e5, 1e5), 2)
            medium_net_inflow_percent = round(np.random.uniform(-2, 2), 2)

            small_net_inflow = round(np.random.uniform(-1e5, 1e5), 2)
            small_net_inflow_percent = round(np.random.uniform(-2, 2), 2)

            item = {
                "date": date,
                "price": price,
                "change_percent": round(change_percent, 2),
                "main_net_inflow": main_net_inflow,
                "main_net_inflow_percent": main_net_inflow_percent,
                "super_large_net_inflow": super_large_net_inflow,
                "super_large_net_inflow_percent": super_large_net_inflow_percent,
                "large_net_inflow": large_net_inflow,
                "large_net_inflow_percent": large_net_inflow_percent,
                "medium_net_inflow": medium_net_inflow,
                "medium_net_inflow_percent": medium_net_inflow_percent,
                "small_net_inflow": small_net_inflow,
                "small_net_inflow_percent": small_net_inflow_percent
            }
            result["data"].append(item)

        # Sort data by date descending (newest first)
        result["data"].sort(key=lambda x: x["date"], reverse=True)

        # Calculate summary statistics
        recent_data = result["data"][:10]

        result["summary"] = {
            "recent_days": len(recent_data),
            "total_main_net_inflow": sum(item["main_net_inflow"] for item in recent_data),
            "avg_main_net_inflow_percent": np.mean([item["main_net_inflow_percent"] for item in recent_data]),
            "positive_days": sum(1 for item in recent_data if item["main_net_inflow"] > 0),
            "negative_days": sum(1 for item in recent_data if item["main_net_inflow"] <= 0)
        }

        return result

    def _generate_mock_sector_stocks(self, sector):
        """Generate mock stocks for a sector"""
        # self.logger.warning(f"Generating mock sector stocks for: {sector}")

        # Number of stocks to generate
        num_stocks = np.random.randint(20, 50)

        result = []
        for i in range(num_stocks):
            prefix = "6" if np.random.random() > 0.5 else "0"
            stock_code = prefix + str(100000 + i).zfill(5)[-5:]

            change_percent = round(np.random.uniform(-5, 5), 2)

            item = {
                "code": stock_code,
                "name": f"{sector}股票{i + 1}",
                "price": round(np.random.uniform(10, 100), 2),
                "change_percent": change_percent,
                "main_net_inflow": round(np.random.uniform(-1e6, 1e6), 2),
                "main_net_inflow_percent": round(np.random.uniform(-5, 5), 2)
            }
            result.append(item)

        # Sort by main net inflow (descending)
        return sorted(result, key=lambda x: x["main_net_inflow"], reverse=True)