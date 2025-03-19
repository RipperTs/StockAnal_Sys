import os
import openai

class StockQA:
    """
    股票智能问答系统
    """
    def __init__(self, analyzer, openai_api_key=None, openai_model=None):
        self.analyzer = analyzer
        self.openai_api_key = os.getenv('OPENAI_API_KEY', os.getenv('OPENAI_API_KEY'))
        self.openai_api_url = os.getenv('OPENAI_API_URL', 'https://api.openai.com/v1')
        self.openai_model = os.getenv('OPENAI_API_MODEL', 'gemini-2.0-pro-exp-02-05')

    def answer_question(self, stock_code, question, market_type='A'):
        """回答关于股票的问题"""
        try:
            if not self.openai_api_key:
                return {"error": "未配置API密钥，无法使用智能问答功能"}

            # 获取股票信息
            stock_info = self.analyzer.get_stock_info(stock_code)

            # 获取技术指标数据
            df = self.analyzer.get_stock_data(stock_code, market_type)
            df = self.analyzer.calculate_indicators(df)

            # 提取最新数据
            latest = df.iloc[-1]

            # 计算评分
            score = self.analyzer.calculate_score(df)

            # 获取支撑压力位
            sr_levels = self.analyzer.identify_support_resistance(df)

            # 构建上下文
            context = f"""股票信息:
- 代码: {stock_code}
- 名称: {stock_info.get('股票名称', '未知')}
- 行业: {stock_info.get('行业', '未知')}

技术指标(最新数据):
- 价格: {latest['close']}
- 5日均线: {latest['MA5']}
- 20日均线: {latest['MA20']}
- 60日均线: {latest['MA60']}
- RSI: {latest['RSI']}
- MACD: {latest['MACD']}
- MACD信号线: {latest['Signal']}
- 布林带上轨: {latest['BB_upper']}
- 布林带中轨: {latest['BB_middle']}
- 布林带下轨: {latest['BB_lower']}
- 波动率: {latest['Volatility']}%

技术评分: {score}分

支撑位:
- 短期: {', '.join([str(level) for level in sr_levels['support_levels']['short_term']])}
- 中期: {', '.join([str(level) for level in sr_levels['support_levels']['medium_term']])}

压力位:
- 短期: {', '.join([str(level) for level in sr_levels['resistance_levels']['short_term']])}
- 中期: {', '.join([str(level) for level in sr_levels['resistance_levels']['medium_term']])}"""

            # 特定问题类型的补充信息
            if '基本面' in question or '财务' in question or '估值' in question:
                try:
                    # 导入基本面分析器
                    from fundamental_analyzer import FundamentalAnalyzer
                    fundamental = FundamentalAnalyzer()

                    # 获取基本面数据
                    indicators = fundamental.get_financial_indicators(stock_code)

                    # 添加到上下文
                    context += f"""

基本面指标:
- PE(TTM): {indicators.get('pe_ttm', '未知')}
- PB: {indicators.get('pb', '未知')}
- ROE: {indicators.get('roe', '未知')}%
- 毛利率: {indicators.get('gross_margin', '未知')}%
- 净利率: {indicators.get('net_profit_margin', '未知')}%"""
                except:
                    context += "\n\n注意：未能获取基本面数据"

            # 调用AI API回答问题
            openai.api_key = self.openai_api_key
            openai.api_base = self.openai_api_url

            system_content = """你是专业的股票分析师助手，基于'时空共振交易体系'提供分析。
            请基于技术指标和市场数据进行客观分析。
            """

            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user",
                     "content": f"请回答关于股票的问题，并参考以下股票数据：\n\n{context}\n\n问题：{question}"}
                ],
                temperature=0.7
            )

            answer = response.choices[0].message.content

            return {
                "question": question,
                "answer": answer,
                "stock_code": stock_code,
                "stock_name": stock_info.get('股票名称', '未知')
            }

        except Exception as e:
            print(f"智能问答出错: {str(e)}")
            return {
                "question": question,
                "answer": f"抱歉，回答问题时出错: {str(e)}",
                "stock_code": stock_code
            }