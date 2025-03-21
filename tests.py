import yfinance as yf

if __name__ == '__main__':
    res = ticker = yf.Ticker("001382")

    kwargs = {
        'period': '1y',  # 时间范围
        'interval': '1d',  # 数据粒度：日数据
        'auto_adjust': True,  # 自动调整价格
        'actions': False  # 不包含分红和拆分信息
    }

    history = ticker.history(**kwargs)
    print(history)