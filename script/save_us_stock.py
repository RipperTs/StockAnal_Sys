import sys
import os
import time

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from database import get_session, StockInfo, init_db
import requests
import json


def getStockList(page: int = 1, size: int = 90):
    """
    从雪球获取所有美股股票信息
    https://finance.sina.com.cn/stock/usstock/sector.shtml
    page: 页码
    size: 每页数量, 20,40,60
    """
    url = f"https://stock.xueqiu.com/v5/stock/screener/quote/list.json?page={page}&size={size}&order=desc&order_by=percent&market=US&type=us"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.",
        "cookie": "xq_a_token=ded7d04ca80c7e01078faaba6910b799630b71d4;u=2525085352"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # 确保请求成功
        data_json = response.json()
        if data_json.get('error_code', 400) == 0:
            return data_json.get('data', {}).get('list', [])
    except Exception as e:
        print(f"获取或解析数据时发生错误: {e}")

    return []


if __name__ == '__main__':
    init_db()

    # 连接数据库
    session = get_session()

    try:
        # 清空现有的美股数据
        print("正在清空数据库中的美股数据...")
        session.query(StockInfo).filter(StockInfo.market_type == "US").delete()
        session.commit()

        page = 1
        batch_size = 50
        batch_data = []

        while True:  # 修正循环条件
            print(f"正在获取第 {page} 页的美股数据...")
            lists = getStockList(page)
            if not lists:
                print(f"第 {page} 页没有数据，停止获取")
                break

            for item in lists:
                stock_info = StockInfo(
                    stock_code=item["symbol"],
                    stock_name=item["name"],
                    market_type="US",
                    industry="",
                    pe_ratio=item['pe_ttm'])

                batch_data.append(stock_info)

                # 批量提交以提高性能
                if len(batch_data) >= batch_size:
                    session.add_all(batch_data)
                    session.commit()
                    print(f"批量插入了 {len(batch_data)} 条美股数据")
                    batch_data = []

            page += 1
            time.sleep(1)  # 休眠3秒，避免请求过于频繁被封IP

        # 提交剩余的数据
        if batch_data:
            session.add_all(batch_data)
            session.commit()
            print(f"批量插入了 {len(batch_data)} 条美股数据")

        print("美股数据更新完成")
    except Exception as e:
        session.rollback()
        print(f"更新美股数据时发生错误: {e}")
    finally:
        session.close()
