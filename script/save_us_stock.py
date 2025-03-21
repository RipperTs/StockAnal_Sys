import sys
import os
import time

from service.xueqiu_stock import XueQiuStock

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from database import get_session, StockInfo, init_db
import requests
import json

if __name__ == '__main__':

    market_type = "A"
    init_db()

    xueqiu_stock = XueQiuStock()

    # 连接数据库
    session = get_session()

    try:
        # 清空现有的美股数据
        print(f"正在清空数据库中的{market_type}股票数据...")
        session.query(StockInfo).filter(StockInfo.market_type == market_type).delete()
        session.commit()

        page = 1
        batch_size = 50
        batch_data = []

        while True:  # 修正循环条件
            print(f"正在获取第 {page} 页的数据...")
            lists = xueqiu_stock.getStockList(page=page, market_type=market_type)
            if not lists:
                print(f"第 {page} 页没有数据，停止获取")
                break

            for item in lists:
                stock_info = StockInfo(
                    stock_code=item["symbol"],
                    stock_name=item["name"],
                    market_type=market_type,
                    industry="",
                    pe_ratio=item['pe_ttm'])

                batch_data.append(stock_info)

                # 批量提交以提高性能
                if len(batch_data) >= batch_size:
                    session.add_all(batch_data)
                    session.commit()
                    print(f"批量插入了 {len(batch_data)} 条{market_type}股票数据数据")
                    batch_data = []

            page += 1
            time.sleep(1)  # 休眠3秒，避免请求过于频繁被封IP

        # 提交剩余的数据
        if batch_data:
            session.add_all(batch_data)
            session.commit()
            print(f"批量插入了 {len(batch_data)} 条{market_type}股票数据数据")

        print("数据更新完成")
    except Exception as e:
        session.rollback()
        print(f"更新数据时发生错误: {e}")
    finally:
        session.close()
