import os
import sys

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from database import get_session, StockInfo
from sqlalchemy import or_


class StockInfoDAO:
    """股票信息数据访问对象"""

    @staticmethod
    def find_by_code(market_type, stock_code):
        """
        根据市场类型和股票代码查询股票信息
        
        Args:
            market_type (str): 市场类型，如 'US'
            stock_code (str): 股票代码
            
        Returns:
            StockInfo: 股票信息对象，如果未找到则返回None
        """
        session = get_session()
        try:
            return session.query(StockInfo).filter(
                StockInfo.market_type == market_type,
                StockInfo.stock_code == stock_code
            ).first()
        finally:
            session.close()

    @staticmethod
    def find_by_name(market_type, stock_name):
        """
        根据市场类型和股票名称查询股票信息
        
        Args:
            market_type (str): 市场类型，如 'US'
            stock_name (str): 股票名称
            
        Returns:
            StockInfo: 股票信息对象，如果未找到则返回None
        """
        session = get_session()
        try:
            return session.query(StockInfo).filter(
                StockInfo.market_type == market_type,
                StockInfo.stock_name == stock_name
            ).first()
        finally:
            session.close()
    
    @staticmethod
    def find_by_name_like(market_type, name_pattern):
        """
        根据市场类型和股票名称模糊查询股票信息
        
        Args:
            market_type (str): 市场类型，如 'US'
            name_pattern (str): 股票名称模式，如 '苹果'
            
        Returns:
            list: 股票信息对象列表
        """
        session = get_session()
        try:
            return session.query(StockInfo).filter(
                StockInfo.market_type == market_type,
                StockInfo.stock_name.like(f'%{name_pattern}%')
            ).all()
        finally:
            session.close()
    
    @staticmethod
    def find_by_market_type(market_type):
        """
        根据市场类型查询所有股票信息
        
        Args:
            market_type (str): 市场类型，如 'US'
            
        Returns:
            list: 股票信息对象列表
        """
        session = get_session()
        try:
            return session.query(StockInfo).filter(
                StockInfo.market_type == market_type
            ).all()
        finally:
            session.close()
    
    @staticmethod
    def search(market_type, keyword):
        """
        根据市场类型和关键字搜索股票（代码或名称包含关键字）
        
        Args:
            market_type (str): 市场类型，如 'US'
            keyword (str): 搜索关键字
            
        Returns:
            list: 股票信息对象列表
        """
        session = get_session()
        try:
            return session.query(StockInfo).filter(
                StockInfo.market_type == market_type,
                or_(
                    StockInfo.stock_code.like(f'%{keyword}%'),
                    StockInfo.stock_name.like(f'%{keyword}%')
                )
            ).all()
        finally:
            session.close()


if __name__ == '__main__':
    # 测试查询
    stock_info = StockInfoDAO.find_by_code('US', 'AAPL')
    if stock_info:
        print(stock_info.to_dict())
    else:
        print('未找到股票信息')

    stock_info = StockInfoDAO.find_by_name('US', '苹果')
    if stock_info:
        print(stock_info.to_dict())
    else:
        print('未找到股票信息')

    stock_list = StockInfoDAO.find_by_name_like('US', '苹果')
    for stock_info in stock_list:
        print(stock_info.to_dict())

    stock_list = StockInfoDAO.find_by_market_type('US')
    for stock_info in stock_list:
        print(stock_info.to_dict())

    stock_list = StockInfoDAO.search('US', '苹果')
    for stock_info in stock_list:
        print(stock_info.to_dict())