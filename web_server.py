# -*- coding: utf-8 -*-
"""
智能分析系统（股票） - 股票市场数据分析系统
修改：熊猫大侠
版本：v2.1.0
"""
# web_server.py

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from stock_analyzer import StockAnalyzer
from us_stock_service import USStockService
import threading
import logging
from logging.handlers import RotatingFileHandler
import traceback
import os
import json
from datetime import date, datetime, timedelta
from flask_cors import CORS
import time
from flask_caching import Cache
import threading
import sys
from flask_swagger_ui import get_swaggerui_blueprint
from database import get_session, AnalysisResult, USE_DATABASE, init_db
from dotenv import load_dotenv
from industry_analyzer import IndustryAnalyzer

# 加载环境变量
load_dotenv()

# 检查是否需要初始化数据库
if USE_DATABASE:
    init_db()

# 配置Swagger
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "股票智能分析系统 API文档"
    }
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
analyzer = StockAnalyzer()
us_stock_service = USStockService()

# 配置缓存
cache_config = {
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300
}

# 如果配置了Redis，使用Redis作为缓存后端
if os.getenv('USE_REDIS_CACHE', 'False').lower() == 'true' and os.getenv('REDIS_URL'):
    cache_config = {
        'CACHE_TYPE': 'RedisCache',
        'CACHE_REDIS_URL': os.getenv('REDIS_URL'),
        'CACHE_DEFAULT_TIMEOUT': 300
    }

cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# 确保全局变量在重新加载时不会丢失
if 'analyzer' not in globals():
    try:
        from stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()
        print("成功初始化全局StockAnalyzer实例")
    except Exception as e:
        print(f"初始化StockAnalyzer时出错: {e}", file=sys.stderr)
        raise

# 导入新模块
from fundamental_analyzer import FundamentalAnalyzer
from capital_flow_analyzer import CapitalFlowAnalyzer
from scenario_predictor import ScenarioPredictor
from stock_qa import StockQA
from risk_monitor import RiskMonitor
from index_industry_analyzer import IndexIndustryAnalyzer

# 初始化模块实例
fundamental_analyzer = FundamentalAnalyzer()
capital_flow_analyzer = CapitalFlowAnalyzer()
scenario_predictor = ScenarioPredictor(analyzer, os.getenv('OPENAI_API_KEY'), os.getenv('OPENAI_API_MODEL'))
stock_qa = StockQA(analyzer, os.getenv('OPENAI_API_KEY'), os.getenv('OPENAI_API_MODEL'))
risk_monitor = RiskMonitor(analyzer)
index_industry_analyzer = IndexIndustryAnalyzer(analyzer)
industry_analyzer = IndustryAnalyzer()

# Thread-local storage
thread_local = threading.local()


def get_analyzer():
    """获取线程本地的分析器实例"""
    # 如果线程本地存储中没有分析器实例，创建一个新的
    if not hasattr(thread_local, 'analyzer'):
        thread_local.analyzer = StockAnalyzer()
    return thread_local.analyzer


# 配置日志
logging.basicConfig(level=logging.INFO)
handler = RotatingFileHandler('flask_app.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
))
app.logger.addHandler(handler)

# 扩展任务管理系统以支持不同类型的任务
task_types = {
    'scan': 'market_scan',  # 市场扫描任务
    'analysis': 'stock_analysis'  # 个股分析任务
}

# 任务数据存储
tasks = {
    'market_scan': {},  # 原来的scan_tasks
    'stock_analysis': {}  # 新的个股分析任务
}


def get_task_store(task_type):
    """获取指定类型的任务存储"""
    return tasks.get(task_type, {})


def generate_task_key(task_type, **params):
    """生成任务键"""
    if task_type == 'stock_analysis':
        # 对于个股分析，使用股票代码和市场类型作为键
        return f"{params.get('stock_code')}_{params.get('market_type', 'A')}"
    return None  # 其他任务类型不使用预生成的键


def get_or_create_task(task_type, **params):
    """获取或创建任务"""
    store = get_task_store(task_type)
    task_key = generate_task_key(task_type, **params)

    # 检查是否有现有任务
    if task_key and task_key in store:
        task = store[task_key]
        # 检查任务是否仍然有效
        if task['status'] in [TASK_PENDING, TASK_RUNNING]:
            return task['id'], task, False
        if task['status'] == TASK_COMPLETED and 'result' in task:
            # 任务已完成且有结果，重用它
            return task['id'], task, False

    # 创建新任务
    task_id = generate_task_id()
    task = {
        'id': task_id,
        'key': task_key,  # 存储任务键以便以后查找
        'type': task_type,
        'status': TASK_PENDING,
        'progress': 0,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'params': params
    }

    with task_lock:
        if task_key:
            store[task_key] = task
        store[task_id] = task

    return task_id, task, True


# 添加到web_server.py顶部
# 任务管理系统
scan_tasks = {}  # 存储扫描任务的状态和结果
task_lock = threading.Lock()  # 用于线程安全操作

# 任务状态常量
TASK_PENDING = 'pending'
TASK_RUNNING = 'running'
TASK_COMPLETED = 'completed'
TASK_FAILED = 'failed'


def generate_task_id():
    """生成唯一的任务ID"""
    import uuid
    return str(uuid.uuid4())


def start_market_scan_task_status(task_id, status, progress=None, result=None, error=None):
    """更新任务状态 - 保持原有签名"""
    with task_lock:
        if task_id in scan_tasks:
            task = scan_tasks[task_id]
            task['status'] = status
            if progress is not None:
                task['progress'] = progress
            if result is not None:
                task['result'] = result
            if error is not None:
                task['error'] = error
            task['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def update_task_status(task_type, task_id, status, progress=None, result=None, error=None):
    """更新任务状态"""
    store = get_task_store(task_type)
    with task_lock:
        if task_id in store:
            task = store[task_id]
            task['status'] = status
            if progress is not None:
                task['progress'] = progress
            if result is not None:
                task['result'] = result
            if error is not None:
                task['error'] = error
            task['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 更新键索引的任务
            if 'key' in task and task['key'] in store:
                store[task['key']] = task


analysis_tasks = {}


def get_or_create_analysis_task(stock_code, market_type='A'):
    """获取或创建个股分析任务"""
    # 创建一个键，用于查找现有任务
    task_key = f"{stock_code}_{market_type}"

    with task_lock:
        # 检查是否有现有任务
        for task_id, task in analysis_tasks.items():
            if task.get('key') == task_key:
                # 检查任务是否仍然有效
                if task['status'] in [TASK_PENDING, TASK_RUNNING]:
                    return task_id, task, False
                if task['status'] == TASK_COMPLETED and 'result' in task:
                    # 任务已完成且有结果，重用它
                    return task_id, task, False

        # 创建新任务
        task_id = generate_task_id()
        task = {
            'id': task_id,
            'key': task_key,
            'status': TASK_PENDING,
            'progress': 0,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'params': {
                'stock_code': stock_code,
                'market_type': market_type
            }
        }

        analysis_tasks[task_id] = task

        return task_id, task, True


def update_analysis_task(task_id, status, progress=None, result=None, error=None):
    """更新个股分析任务状态"""
    with task_lock:
        if task_id in analysis_tasks:
            task = analysis_tasks[task_id]
            task['status'] = status
            if progress is not None:
                task['progress'] = progress
            if result is not None:
                task['result'] = result
            if error is not None:
                task['error'] = error
            task['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# Define the custom JSON encoder


# In web_server.py, update the convert_numpy_types function to handle NaN values

# Function to convert NumPy types to Python native types
def convert_numpy_types(obj):
    """Recursively converts NumPy types in dictionaries and lists to Python native types"""
    try:
        import numpy as np
        import math

        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Handle NaN and Infinity specifically
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return None if obj < 0 else 1e308  # Use a very large number for +Infinity
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle Python's own float NaN and Infinity
        elif isinstance(obj, float):
            if math.isnan(obj):
                return None
            elif math.isinf(obj):
                return None
            return obj
        # 添加对date和datetime类型的处理
        elif isinstance(obj, (date, datetime)):
            return obj.isoformat()
        else:
            return obj
    except ImportError:
        # 如果没有安装numpy，但需要处理date和datetime
        import math
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (date, datetime)):
            return obj.isoformat()
        # Handle Python's own float NaN and Infinity
        elif isinstance(obj, float):
            if math.isnan(obj):
                return None
            elif math.isinf(obj):
                return None
            return obj
        return obj


# Also update the NumpyJSONEncoder class
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # For NumPy data types
        try:
            import numpy as np
            import math
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                # Handle NaN and Infinity specifically
                if np.isnan(obj):
                    return None
                elif np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            # Handle Python's own float NaN and Infinity
            elif isinstance(obj, float):
                if math.isnan(obj):
                    return None
                elif math.isinf(obj):
                    return None
                return obj
        except ImportError:
            # Handle Python's own float NaN and Infinity if numpy is not available
            import math
            if isinstance(obj, float):
                if math.isnan(obj):
                    return None
                elif math.isinf(obj):
                    return None

        # 添加对date和datetime类型的处理
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()

        return super(NumpyJSONEncoder, self).default(obj)


# Custom jsonify function that uses our encoder
def custom_jsonify(data):
    return app.response_class(
        json.dumps(convert_numpy_types(data), cls=NumpyJSONEncoder),
        mimetype='application/json'
    )


# 保持API兼容的路由
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        stock_codes = data.get('stock_codes', [])
        market_type = data.get('market_type', 'A')

        if not stock_codes:
            return jsonify({'error': '请输入代码'}), 400

        app.logger.info(f"分析股票请求: {stock_codes}, 市场类型: {market_type}")

        # 设置最大处理时间，每只股票10秒
        max_time_per_stock = 10  # 秒
        max_total_time = max(30, min(60, len(stock_codes) * max_time_per_stock))  # 至少30秒，最多60秒

        start_time = time.time()
        results = []

        for stock_code in stock_codes:
            try:
                # 检查是否已超时
                if time.time() - start_time > max_total_time:
                    app.logger.warning(f"分析股票请求已超过{max_total_time}秒，提前返回已处理的{len(results)}只股票")
                    break

                # 使用线程本地缓存的分析器实例
                current_analyzer = get_analyzer()
                result = current_analyzer.quick_analyze_stock(stock_code.strip(), market_type)
                
                # 确保推荐是字符串而不是对象
                if isinstance(result['recommendation'], dict):
                    result['recommendation'] = result['recommendation'].get('action', '无建议')
                elif result['recommendation'] is None:
                    result['recommendation'] = '无建议'

                app.logger.info(
                    f"分析结果: 股票={stock_code}, 名称={result.get('stock_name', '未知')}, 行业={result.get('industry', '未知')}")
                results.append(result)
            except Exception as e:
                app.logger.error(f"分析股票 {stock_code} 时出错: {str(e)}")
                results.append({
                    'stock_code': stock_code,
                    'error': str(e),
                    'stock_name': '分析失败',
                    'industry': '未知'
                })

        return jsonify({'results': results})
    except Exception as e:
        app.logger.error(f"分析股票时出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/north_flow_history', methods=['POST'])
def api_north_flow_history():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        days = data.get('days', 10)  # 默认为10天，对应前端的默认选项

        # 计算 end_date 为当前时间
        end_date = datetime.now().strftime('%Y%m%d')

        # 计算 start_date 为 end_date 减去指定的天数
        start_date = (datetime.now() - timedelta(days=int(days))).strftime('%Y%m%d')

        if not stock_code:
            return jsonify({'error': '请提供股票代码'}), 400

        # 调用北向资金历史数据方法
        from capital_flow_analyzer import CapitalFlowAnalyzer

        analyzer = CapitalFlowAnalyzer()
        result = analyzer.get_north_flow_history(stock_code, start_date, end_date)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"获取北向资金历史数据出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/search_us_stocks', methods=['GET'])
def search_us_stocks():
    try:
        keyword = request.args.get('keyword', '')
        if not keyword:
            return jsonify({'error': '请输入搜索关键词'}), 400

        results = us_stock_service.search_us_stocks(keyword)
        return jsonify({'results': results})

    except Exception as e:
        app.logger.error(f"搜索美股代码时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500


# 新增可视化分析页面路由
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/stock_detail/<string:stock_code>')
def stock_detail(stock_code):
    market_type = request.args.get('market_type', 'A')
    return render_template('stock_detail.html', stock_code=stock_code, market_type=market_type)


@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')


@app.route('/market_scan')
def market_scan():
    return render_template('market_scan.html')


# 基本面分析页面
@app.route('/fundamental')
def fundamental():
    return render_template('fundamental.html')


# 资金流向页面
@app.route('/capital_flow')
def capital_flow():
    return render_template('capital_flow.html')


# 情景预测页面
@app.route('/scenario_predict')
def scenario_predict():
    return render_template('scenario_predict.html')


# 风险监控页面
@app.route('/risk_monitor')
def risk_monitor_page():
    return render_template('risk_monitor.html')


# 智能问答页面
@app.route('/qa')
def qa_page():
    return render_template('qa.html')


# 行业分析页面
@app.route('/industry_analysis')
def industry_analysis():
    return render_template('industry_analysis.html')


def make_cache_key_with_stock():
    """创建包含股票代码的自定义缓存键"""
    path = request.path

    # 从请求体中获取股票代码
    stock_code = None
    if request.is_json:
        stock_code = request.json.get('stock_code')

    # 构建包含股票代码的键
    if stock_code:
        return f"{path}_{stock_code}"
    else:
        return path


@app.route('/api/start_stock_analysis', methods=['POST'])
def start_stock_analysis():
    """启动个股分析任务"""
    try:
        data = request.json
        stock_code = data.get('stock_code')
        market_type = data.get('market_type', 'A')

        if not stock_code:
            return jsonify({'error': '请输入股票代码'}), 400

        app.logger.info(f"准备分析股票: {stock_code}")

        # 获取或创建任务
        task_id, task, is_new = get_or_create_task(
            'stock_analysis',
            stock_code=stock_code,
            market_type=market_type
        )

        # 如果是已完成的任务，直接返回结果
        if task['status'] == TASK_COMPLETED and 'result' in task:
            app.logger.info(f"使用缓存的分析结果: {stock_code}")
            return jsonify({
                'task_id': task_id,
                'status': task['status'],
                'result': task['result']
            })

        # 如果是新创建的任务，启动后台处理
        if is_new:
            app.logger.info(f"创建新的分析任务: {task_id}")

            # 启动后台线程执行分析
            def run_analysis():
                try:
                    update_task_status('stock_analysis', task_id, TASK_RUNNING, progress=10)

                    # 执行分析
                    result = analyzer.perform_enhanced_analysis(stock_code, market_type)

                    # 确保recommendation格式正确
                    if result and 'recommendation' in result and isinstance(result['recommendation'], dict):
                        if 'action' in result['recommendation']:
                            action = result['recommendation']['action']
                            # 如果action本身是一个对象，提取其文本内容
                            if isinstance(action, dict) and 'action' in action:
                                result['recommendation']['action'] = action['action']
                    
                    # 更新任务状态为完成
                    update_task_status('stock_analysis', task_id, TASK_COMPLETED, progress=100, result=result)
                    app.logger.info(f"分析任务 {task_id} 完成")

                except Exception as e:
                    app.logger.error(f"分析任务 {task_id} 失败: {str(e)}")
                    app.logger.error(traceback.format_exc())
                    update_task_status('stock_analysis', task_id, TASK_FAILED, error=str(e))

            # 启动后台线程
            thread = threading.Thread(target=run_analysis)
            thread.daemon = True
            thread.start()

        # 返回任务ID和状态
        return jsonify({
            'task_id': task_id,
            'status': task['status'],
            'message': f'已启动分析任务: {stock_code}'
        })

    except Exception as e:
        app.logger.error(f"启动个股分析任务时出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis_status/<task_id>', methods=['GET'])
def get_analysis_status(task_id):
    """获取个股分析任务状态"""
    store = get_task_store('stock_analysis')
    with task_lock:
        if task_id not in store:
            return jsonify({'error': '找不到指定的分析任务'}), 404

        task = store[task_id]

        # 基本状态信息
        status = {
            'id': task['id'],
            'status': task['status'],
            'progress': task.get('progress', 0),
            'created_at': task['created_at'],
            'updated_at': task['updated_at']
        }

        # 如果任务完成，包含结果
        if task['status'] == TASK_COMPLETED and 'result' in task:
            status['result'] = task['result']

        # 如果任务失败，包含错误信息
        if task['status'] == TASK_FAILED and 'error' in task:
            status['error'] = task['error']

        return custom_jsonify(status)


@app.route('/api/cancel_analysis/<task_id>', methods=['POST'])
def cancel_analysis(task_id):
    """取消个股分析任务"""
    store = get_task_store('stock_analysis')
    with task_lock:
        if task_id not in store:
            return jsonify({'error': '找不到指定的分析任务'}), 404

        task = store[task_id]

        if task['status'] in [TASK_COMPLETED, TASK_FAILED]:
            return jsonify({'message': '任务已完成或失败，无法取消'})

        # 更新状态为失败
        task['status'] = TASK_FAILED
        task['error'] = '用户取消任务'
        task['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 更新键索引的任务
        if 'key' in task and task['key'] in store:
            store[task['key']] = task

        return jsonify({'message': '任务已取消'})


# 保留原有API用于向后兼容
@app.route('/api/enhanced_analysis', methods=['POST'])
def enhanced_analysis():
    """原增强分析API的向后兼容版本"""
    try:
        data = request.json
        stock_code = data.get('stock_code')
        market_type = data.get('market_type', 'A')

        if not stock_code:
            return custom_jsonify({'error': '请输入股票代码'}), 400

        # 调用新的任务系统，但模拟同步行为
        # 这会导致和之前一样的超时问题，但保持兼容
        timeout = 300
        start_time = time.time()

        # 获取或创建任务
        task_id, task, is_new = get_or_create_task(
            'stock_analysis',
            stock_code=stock_code,
            market_type=market_type
        )

        # 如果是已完成的任务，直接返回结果
        if task['status'] == TASK_COMPLETED and 'result' in task:
            app.logger.info(f"使用缓存的分析结果: {stock_code}")
            return custom_jsonify({'result': task['result']})

        # 启动分析（如果是新任务）
        if is_new:
            # 同步执行分析
            try:
                result = analyzer.perform_enhanced_analysis(stock_code, market_type)
                
                # 确保recommendation格式正确
                if result and 'recommendation' in result and isinstance(result['recommendation'], dict):
                    if 'action' in result['recommendation']:
                        action = result['recommendation']['action']
                        # 如果action本身是一个对象，提取其文本内容
                        if isinstance(action, dict) and 'action' in action:
                            result['recommendation']['action'] = action['action']
                
                update_task_status('stock_analysis', task_id, TASK_COMPLETED, progress=100, result=result)
                app.logger.info(f"分析完成: {stock_code}，耗时 {time.time() - start_time:.2f} 秒")
                return custom_jsonify({'result': result})
            except Exception as e:
                app.logger.error(f"分析过程中出错: {str(e)}")
                update_task_status('stock_analysis', task_id, TASK_FAILED, error=str(e))
                return custom_jsonify({'error': f'分析过程中出错: {str(e)}'}), 500
        else:
            # 已存在正在处理的任务，等待其完成
            max_wait = timeout - (time.time() - start_time)
            wait_interval = 0.5
            waited = 0

            while waited < max_wait:
                with task_lock:
                    current_task = store[task_id]
                    if current_task['status'] == TASK_COMPLETED and 'result' in current_task:
                        return custom_jsonify({'result': current_task['result']})
                    if current_task['status'] == TASK_FAILED:
                        error = current_task.get('error', '任务失败，无详细信息')
                        return custom_jsonify({'error': error}), 500

                time.sleep(wait_interval)
                waited += wait_interval

            # 超时
            return custom_jsonify({'error': '处理超时，请稍后重试'}), 504

    except Exception as e:
        app.logger.error(f"执行增强版分析时出错: {traceback.format_exc()}")
        return custom_jsonify({'error': str(e)}), 500


# 添加在web_server.py主代码中
@app.errorhandler(404)
def not_found(error):
    """处理404错误"""
    if request.path.startswith('/api/'):
        # 为API请求返回JSON格式的错误
        return jsonify({
            'error': '找不到请求的API端点',
            'path': request.path,
            'method': request.method
        }), 404
    # 为网页请求返回HTML错误页
    return render_template('error.html', error_code=404, message="找不到请求的页面"), 404


@app.errorhandler(500)
def server_error(error):
    """处理500错误"""
    app.logger.error(f"服务器错误: {str(error)}")
    if request.path.startswith('/api/'):
        # 为API请求返回JSON格式的错误
        return jsonify({
            'error': '服务器内部错误',
            'message': str(error)
        }), 500
    # 为网页请求返回HTML错误页
    return render_template('error.html', error_code=500, message="服务器内部错误"), 500


# Update the get_stock_data function in web_server.py to handle date formatting properly
@app.route('/api/stock_data', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_stock_data():
    try:
        stock_code = request.args.get('stock_code')
        market_type = request.args.get('market_type', 'A')
        period = request.args.get('period', '1y')  # 默认1年

        if not stock_code:
            return custom_jsonify({'error': '请提供股票代码'}), 400

        # 特殊处理US股票代码，比如添加105.前缀如果需要
        original_stock_code = stock_code
        if market_type == 'US':
            # 记录美股代码格式
            app.logger.info(f"处理美股代码: {stock_code}")
            
            # 美股代码不需要特殊处理，直接使用原始代码
            # USStockService内部将处理不同格式的美股代码
            # 这里只是记录日志以便于调试

        # 根据period计算start_date
        end_date = datetime.now().strftime('%Y%m%d')
        if period == '1m':
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        elif period == '3m':
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
        elif period == '6m':
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
        elif period == '1y':
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        else:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

        # 获取股票历史数据
        app.logger.info(
            f"获取股票 {stock_code} 的历史数据，市场: {market_type}, 起始日期: {start_date}, 结束日期: {end_date}")
        df = analyzer.get_stock_data(stock_code, market_type, start_date, end_date)

        # 计算技术指标
        app.logger.info(f"计算股票 {stock_code} 的技术指标")
        df = analyzer.calculate_indicators(df)

        # 检查数据是否为空
        if df.empty:
            app.logger.warning(f"股票 {stock_code} 的数据为空")
            return custom_jsonify({'error': '未找到股票数据'}), 404

        # 将DataFrame转为JSON格式
        app.logger.info(f"将数据转换为JSON格式，行数: {len(df)}")

        # 确保日期列是字符串格式 - 修复缓存问题
        if 'date' in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                else:
                    df = df.copy()
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            except Exception as e:
                app.logger.error(f"处理日期列时出错: {str(e)}")
                df['date'] = df['date'].astype(str)

        # 将NaN值替换为None
        df = df.replace({np.nan: None, np.inf: None, -np.inf: None})

        records = df.to_dict('records')

        app.logger.info(f"数据处理完成，返回 {len(records)} 条记录")
        return custom_jsonify({'data': records})
    except Exception as e:
        app.logger.error(f"获取股票数据时出错: {str(e)}")
        app.logger.error(traceback.format_exc())
        return custom_jsonify({'error': str(e)}), 500


# 添加美股搜索API端点
@app.route('/api/search_us_stocks', methods=['GET'])
def api_search_us_stocks():
    """搜索美股股票"""
    try:
        keyword = request.args.get('keyword', '')
        if not keyword:
            return jsonify({'error': '请提供搜索关键词'}), 400

        # 使用USStockService搜索美股
        from us_stock_service import USStockService
        us_service = USStockService()
        
        app.logger.info(f"搜索美股: {keyword}")
        results = us_service.search_us_stocks(keyword)
        
        app.logger.info(f"找到 {len(results)} 条美股匹配结果")
        return jsonify({'results': results})
    except Exception as e:
        app.logger.error(f"搜索美股时出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/start_market_scan', methods=['POST'])
def start_market_scan():
    """启动市场扫描任务"""
    try:
        data = request.json
        stock_list = data.get('stock_list', [])
        min_score = data.get('min_score', 60)
        market_type = data.get('market_type', 'A')

        if not stock_list:
            return jsonify({'error': '请提供股票列表'}), 400

        # 限制股票数量，避免过长处理时间
        if len(stock_list) > 100:
            app.logger.warning(f"股票列表过长 ({len(stock_list)}只)，截取前100只")
            stock_list = stock_list[:100]

        # 创建新任务
        task_id = generate_task_id()
        task = {
            'id': task_id,
            'status': TASK_PENDING,
            'progress': 0,
            'total': len(stock_list),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'params': {
                'stock_list': stock_list,
                'min_score': min_score,
                'market_type': market_type
            }
        }

        with task_lock:
            scan_tasks[task_id] = task

        # 启动后台线程执行扫描
        def run_scan():
            try:
                start_market_scan_task_status(task_id, TASK_RUNNING)

                # 执行分批处理
                results = []
                total = len(stock_list)
                batch_size = 10
                
                # 处理美股代码格式 (如果需要)
                processed_stock_list = stock_list.copy()
                if market_type == 'US':
                    app.logger.info(f"处理美股代码格式 {len(stock_list)} 只股票")
                    # 清理美股代码，确保格式正确
                    cleaned_list = []
                    for code in stock_list:
                        # 检查是否需要处理美股代码格式
                        if code.startswith('105.'):
                            # 已经是东方财富格式，保持不变
                            cleaned_list.append(code)
                        else:
                            # 普通美股代码，确保正确格式
                            cleaned_list.append(code)
                    processed_stock_list = cleaned_list

                for i in range(0, total, batch_size):
                    if task_id not in scan_tasks or scan_tasks[task_id]['status'] != TASK_RUNNING:
                        # 任务被取消
                        app.logger.info(f"扫描任务 {task_id} 被取消")
                        return

                    batch = processed_stock_list[i:i + batch_size]
                    batch_results = []

                    for stock_code in batch:
                        try:
                            # 获取线程本地分析器实例
                            current_analyzer = get_analyzer()
                            report = current_analyzer.quick_analyze_stock(stock_code, market_type)
                            
                            if report['score'] >= min_score:
                                # 对于美股，确保名称中包含美股标识
                                if market_type == 'US' and 'stock_name' in report:
                                    if not report['stock_name'].endswith('(US)'):
                                        report['stock_name'] = f"{report['stock_name']} (US)"
                                
                                # 确保推荐是字符串而不是对象
                                if isinstance(report['recommendation'], dict):
                                    report['recommendation'] = report['recommendation'].get('action', '无建议')
                                elif report['recommendation'] is None:
                                    report['recommendation'] = '无建议'
                                
                                batch_results.append(report)
                        except Exception as e:
                            app.logger.error(f"分析股票 {stock_code} 时出错: {str(e)}")
                            continue

                    results.extend(batch_results)

                    # 更新进度
                    progress = min(100, int((i + len(batch)) / total * 100))
                    start_market_scan_task_status(task_id, TASK_RUNNING, progress=progress)

                # 按得分排序
                results.sort(key=lambda x: x['score'], reverse=True)

                # 更新任务状态为完成
                start_market_scan_task_status(task_id, TASK_COMPLETED, progress=100, result=results)
                app.logger.info(f"扫描任务 {task_id} 完成，找到 {len(results)} 只符合条件的股票")

            except Exception as e:
                app.logger.error(f"扫描任务 {task_id} 失败: {str(e)}")
                app.logger.error(traceback.format_exc())
                start_market_scan_task_status(task_id, TASK_FAILED, error=str(e))

        # 启动后台线程
        thread = threading.Thread(target=run_scan)
        thread.daemon = True
        thread.start()

        return jsonify({
            'task_id': task_id,
            'status': 'pending',
            'message': f'已启动扫描任务，正在处理 {len(stock_list)} 只股票'
        })

    except Exception as e:
        app.logger.error(f"启动市场扫描任务时出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/scan_status/<task_id>', methods=['GET'])
def get_scan_status(task_id):
    """获取扫描任务状态"""
    with task_lock:
        if task_id not in scan_tasks:
            return jsonify({'error': '找不到指定的扫描任务'}), 404

        task = scan_tasks[task_id]

        # 基本状态信息
        status = {
            'id': task['id'],
            'status': task['status'],
            'progress': task.get('progress', 0),
            'total': task.get('total', 0),
            'created_at': task['created_at'],
            'updated_at': task['updated_at']
        }

        # 如果任务完成，包含结果
        if task['status'] == TASK_COMPLETED and 'result' in task:
            status['result'] = task['result']

        # 如果任务失败，包含错误信息
        if task['status'] == TASK_FAILED and 'error' in task:
            status['error'] = task['error']

        return custom_jsonify(status)


@app.route('/api/cancel_scan/<task_id>', methods=['POST'])
def cancel_scan(task_id):
    """取消扫描任务"""
    with task_lock:
        if task_id not in scan_tasks:
            return jsonify({'error': '找不到指定的扫描任务'}), 404

        task = scan_tasks[task_id]

        if task['status'] in [TASK_COMPLETED, TASK_FAILED]:
            return jsonify({'message': '任务已完成或失败，无法取消'})

        # 更新状态为失败
        task['status'] = TASK_FAILED
        task['error'] = '用户取消任务'
        task['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return jsonify({'message': '任务已取消'})


@app.route('/api/index_stocks', methods=['GET'])
def get_index_stocks():
    """获取指数成分股"""
    try:
        import akshare as ak
        index_code = request.args.get('index_code', '000300')  # 默认沪深300
        market_type = request.args.get('market_type', 'A')  # 获取市场类型

        # 检查是否是美股指数
        if market_type == 'US':
            app.logger.warning(f"美股市场暂不支持指数成分股的自动获取: {index_code}")
            return jsonify({'error': '美股市场暂不支持指数成分股的自动获取，请使用自定义股票代码功能'}), 400

        # 获取指数成分股
        app.logger.info(f"获取指数 {index_code} 成分股")
        if index_code == '000300':
            # 沪深300成分股
            stocks = ak.index_stock_cons_weight_csindex(symbol="000300")
        elif index_code == '000905':
            # 中证500成分股
            stocks = ak.index_stock_cons_weight_csindex(symbol="000905")
        elif index_code == '000852':
            # 中证1000成分股
            stocks = ak.index_stock_cons_weight_csindex(symbol="000852")
        elif index_code == '000001':
            # 上证指数
            stocks = ak.index_stock_cons_weight_csindex(symbol="000001")
        else:
            return jsonify({'error': '不支持的指数代码'}), 400

        # 提取股票代码列表
        stock_list = stocks['成分券代码'].tolist() if '成分券代码' in stocks.columns else []
        app.logger.info(f"找到 {len(stock_list)} 只成分股")

        return jsonify({'stock_list': stock_list})
    except Exception as e:
        app.logger.error(f"获取指数成分股时出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/popular_us_stocks', methods=['GET'])
def get_popular_us_stocks():
    """获取热门美股列表"""
    try:
        import akshare as ak
        
        # 使用akshare获取知名美股列表
        app.logger.info("获取知名美股列表")
        try:
            famous_stocks = ak.stock_us_famous_spot_em()
            # 提取股票代码列表
            stock_list = famous_stocks['代码'].tolist() if '代码' in famous_stocks.columns else []
            app.logger.info(f"找到 {len(stock_list)} 只知名美股")
            
            # 如果返回的列表为空或获取失败，使用备用列表
            if not stock_list:
                raise Exception("获取知名美股列表为空")
        except Exception as ak_e:
            app.logger.warning(f"通过akshare获取知名美股失败: {str(ak_e)}，使用备用列表")
            # 备用的热门美股列表
            stock_list = [
                # 科技股
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 
                # 金融股
                'JPM', 'BAC', 'WFC', 'GS', 
                # 医疗保健
                'JNJ', 'PFE', 'MRK', 
                # 消费品
                'KO', 'PEP', 'WMT', 'MCD',
                # 半导体
                'AMD', 'INTC', 'TSM', 'QCOM'
            ]
        
        return jsonify({'stock_list': stock_list})
    except Exception as e:
        app.logger.error(f"获取热门美股列表时出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/industry_stocks', methods=['GET'])
def get_industry_stocks():
    """获取行业成分股"""
    try:
        import akshare as ak
        industry = request.args.get('industry', '')
        market_type = request.args.get('market_type', 'A')  # 默认为A股

        if not industry:
            return jsonify({'error': '请提供行业名称'}), 400
            
        # 检查市场类型
        if market_type == 'US':
            app.logger.warning(f"美股市场暂不支持行业筛选: {industry}")
            return jsonify({'error': '美股市场暂不支持行业筛选功能，请使用自定义股票代码进行筛选'}), 400

        # 获取行业成分股
        app.logger.info(f"获取 {industry} 行业成分股")
        stocks = ak.stock_board_industry_cons_em(symbol=industry)

        # 提取股票代码列表
        stock_list = stocks['代码'].tolist() if '代码' in stocks.columns else []
        app.logger.info(f"找到 {len(stock_list)} 只 {industry} 行业股票")

        return jsonify({'stock_list': stock_list})
    except Exception as e:
        app.logger.error(f"获取行业成分股时出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 添加到web_server.py
def clean_old_tasks():
    """清理旧的扫描任务"""
    with task_lock:
        now = datetime.now()
        to_delete = []

        for task_id, task in scan_tasks.items():
            # 解析更新时间
            try:
                updated_at = datetime.strptime(task['updated_at'], '%Y-%m-%d %H:%M:%S')
                # 如果任务完成或失败且超过1小时，或者任务状态异常且超过3小时，清理它
                if ((task['status'] in [TASK_COMPLETED, TASK_FAILED] and
                     (now - updated_at).total_seconds() > 3600) or
                        ((now - updated_at).total_seconds() > 10800)):
                    to_delete.append(task_id)
            except:
                # 日期解析错误，添加到删除列表
                to_delete.append(task_id)

        # 删除旧任务
        for task_id in to_delete:
            del scan_tasks[task_id]

        return len(to_delete)


# 修改 run_task_cleaner 函数，使其每 5 分钟运行一次并在 16:30 左右清理所有缓存
def run_task_cleaner():
    """定期运行任务清理，并在每天 16:30 左右清理所有缓存"""
    while True:
        try:
            now = datetime.now()
            # 判断是否在收盘时间附近（16:25-16:35）
            is_market_close_time = (now.hour == 16 and 25 <= now.minute <= 35)

            cleaned = clean_old_tasks()

            # 如果是收盘时间，清理所有缓存
            if is_market_close_time:
                # 清理分析器的数据缓存
                analyzer.data_cache.clear()

                # 清理 Flask 缓存
                cache.clear()

                # 清理任务存储
                with task_lock:
                    for task_type in tasks:
                        task_store = tasks[task_type]
                        completed_tasks = [task_id for task_id, task in task_store.items()
                                           if task['status'] == TASK_COMPLETED]
                        for task_id in completed_tasks:
                            del task_store[task_id]

                app.logger.info("市场收盘时间检测到，已清理所有缓存数据")

            if cleaned > 0:
                app.logger.info(f"清理了 {cleaned} 个旧的扫描任务")
        except Exception as e:
            app.logger.error(f"任务清理出错: {str(e)}")

        # 每 5 分钟运行一次，而不是每小时
        time.sleep(600)


# 基本面分析路由
@app.route('/api/fundamental_analysis', methods=['POST'])
def api_fundamental_analysis():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        market_type = data.get('market_type', 'A')  # 添加市场类型参数，默认为A股

        if not stock_code:
            return jsonify({'error': '请提供股票代码'}), 400

        # 获取基本面分析结果，传递市场类型参数
        result = fundamental_analyzer.calculate_fundamental_score(stock_code, market_type)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"基本面分析出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 资金流向分析路由
# Add to web_server.py

# API endpoint to get concept fund flow
@app.route('/api/concept_fund_flow', methods=['GET'])
def api_concept_fund_flow():
    try:
        period = request.args.get('period', '90日排行')  # Default to 10-day ranking

        # Get concept fund flow data
        result = capital_flow_analyzer.get_concept_fund_flow(period)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"Error getting concept fund flow: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# API endpoint to get individual stock fund flow ranking
@app.route('/api/individual_fund_flow_rank', methods=['GET'])
def api_individual_fund_flow_rank():
    try:
        period = request.args.get('period', '10日')  # Default to today

        # Get individual fund flow ranking data
        result = capital_flow_analyzer.get_individual_fund_flow_rank(period)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"Error getting individual fund flow ranking: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# API endpoint to get individual stock fund flow
@app.route('/api/individual_fund_flow', methods=['GET'])
def api_individual_fund_flow():
    try:
        stock_code = request.args.get('stock_code')
        market_type = request.args.get('market_type', '')  # Auto-detect if not provided
        re_date = request.args.get('period-select')

        if not stock_code:
            return jsonify({'error': 'Stock code is required'}), 400

        # Get individual fund flow data
        result = capital_flow_analyzer.get_individual_fund_flow(stock_code, market_type, re_date)
        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"Error getting individual fund flow: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# API endpoint to get stocks in a sector
@app.route('/api/sector_stocks', methods=['GET'])
def api_sector_stocks():
    try:
        sector = request.args.get('sector')

        if not sector:
            return jsonify({'error': 'Sector name is required'}), 400

        # Get sector stocks data
        result = capital_flow_analyzer.get_sector_stocks(sector)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"Error getting sector stocks: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# Update the existing capital flow API endpoint
@app.route('/api/capital_flow', methods=['POST'])
def api_capital_flow():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        market_type = data.get('market_type', '')  # Auto-detect if not provided

        if not stock_code:
            return jsonify({'error': 'Stock code is required'}), 400

        # Calculate capital flow score
        result = capital_flow_analyzer.calculate_capital_flow_score(stock_code, market_type)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"Error calculating capital flow score: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 情景预测路由
@app.route('/api/scenario_predict', methods=['POST'])
def api_scenario_predict():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        market_type = data.get('market_type', 'A')
        days = data.get('days', 60)

        if not stock_code:
            return jsonify({'error': '请提供股票代码'}), 400

        # 获取情景预测结果
        result = scenario_predictor.generate_scenarios(stock_code, market_type, days)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"情景预测出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 智能问答路由
@app.route('/api/qa', methods=['POST'])
def api_qa():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        question = data.get('question')
        market_type = data.get('market_type', 'A')

        if not stock_code or not question:
            return jsonify({'error': '请提供股票代码和问题'}), 400

        # 获取智能问答结果
        result = stock_qa.answer_question(stock_code, question, market_type)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"智能问答出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 风险分析路由
@app.route('/api/risk_analysis', methods=['POST'])
def api_risk_analysis():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        market_type = data.get('market_type', 'A')

        if not stock_code:
            return jsonify({'error': '请提供股票代码'}), 400

        # 获取风险分析结果
        result = risk_monitor.analyze_stock_risk(stock_code, market_type)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"风险分析出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 投资组合风险分析路由
@app.route('/api/portfolio_risk', methods=['POST'])
def api_portfolio_risk():
    try:
        data = request.json
        portfolio = data.get('portfolio', [])

        if not portfolio:
            return jsonify({'error': '请提供投资组合'}), 400

        # 获取投资组合风险分析结果
        result = risk_monitor.analyze_portfolio_risk(portfolio)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"投资组合风险分析出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 指数分析路由
@app.route('/api/index_analysis', methods=['GET'])
def api_index_analysis():
    try:
        index_code = request.args.get('index_code')
        limit = int(request.args.get('limit', 30))

        if not index_code:
            return jsonify({'error': '请提供指数代码'}), 400

        # 获取指数分析结果
        result = index_industry_analyzer.analyze_index(index_code, limit)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"指数分析出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 行业分析路由
@app.route('/api/industry_analysis', methods=['GET'])
def api_industry_analysis():
    try:
        industry = request.args.get('industry')
        limit = int(request.args.get('limit', 30))

        if not industry:
            return jsonify({'error': '请提供行业名称'}), 400

        # 获取行业分析结果
        result = index_industry_analyzer.analyze_industry(industry, limit)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"行业分析出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/industry_fund_flow', methods=['GET'])
def api_industry_fund_flow():
    """获取行业资金流向数据"""
    try:
        symbol = request.args.get('symbol', '即时')

        result = industry_analyzer.get_industry_fund_flow(symbol)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"获取行业资金流向数据出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/industry_detail', methods=['GET'])
def api_industry_detail():
    """获取行业详细信息"""
    try:
        industry = request.args.get('industry')

        if not industry:
            return jsonify({'error': '请提供行业名称'}), 400

        result = industry_analyzer.get_industry_detail(industry)

        app.logger.info(f"返回前 (result)：{result}")
        if not result:
            return jsonify({'error': f'未找到行业 {industry} 的详细信息'}), 404

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"获取行业详细信息出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 行业比较路由
@app.route('/api/industry_compare', methods=['GET'])
def api_industry_compare():
    try:
        limit = int(request.args.get('limit', 10))

        # 获取行业比较结果
        result = index_industry_analyzer.compare_industries(limit)

        return custom_jsonify(result)
    except Exception as e:
        app.logger.error(f"行业比较出错: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# 保存股票分析结果到数据库
def save_analysis_result(stock_code, market_type, result):
    """保存分析结果到数据库"""
    if not USE_DATABASE:
        return

    try:
        session = get_session()

        # 创建新的分析结果记录
        analysis = AnalysisResult(
            stock_code=stock_code,
            market_type=market_type,
            score=result.get('scores', {}).get('total', 0),
            recommendation=result.get('recommendation', {}).get('action', ''),
            technical_data=result.get('technical_analysis', {}),
            fundamental_data=result.get('fundamental_data', {}),
            capital_flow_data=result.get('capital_flow_data', {}),
            ai_analysis=result.get('ai_analysis', '')
        )

        session.add(analysis)
        session.commit()

    except Exception as e:
        app.logger.error(f"保存分析结果到数据库时出错: {str(e)}")
        if session:
            session.rollback()
    finally:
        if session:
            session.close()


# 从数据库获取历史分析结果
@app.route('/api/history_analysis', methods=['GET'])
def get_history_analysis():
    """获取股票的历史分析结果"""
    if not USE_DATABASE:
        return jsonify({'error': '数据库功能未启用'}), 400

    stock_code = request.args.get('stock_code')
    limit = int(request.args.get('limit', 10))

    if not stock_code:
        return jsonify({'error': '请提供股票代码'}), 400

    try:
        session = get_session()

        # 查询历史分析结果
        results = session.query(AnalysisResult) \
            .filter(AnalysisResult.stock_code == stock_code) \
            .order_by(AnalysisResult.analysis_date.desc()) \
            .limit(limit) \
            .all()

        # 转换为字典列表
        history = [result.to_dict() for result in results]

        return jsonify({'history': history})

    except Exception as e:
        app.logger.error(f"获取历史分析结果时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if session:
            session.close()


# 在应用启动时启动清理线程（保持原有代码不变）
cleaner_thread = threading.Thread(target=run_task_cleaner)
cleaner_thread.daemon = True
cleaner_thread.start()

if __name__ == '__main__':
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=8888, debug=DEBUG)