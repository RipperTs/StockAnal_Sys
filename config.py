import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

OPENAI_API_URL = os.getenv('OPENAI_API_URL', 'https://api.openai.com/v1')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_API_MODEL = os.getenv('OPENAI_API_MODEL', 'gemini-2.0-pro-exp-02-05')
NEWS_MODEL = os.getenv('NEWS_MODEL', 'bot-20250310130334-255bb')

USE_REDIS_CACHE = os.getenv('USE_REDIS_CACHE', 'false').lower() == 'true'
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

PORT = int(os.getenv('PORT', 8888))

BOCHA_API_KEY = os.getenv('BOCHA_API_KEY','')

XUEQIU_COOKIE = os.getenv('XUEQIU_COOKIE','')