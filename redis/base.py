import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from runtime_env import require_env, require_int


class Config:
    REDIS_HOST = require_env('REDIS_HOST')
    REDIS_PORT = require_int('REDIS_PORT')
    REDIS_PASSWORD = require_env('REDIS_PASSWORD')
    REDIS_DB = require_int('REDIS_DB')
    REDIS_ANSWER_PREFIX = require_env('REDIS_ANSWER_PREFIX')


if __name__ == '__main__':
    print('runtime configuration loaded')
