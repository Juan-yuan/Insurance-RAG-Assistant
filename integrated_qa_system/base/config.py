# This script is used for: configuration file management.

# Imports
import ast
import configparser                 # Import config file parser library, parse INI-format config files.
import os                           # Import path operation module


def _load_env_files():
    """Load repo .env files into os.environ without overriding existing vars."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, '..', '..', '.env')),
        os.path.abspath(os.path.join(here, '..', '.env')),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path, encoding='utf-8') as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and not os.environ.get(key):
                    os.environ[key] = value
    lowercase_key = os.environ.get('openai_api_key', '').strip()
    if lowercase_key and not os.environ.get('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = lowercase_key


def _first_env(*names, default=''):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


_load_env_files()


# todo 1. Compute config file path.
# 1. Get the absolute path of the current file.
current_file_path = os.path.abspath(__file__)
# print(f'current_file_path: {current_file_path}')    # D:\workspace\ai_06_line\integrated_qa_system\base\config.py

# 2. Get the absolute path of the directory containing the current file.
current_dir_path = os.path.dirname(current_file_path)
# print(f'current_dir_path: {current_dir_path}')      # D:\workspace\ai_06_line\integrated_qa_system\base

# 3. Get the absolute path of the project root directory
project_root = os.path.dirname(current_dir_path)
# print(f'project_root: {project_root}')                # D:\workspace\ai_06_line\integrated_qa_system

# 4. Join the full path of the config file (config.ini).
config_file_path = os.path.join(project_root, 'config.ini')
# print(f'config_file_path: {config_file_path}')          # D:\workspace\ai_06_line\integrated_qa_system\config.ini


# todo 2. Config parser class -> encapsulate config file reading logic, provide a unified interface for accessing config parameters.
class Config:
    # todo 2.1 Initializer: read the config file and parse config parameters for each service.
    def __init__(self, config_file=config_file_path):
        # Project paths
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/data')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/models')
        self.EDU_DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/edu_document_loaders')

        # If no config file path is provided, use the default path.
        if not config_file:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')

        # Create a config parser with interpolation enabled, then read the ini file once.
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.config.read(config_file, encoding='utf-8')

        # MySQL config (environment variables override ini)
        # MySQL host address
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='localhost'))
        # MySQL username
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='root'))
        # MySQL password
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', self.config.get('mysql', 'password', fallback='123456'))
        # MySQL database name
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE',
                                        self.config.get('mysql', 'database', fallback='subjects_kg'))

        # Redis config. Docker maps container 6379 to host 8379.
        # Redis host address
        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='localhost'))
        # Redis port
        self.REDIS_PORT = int(os.getenv('REDIS_PORT') or self.config.get('redis', 'port', fallback='8379'))
        # Redis password
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', self.config.get('redis', 'password', fallback='1234'))
        # Redis database index
        self.REDIS_DB = int(os.getenv('REDIS_DB') or self.config.get('redis', 'db', fallback='0'))

        # Milvus config
        # Milvus host address
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        # Milvus port
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback='19530'))
        # Milvus database name
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME',
                                              self.config.get('milvus', 'database_name', fallback='edurag'))
        # Milvus collection name
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME',
                                                self.config.get('milvus', 'collection_name',
                                                                fallback='edurag_final'))

        # OpenAI LLM model
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='gpt-5-mini')
        # OpenAI API key (.env may use OPENAI_API_KEY or openai_api_key)
        ini_api_key = self.config.get('llm', 'openai_api_key', fallback='')
        self.OPENAI_API_KEY = _first_env('OPENAI_API_KEY', 'openai_api_key', default=ini_api_key)
        # OpenAI API base URL
        self.OPENAI_BASE_URL = self.config.get('llm', 'openai_base_url', fallback='https://api.openai.com/v1')

        # Retrieval parameters
        # Parent chunk size
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # Child chunk size
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # Chunk overlap size
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # Number of retrieved results
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # Number of final candidates
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # App config
        # List of valid sources
        self.VALID_SOURCES = ast.literal_eval(
            self.config.get('app', 'valid_sources', fallback='["clause", "medical", "case"]'))
        # Customer service phone
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='13112345678')

        # Log file path (relative, so logger.py can join it with the project root)
        self.LOG_FILE = self.config.get('logger', 'log_file', fallback='logs/app.log')

# Extension: create an instance of the Config class.
config = Config()

# todo 3. Main entry -> test code.
if __name__ == '__main__':
    # 1. Manually specify the config file path -> for testing; can be ignored in actual use.
    # config_file = 'D:/workspace/ai_30_bj/integrated_qa_system/config.ini'      # Absolute path.
    config_file = '../config.ini'      # Relative path.

    # 2. Instantiate the Config class -> load and parse the config file.
    conf = Config(config_file)
    # 3. Test: get config parameters for each service.
    print(f'mysql: {conf.MYSQL_HOST}, {conf.MYSQL_USER}, {conf.MYSQL_PASSWORD}, {conf.MYSQL_DATABASE}')
    print(f'redis: {conf.REDIS_HOST}, {conf.REDIS_PORT}, {conf.REDIS_PASSWORD}, {conf.REDIS_DB}')
    print(f'log: {conf.LOG_FILE}')
