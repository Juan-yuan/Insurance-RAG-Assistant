import ast
import configparser
import os

def _load_env_files():
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
current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir_path)
config_file_path = os.path.join(project_root, 'config.ini')

class Config:

    def __init__(self, config_file=config_file_path):
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/data')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/models')
        self.EDU_DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/edu_document_loaders')
        if not config_file:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.config.read(config_file, encoding='utf-8')
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='localhost'))
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='root'))
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', self.config.get('mysql', 'password', fallback='123456'))
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE',
                                        self.config.get('mysql', 'database', fallback='subjects_kg'))
        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='localhost'))
        self.REDIS_PORT = int(os.getenv('REDIS_PORT') or self.config.get('redis', 'port', fallback='8379'))
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', self.config.get('redis', 'password', fallback='1234'))
        self.REDIS_DB = int(os.getenv('REDIS_DB') or self.config.get('redis', 'db', fallback='0'))
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback='19530'))
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME',
                                              self.config.get('milvus', 'database_name', fallback='edurag'))
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME',
                                                self.config.get('milvus', 'collection_name',
                                                                fallback='edurag_final'))
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='gpt-5-mini')
        ini_api_key = self.config.get('llm', 'openai_api_key', fallback='')
        self.OPENAI_API_KEY = _first_env('OPENAI_API_KEY', 'openai_api_key', default=ini_api_key)
        self.OPENAI_BASE_URL = self.config.get('llm', 'openai_base_url', fallback='https://api.openai.com/v1')
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)
        self.VALID_SOURCES = ast.literal_eval(
            self.config.get('app', 'valid_sources', fallback='["clause", "medical", "case"]'))
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='13112345678')
        self.LOG_FILE = self.config.get('logger', 'log_file', fallback='logs/app.log')

config = Config()

if __name__ == '__main__':
    config_file = '../config.ini'
    conf = Config(config_file)
    print(f'mysql: {conf.MYSQL_HOST}, {conf.MYSQL_USER}, {conf.MYSQL_PASSWORD}, {conf.MYSQL_DATABASE}')
    print(f'redis: {conf.REDIS_HOST}, {conf.REDIS_PORT}, {conf.REDIS_PASSWORD}, {conf.REDIS_DB}')
    print(f'log: {conf.LOG_FILE}')
