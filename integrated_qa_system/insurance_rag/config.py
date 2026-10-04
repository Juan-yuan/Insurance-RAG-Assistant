import ast
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from runtime_env import load_env, require_env, require_int

load_env()


class Config:

    def __init__(self):
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/data')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/models')
        self.EDU_DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/edu_document_loaders')
        self.MYSQL_HOST = require_env('MYSQL_HOST')
        self.MYSQL_USER = require_env('MYSQL_USER')
        self.MYSQL_PASSWORD = require_env('MYSQL_PASSWORD')
        self.MYSQL_DATABASE = require_env('MYSQL_DATABASE')
        self.REDIS_HOST = require_env('REDIS_HOST')
        self.REDIS_PORT = require_int('REDIS_PORT')
        self.REDIS_PASSWORD = require_env('REDIS_PASSWORD')
        self.REDIS_DB = require_int('REDIS_DB')
        self.REDIS_QA_ORIGINAL_KEY = require_env('REDIS_QA_ORIGINAL_KEY')
        self.REDIS_QA_TOKENIZED_KEY = require_env('REDIS_QA_TOKENIZED_KEY')
        self.REDIS_ANSWER_PREFIX = require_env('REDIS_ANSWER_PREFIX')
        self.MILVUS_HOST = require_env('MILVUS_HOST')
        self.MILVUS_PORT = require_env('MILVUS_PORT')
        self.MILVUS_DATABASE_NAME = require_env('MILVUS_DATABASE_NAME')
        self.MILVUS_COLLECTION_NAME = require_env('MILVUS_COLLECTION_NAME')
        self.MILVUS_URI = require_env('MILVUS_URI')
        self.LLM_MODEL = require_env('LLM_MODEL')
        self.OPENAI_API_KEY = require_env('OPENAI_API_KEY')
        self.OPENAI_BASE_URL = require_env('OPENAI_BASE_URL')
        self.EMBEDDING_MODEL = require_env('EMBEDDING_MODEL')
        self.BGE_M3_MODEL = require_env('BGE_M3_MODEL')
        self.BGE_RERANKER_MODEL = require_env('BGE_RERANKER_MODEL')
        self.BERT_MODEL_NAME = require_env('BERT_MODEL_NAME')
        self.BERT_MODEL_ALT = require_env('BERT_MODEL_ALT')
        self.PARENT_CHUNK_SIZE = require_int('PARENT_CHUNK_SIZE')
        self.CHILD_CHUNK_SIZE = require_int('CHILD_CHUNK_SIZE')
        self.CHUNK_OVERLAP = require_int('CHUNK_OVERLAP')
        self.RETRIEVAL_K = require_int('RETRIEVAL_K')
        self.CANDIDATE_M = require_int('CANDIDATE_M')
        self.BM25_THRESHOLD = float(require_env('BM25_THRESHOLD'))
        self.VALID_SOURCES = ast.literal_eval(require_env('VALID_SOURCES'))
        if not isinstance(self.VALID_SOURCES, list) or not self.VALID_SOURCES:
            raise RuntimeError('Missing required runtime configuration: VALID_SOURCES')
        self.CUSTOMER_SERVICE_PHONE = require_env('CUSTOMER_SERVICE_PHONE')
        self.LOG_FILE = require_env('LOG_FILE')
        self.HOST = require_env('HOST')
        self.PORT = require_int('PORT')
        self.LLM_SYSTEM_PROMPT = require_env('LLM_SYSTEM_PROMPT')
        self.LLM_ROUTING_SYSTEM_PROMPT = require_env('LLM_ROUTING_SYSTEM_PROMPT')
        self.LLM_STREAM_SYSTEM_PROMPT = require_env('LLM_STREAM_SYSTEM_PROMPT')
        self.PROMPT_RAG = require_env('PROMPT_RAG')
        self.PROMPT_RAG_WITH_HISTORY = require_env('PROMPT_RAG_WITH_HISTORY')
        self.PROMPT_HYDE = require_env('PROMPT_HYDE')
        self.PROMPT_SUBQUERY = require_env('PROMPT_SUBQUERY')
        self.PROMPT_BACKTRACK = require_env('PROMPT_BACKTRACK')
        self.PROMPT_STRATEGY = require_env('PROMPT_STRATEGY')
        self.PROMPT_INTENT = require_env('PROMPT_INTENT')
        self.PROMPT_CLASSIFIER = require_env('PROMPT_CLASSIFIER')
        self.GREETING_RULES = json.loads(require_env('GREETING_RULES'))
        self.FALLBACK_REVIEW = require_env('FALLBACK_REVIEW')
        self.FALLBACK_GENERAL_ERROR = require_env('FALLBACK_GENERAL_ERROR')
        self.FALLBACK_CLAIMS_ERROR = require_env('FALLBACK_CLAIMS_ERROR')
        self.SQL_CREATE_JPKB = require_env('SQL_CREATE_JPKB')
        self.SQL_INSERT_JPKB = require_env('SQL_INSERT_JPKB')
        self.SQL_SELECT_QUESTIONS = require_env('SQL_SELECT_QUESTIONS')
        self.SQL_SELECT_ANSWER = require_env('SQL_SELECT_ANSWER')
        self.SQL_CREATE_CONVERSATIONS = require_env('SQL_CREATE_CONVERSATIONS')
        self.SQL_SELECT_RECENT_HISTORY = require_env('SQL_SELECT_RECENT_HISTORY')
        self.SQL_INSERT_CONVERSATION = require_env('SQL_INSERT_CONVERSATION')
        self.SQL_DELETE_OLD_CONVERSATIONS = require_env('SQL_DELETE_OLD_CONVERSATIONS')
        self.SQL_DELETE_SESSION = require_env('SQL_DELETE_SESSION')
        os.environ['OPENAI_API_KEY'] = self.OPENAI_API_KEY


config = Config()

if __name__ == '__main__':
    print('runtime configuration loaded')
