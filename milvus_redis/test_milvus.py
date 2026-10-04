import os
import sys

from pymilvus import MilvusClient, connections, utility
import redis

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from runtime_env import require_env, require_int

milvus_host = require_env('MILVUS_HOST')
milvus_port = require_env('MILVUS_PORT')
milvus_uri = require_env('MILVUS_URI')
milvus_database = require_env('MILVUS_DATABASE_NAME')
connections.connect(host=milvus_host, port=milvus_port)
milvus = MilvusClient(uri=milvus_uri)
databases = milvus.list_databases()
if milvus_database not in databases:
    milvus.create_database(db_name=milvus_database)

milvus.using_database(db_name=milvus_database)
client = redis.Redis(
    host=require_env('REDIS_HOST'),
    port=require_int('REDIS_PORT'),
    password=require_env('REDIS_PASSWORD'),
    decode_responses=True,
)
client.set('test_key', 'Hello, Redis!')
value = client.get('test_key')
print(f'Milvus version: {utility.get_server_version()}')
print(f'Milvus databases: {milvus.list_databases()}')
print(f'Using database: {milvus_database}')
print(f'Redis value: {value}')
