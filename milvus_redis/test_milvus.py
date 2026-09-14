from pymilvus import MilvusClient, connections, utility
import redis
connections.connect(host="localhost", port="19530")
milvus = MilvusClient(uri="http://localhost:19530")
databases = milvus.list_databases()
if "edurag" not in databases:
    milvus.create_database(db_name="edurag")

milvus.using_database(db_name="edurag")
client = redis.Redis(
    host="localhost",
    port=8379,
    password="1234",
    decode_responses=True
)
client.set("test_key", "Hello, Redis!")
value = client.get("test_key")
print(f"Milvus version: {utility.get_server_version()}")
print(f"Milvus databases: {milvus.list_databases()}")
print(f"Using database: edurag")
print(f"Redis value: {value}")
