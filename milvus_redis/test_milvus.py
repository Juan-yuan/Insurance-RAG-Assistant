from pymilvus import connections, utility
import redis

connections.connect(host="localhost", port="19530")

# Connect to Redis
client = redis.Redis(
    host="localhost",
    port=8379,
    password="1234",
    decode_responses=True
)

# Test read/write
client.set("test_key", "Hello, Redis!")
value = client.get("test_key")

print(f"Milvus version: {utility.get_server_version()}")
print(f"Redis value: {value}")