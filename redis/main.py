from redis_client import RedisClient
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    redis_client = RedisClient()
    key = 'answer:tell me about yourself?'
    value = {"name": 'Lily', "age": 20}
    keys = redis_client.get_keys()
    print(keys)
    print('-' * 30)
    print(redis_client.get_answer('datetime update to string'))
    print('♥️' * 30)
    print(redis_client.get_answer('datetime update to string'))
