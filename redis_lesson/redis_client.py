import redis
import json
from base import Config, logger

class RedisClient:
    def __init__(self):
        self.logger = logger

        try:
            self.client = redis.StrictRedis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                password=Config.REDIS_PASSWORD,
                db=Config.REDIS_DB,
                decode_responses=True
            )
        except redis.RedisError as e:
            logger.error(f'Redis connection failed: {e}')
            raise

    def set_data(self, key, value):
        try:
            self.client.set(key, json.dumps(value, ensure_ascii=False))
            self.logger.info(f'Redis store data success: {key}')
        except redis.RedisError as e:
            self.logger.error(f'Redis failed: {e}')


    def get_data(self, key):
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f'Redis get data failed: {e}')
            return None


    def get_answer(self, query):
        try:
            answer = self.client.get(f'answer:{query}')
            if answer:
                self.logger.info(f'Get data from Redis : {query}')
                return answer
            return None
        except redis.RedisError as e:
            self.logger.error(f'Redis query data failed: {e}')
            return None


    def delete_key(self, key):
        try:
            result = self.client.delete(key)
            if result == 1:
                self.logger.info(f'delete Redis data success: {key}')
            else:
                self.logger.info(f'key is not in redis : {key}')
            return result

        except redis.RedisError as e:
            self.logger.error(f'Redis delete data failed: {e}')


    def get_keys(self):
        return self.client.keys("*")



if __name__ == '__main__':
    redis_client = RedisClient()

    key = 'answer:tell me about yourself ?'
    value = { "name": 'Lily', "age": 21 }

    # redis_client.set_data(key, value)

    # data = redis_client.get_data(key)
    # print(data)

    # redis_client.delete_key(key)

    # answer = redis_client.get_answer('tell me about yourself ?')
    # print(answer)

    keys = redis_client.get_keys()
    print(keys)
    print('-' * 30)

    print(redis_client.get_answer('datetime update to string'))
    print('♥️' * 30)
    print(redis_client.get_answer('datetime update to string'))
