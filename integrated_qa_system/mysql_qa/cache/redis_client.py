import redis
import json
import os, sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(module_dir)
sys.path.insert(0, project_root)

from base import config, logger


class RedisClient:
    def __init__(self):
        self.logger = logger
        try:
            self.client = redis.StrictRedis(
                host=config.REDIS_HOST,
                port=int(config.REDIS_PORT),
                password=config.REDIS_PASSWORD,
                db=int(config.REDIS_DB),
                decode_responses=True,
            )
            # StrictRedis does not connect until a command is issued.
            self.client.ping()
        except redis.RedisError as e:
            self.logger.error(f'Redis connection failed: {e}')
            raise

        self.logger.info('Redis connected successfully')

    def set_data(self, key, value):
        try:
            # ensure_ascii=False keeps Chinese and other non-ASCII characters readable.
            self.client.set(key, json.dumps(value, ensure_ascii=False))
            self.logger.info(f'Redis set succeeded: {key}')
        except redis.RedisError as e:
            self.logger.error(f'Redis set failed: {e}')

    def get_data(self, key):
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f'Redis get failed: {e}')
            return None

    def get_answer(self, query):
        try:
            answer = self.client.get(f"answer:{query}")
            if answer:
                self.logger.info(f'Redis answer hit: {query}')
                # Values written by set_data() are JSON; decode before returning.
                try:
                    return json.loads(answer)
                except (json.JSONDecodeError, TypeError):
                    return answer
            return None
        except redis.RedisError as e:
            self.logger.error(f'Redis get answer failed: {e}')
            return None

    def load_csv_data(self, csv_path):
        try:
            data = pd.read_csv(csv_path)
            questions = []
            count = 0

            pipe = self.client.pipeline()
            for _, row in data.iterrows():
                question = str(row['问题']).strip() if pd.notna(row['问题']) else ''
                answer = row['答案'] if pd.notna(row['答案']) else ''
                if not question:
                    continue
                questions.append(question)
                # Key format must match get_answer(): answer:{question}
                pipe.set(f'answer:{question}', json.dumps(answer, ensure_ascii=False))
                count += 1

            # Cached for BM25 retrieval.
            pipe.set('qa_original_questions', json.dumps(questions, ensure_ascii=False))
            pipe.execute()

            self.logger.info(f'Redis loaded {count} Q&A rows from {csv_path}')
            return count
        except Exception as e:
            self.logger.error(f'Redis CSV load failed: {e}')
            raise


if __name__ == '__main__':
    redCli = RedisClient()

    csv_path = os.path.join(module_dir, 'data', '../data/JP_questions.csv')
    loaded = redCli.load_csv_data(csv_path)

    keys = redCli.client.keys("*")
    print(f'Loaded {loaded} Q&A rows, Redis key count: {len(keys)}')
    sample_question = 'How to calculate function execution time using a context manager?'
    print(redCli.get_answer(sample_question))
