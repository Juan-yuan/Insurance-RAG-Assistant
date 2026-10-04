import os
import sys
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)
from db.mysql_client import MySQLClient
from cache.redis_client import RedisClient
from retrieval.bm25_search import BM25Search
from insurance_rag import logger

class MySQLQASystem:

    def __init__(self):
        self.logger = logger
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

    def query(self, query):
        start_time = time.time()
        self.logger.info(f'Processing query: {query}')
        answer, _ = self.bm25_search.search(query)
        if answer:
            self.logger.info(f'MySQL answer: {answer}')
        else:
            self.logger.info('No answer found in SQL, RAG is needed')
            answer = 'No answer found in SQL'
        process_time = time.time() - start_time
        self.logger.info(f'Query processing time: {process_time:.3f}s')
        return answer

def main():
    mysql_qa = MySQLQASystem()
    try:
        print('\nWelcome to the MySQL Q&A system:')
        print('Enter a query to get an answer, or type exit to quit.')
        while True:
            query = input('\nEnter query: ').strip()
            if query.lower() == 'exit':
                logger.info('Exiting the MySQL system')
                print('Goodbye!')
                break
            answer = mysql_qa.query(query)
            print(f'Answer: {answer}')
    except Exception as e:
        logger.error(f'System error: {e}')
    finally:
        mysql_qa.mysql_client.close()

if __name__ == '__main__':
    main()
