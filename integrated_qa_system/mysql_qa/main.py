import os
import sys
import time
current_dir = os.path.dirname(os.path.abspath(__file__))   # mysql_qa
project_root = os.path.dirname(current_dir)                # integrated_qa_system
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

from db.mysql_client import MySQLClient
from cache.redis_client import RedisClient
from retrieval.bm25_search import BM25Search
from base import logger

class MySQLQASystem:
    # todo 1.1 Initialization.
    def __init__(self):
        self.logger = logger
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)


    def query(self, query):
        """
        Handle a user query: search MySQL via BM25, return the answer, and log the result.
        :param query: user query text
        :return: matched answer, or a default message if none is found
        """
        start_time = time.time()
        self.logger.info(f'Processing query: {query}')
        answer, _ = self.bm25_search.search(query, threshold=0.85)
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


# todo 3. Test entry point.
if __name__ == '__main__':
    main()
