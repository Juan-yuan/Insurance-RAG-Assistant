from rank_bm25 import BM25Okapi
import numpy as np
from mysql_qa.utils.preprocess import preprocess_text
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.cache.redis_client import RedisClient
from insurance_rag import logger

class BM25Search:

    def __init__(self, redis_client, mysql_client):
        self.logger = logger
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        self.bm25 = None
        self.questions = None
        self.original_questions = None
        self._load_data()

    def _load_data(self):
        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"
        self.original_questions = self.redis_client.get_data(original_key)
        tokenized_questions = self.redis_client.get_data(tokenized_key)
        if not self.original_questions or not tokenized_questions:
            self.original_questions = self.mysql_client.fetch_questions()
            if not self.original_questions:
                self.logger.warning("No question data loaded from MySQL")
                return
            tokenized_questions = [preprocess_text(q[0]) for q in self.original_questions]
            self.redis_client.set_data(original_key, [(q[0]) for q in self.original_questions])
            self.redis_client.set_data(tokenized_key, tokenized_questions)
        self.questions = tokenized_questions
        self.bm25 = BM25Okapi(self.questions)
        self.logger.info("BM25 model initialized")

    def _softmax(self, scores):
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / np.sum(exp_scores)

    def search(self, query, threshold=0.85):
        if not query or not isinstance(query, str):
            self.logger.error('Invalid query: empty or not a string')
            return None, True
        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            return cached_answer, False
        try:
            query_tokens = preprocess_text(query)
            scores = self.bm25.get_scores(query_tokens)
            softmax_score = self._softmax(scores)
            best_idx = softmax_score.argmax()
            best_score = softmax_score[best_idx]
            print(f'best_idx: {best_idx}, best_score: {best_score}')
            if best_score >= threshold:
                original_question = self.original_questions[best_idx]
                answer = self.mysql_client.fetch_answer(original_question)
                if answer:
                    self.redis_client.set_data(f'answer:{query}', answer)
                    self.logger.info(f'Search succeeded, similarity: {best_score:.3f}')
                    return answer, False
            self.logger.info(f'No reliable answer found, best match: {best_score:.3f} (below threshold {threshold:.3f})')
            return None, True
        except Exception as e:
            self.logger.error(f'Database query error: {e}')
            return None, True

if __name__ == '__main__':
    redis_client = RedisClient()
    mysql_client = MySQLClient()
    bm25_search = BM25Search(redis_client, mysql_client)
    result = bm25_search.search(query='lxml tree error')
    print(f'result: \n{result}')
