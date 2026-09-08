from rank_bm25 import BM25Okapi
import numpy as np
from mysql_qa.utils.preprocess import preprocess_text
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.cache.redis_client import RedisClient
from base import logger


# todo 1. Define the BM25 search class -> wrap data loading, model init, similarity scoring, and answer retrieval.
class BM25Search:
    # todo 1.1 Initializer -> client instances, model parameters, load data.
    def __init__(self, redis_client, mysql_client):
        # 1. Logger instance.
        self.logger = logger
        # 2. Redis client instance.
        self.redis_client = redis_client
        # 3. MySQL client instance.
        self.mysql_client = mysql_client
        # 4. BM25 model (initialized later).
        self.bm25 = None
        # 5. Tokenized question list.
        self.questions = None
        # 6. Original question list.
        self.original_questions = None
        # 7. Load question data and initialize the model.
        self._load_data()       # Leading _ means this is an internal method.


    # todo 1.2 Load data -> prefer Redis cache; on a miss, load from MySQL and refresh the cache.
    def _load_data(self):
        # 1. Redis cache keys for original vs tokenized questions.
        original_key = "qa_original_questions"      # Cache key for original questions
        tokenized_key = "qa_tokenized_questions"    # Cache key for tokenized questions

        # 2. Load original questions from Redis.
        self.original_questions = self.redis_client.get_data(original_key)
        # print(f'self.original_questions: {self.original_questions}')

        # 3. Load tokenized questions from Redis.
        tokenized_questions = self.redis_client.get_data(tokenized_key)
        # print(f'tokenized_questions: {tokenized_questions}')

        # 4. If Redis has no data (or it is incomplete), load from MySQL and update the cache.
        if not self.original_questions or not tokenized_questions:
            # 4.1 Fetch original questions from MySQL -> nested tuples, e.g. ((q1,), (q2,), ...)
            self.original_questions = self.mysql_client.fetch_questions()
            # print(f'self.original_questions: {self.original_questions}')

            # 4.2 If MySQL also has no questions, log a warning and return.
            if not self.original_questions:
                self.logger.warning("No question data loaded from MySQL")
                return

            # 4.3 Tokenize the original questions.
            # Original format is nested tuples, e.g. ((q1,), (q2,), ...)
            tokenized_questions = [preprocess_text(q[0]) for q in self.original_questions]
            # print(f'tokenized_questions: {tokenized_questions}')

            # 4.4 Flatten original questions to a string list and cache in Redis.
            self.redis_client.set_data(original_key, [(q[0]) for q in self.original_questions])

            # 4.5 Cache tokenized questions in Redis so they can be reused next time.
            self.redis_client.set_data(tokenized_key, tokenized_questions)

        # 5. Keep the tokenized list for BM25 initialization.
        self.questions = tokenized_questions
        # 6. Initialize BM25 using the tokenized questions as the corpus.
        self.bm25 = BM25Okapi(self.questions)
        # 7. Log that the model is ready.
        self.logger.info("BM25 model initialized")

    # todo 1.3 Softmax normalization -> turn BM25 scores into a probability distribution for thresholding.
    def _softmax(self, scores):
        # Normalize scores with softmax so they sum to 1.
        # 1. Subtract the max before exp to avoid overflow.
        exp_scores = np.exp(scores - np.max(scores))
        # 2. Return the normalized result.
        return exp_scores / np.sum(exp_scores)


    # todo 1.4 Core search: process the query, score similarity, return a matched answer or no-result.
    def search(self, query, threshold=0.85):
        """
        Retrieve the most similar question for the input query and return its answer.
        :param query: user query text
        :param threshold: similarity threshold; scores at or above this count as a match
        :return: on match: (answer, False); on miss: (None, True)
        """
        # 1. Reject empty or non-string queries.
        if not query or not isinstance(query, str):
            self.logger.error('Invalid query: empty or not a string')
            return None, True

        # 2. Check Redis first; return immediately on a cache hit.
        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            # Cache hit: answer came from Redis.
            return cached_answer, False     # False means not a new query (cache hit)

        # Cache miss: search the database.
        try:
            # 3. Preprocess the query with the same tokenization as the question bank.
            query_tokens = preprocess_text(query)
            # 4. Compute BM25 similarity against all questions.
            scores = self.bm25.get_scores(query_tokens)
            # print(f'scores: {scores}')
            # 5. Normalize scores into a probability distribution.
            softmax_score = self._softmax(scores)
            # print(f'softmax_score: {softmax_score}')
            # 6. Find the index and score of the best match.
            best_idx = softmax_score.argmax()       # Index of the highest score.
            best_score = softmax_score[best_idx]    # Highest score.
            print(f'best_idx: {best_idx}, best_score: {best_score}')

            # 7. Treat it as a match if the best score meets the threshold.
            if best_score >= threshold:
                # 7.1 Look up the original question at that index.
                original_question = self.original_questions[best_idx]
                # 7.2 Fetch the answer from MySQL.
                answer = self.mysql_client.fetch_answer(original_question)
                if answer:
                    # 7.3 Cache the query-answer pair in Redis for later queries.
                    self.redis_client.set_data(f'answer:{query}', answer)
                    # 7.4 Log a successful search and the similarity.
                    self.logger.info(f'Search succeeded, similarity: {best_score:.3f}')
                    return answer, False     # False: not a new query; this answer is now cached.

            # 8. Below threshold: log and return no match.
            self.logger.info(f'No reliable answer found, best match: {best_score:.3f} (below threshold {threshold:.3f})')
            return None, True               # True: new query; needs further handling (e.g. RAG)

        except Exception as e:
            self.logger.error(f'Database query error: {e}')
            return None, True



# todo 2. Test code.
if __name__ == '__main__':
    # 1. Create Redis and MySQL clients.
    redis_client = RedisClient()
    mysql_client = MySQLClient()

    # 2. Create a BM25 search instance.
    bm25_search = BM25Search(redis_client, mysql_client)

    # 3. Test search.
    result = bm25_search.search(query='lxml tree error')
    print(f'result: \n{result}')
