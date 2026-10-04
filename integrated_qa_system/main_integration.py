import os
import sys
import time
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_HERE, "integrated_qa_system")
sys.path.insert(0, _HERE)
sys.path.insert(0, _PROJECT_ROOT)
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.retrieval.bm25_search import BM25Search
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem
from insurance_rag.config import config
from insurance_rag.logger import logger
from openai import OpenAI

def _openai_api_key():
    key = (config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key") or "").strip()
    if key:
        os.environ["OPENAI_API_KEY"] = key
    return key

class IntegratedQASystem:
    def __init__(self):
        self.logger = logger
        self.config = config
        self.mysql_client = None
        self.redis_client = None
        self.bm25_search = None
        try:
            self.mysql_client = MySQLClient()
            self.redis_client = RedisClient()
            self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
        except Exception as e:
            self.logger.error(f"MySQL/Redis/BM25 failed to initialize; queries will fall back to RAG: {e}")
        api_key = _openai_api_key()
        if not api_key:
            self.logger.error("Missing required runtime configuration.")
        try:
            self.client = OpenAI(api_key=api_key or None, base_url=self.config.OPENAI_BASE_URL)
        except Exception as e:
            self.logger.error(f"Failed to initialize the OpenAI client: {e}")
            raise
        self.vector_store = None
        try:
            self.vector_store = VectorStore()
        except Exception as e:
            self.logger.error(f"VectorStore failed to initialize; RAG will not retrieve documents: {e}")
            self.vector_store = None
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)

    def call_dashscope(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": self.config.LLM_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            if completion.choices and completion.choices[0].message:
                return (completion.choices[0].message.content or "").strip()
            return ""
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return ""

    def query(self, query, source_filter=None):
        start_time = time.time()
        self.logger.info(f"Handling query: '{query}'")
        answer, need_rag = None, True
        if self.bm25_search is not None and getattr(self.bm25_search, "bm25", None) is not None:
            answer, need_rag = self.bm25_search.search(query, threshold=self.config.BM25_THRESHOLD)
        if answer:
            self.logger.info(f"MySQL answer: {answer}")
            processing_time = time.time() - start_time
            self.logger.info(f"Query handling time {processing_time:.2f}s")
            return answer
        elif need_rag:
            self.logger.info("No reliable MySQL answer; falling back to RAG")
            answer = self.rag_system.generate_answer(query, source_filter=source_filter)
            self.logger.info(f"RAG answer: {answer}")
            processing_time = time.time() - start_time
            self.logger.info(f"Query handling time {processing_time:.2f}s")
            return answer
        else:
            self.logger.info("No answer found")
            processing_time = time.time() - start_time
            self.logger.info(f"Query handling time {processing_time:.2f}s")
            return "No answer found"

def main():
    qa_system = IntegratedQASystem()
    try:
        print("\nWelcome to Insurance RAG Assistant.")
        print(f"Supported sources: {qa_system.config.VALID_SOURCES}")
        print("Enter a claims question, or type 'exit' to quit.")
        while True:
            query = input("\nQuestion: ").strip()
            if query.lower() == "exit":
                logger.info("Exiting")
                break
            source_filter = input(
                f"Source filter ({'/'.join(qa_system.config.VALID_SOURCES)}; press Enter to skip): "
            ).strip()
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                logger.warning(f"Invalid source '{source_filter}'; ignoring filter")
                print(f"Invalid source '{source_filter}'. Continuing with no filter.")
                source_filter = None
            elif not source_filter:
                source_filter = None
            answer = qa_system.query(query, source_filter)
            print(f"\nAnswer: {answer}")
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"An error occurred: {e}")
    finally:
        if qa_system.mysql_client is not None:
            qa_system.mysql_client.close()

if __name__ == "__main__":
    main()
