import os
import sys
import time
import uuid
import warnings
import pymysql
from openai import OpenAI

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from mysql_qa import MySQLClient, RedisClient, BM25Search
from rag_qa import VectorStore, RAGSystem
from insurance_rag.config import config
from insurance_rag.logger import logger


class IntegratedQASystem:
    def __init__(self):
        self.logger = logger
        self.config = config
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
        try:
            self.client = OpenAI(api_key=self.config.OPENAI_API_KEY, base_url=self.config.OPENAI_BASE_URL)
        except Exception as e:
            self.logger.error(f"Failed to initialize the OpenAI client: {e}")
            raise

        self.vector_store = VectorStore()
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)

        self.init_conversation_table()

    def init_conversation_table(self):
        """
        Function: create the Conversations table in MySQL if it does not exist, used to store chat history.
        :return:
        """
        try:
            self.mysql_client.cursor.execute("""
                create table if not exists conversations(
                    id INT AUTO_INCREMENT PRIMARY KEY,         
                    session_id VARCHAR(36) NOT NULL,            
                    question TEXT NOT NULL,                    
                    answer TEXT NOT NULL,                    
                    timestamp DATETIME NOT NULL,         
                    INDEX idx_session_id (session_id)
                )
            """)
            self.mysql_client.connection.commit()
            self.logger.info("Conversation history table initialized successfully")
        except pymysql.MySQLError as e:
            self.logger.error(f"Failed to initialize the conversation history table: {e}")
            raise

    def call_dashscope(self, prompt):
        """
        Function: call the DashScope API and stream the generated answer.
        :param prompt: prompt text sent to the LLM (str), including the user question and context.
        :return: generator that yields answer chunks (str).
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,  # Use the LLM from config.
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},  # System prompt.
                    {"role": "user", "content": prompt},  # User prompt.
                ],
                timeout=30,
                stream=True
            )
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield content
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return f"Error: LLM call failed - {e}"


    def _fetch_recent_history(self, session_id):
        """
        Function: fetch the last 5 turns of conversation history.
        :param session_id: unique session id (str) used to look up the conversation.
        :return: list of dicts, e.g. [{'question': question, 'answer': answer}].
        """
        try:
            self.mysql_client.cursor.execute("""
                 SELECT question, answer
                 FROM conversations
                 WHERE session_id = %s
                 ORDER BY timestamp DESC
                     LIMIT %s
                 """, (session_id, 5))
            history = [{"question": row[0], "answer": row[1]} for row in self.mysql_client.cursor.fetchall()]
            return history[::-1]
        except pymysql.MySQLError as e:
            self.logger.error(f"Failed to fetch conversation history: {e}")
            return []

    def get_session_history(self, session_id):
        """
        Function: load session history from MySQL.
        :param session_id: unique session id (str).
        :return: list of the last 5 turns, same shape as _fetch_recent_history().
        """
        return self._fetch_recent_history(session_id)


    def update_session_history(self, session_id: str, question: str, answer: str):
        """
        Function: write a new turn to MySQL and keep only the last 5 turns.
        :param session_id: unique session id (str).
        :param question: user question (str).
        :param answer: system answer (str).
        :return:
        """
        try:
            self.mysql_client.cursor.execute("""
                 INSERT INTO conversations (session_id, question, answer, timestamp)
                 VALUES (%s, %s, %s, NOW())
                 """, (session_id, question, answer))
            history = self._fetch_recent_history(session_id)
            self.mysql_client.cursor.execute("""
                 DELETE
                 FROM conversations
                 WHERE session_id = %s
                   AND id NOT IN (SELECT id
                      FROM (SELECT id
                            FROM conversations
                            WHERE session_id = %s
                            ORDER BY timestamp DESC
                                LIMIT %s) AS sub)
                 """, (session_id, session_id, 5))
            self.mysql_client.connection.commit()
            self.logger.info(f"Session {session_id} history updated successfully")
            return history
        except pymysql.MySQLError as e:
            self.logger.error(f"Failed to update session history: {e}")
            self.mysql_client.connection.rollback()
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while updating session history: {e}")
            self.mysql_client.connection.rollback()
            raise


    def clear_session_history(self, session_id: str) -> bool:
        """
        Function: delete all history for the given session.
        :param session_id: unique session id (str).
        :return: True if cleared successfully, False otherwise.
        """
        try:
            self.mysql_client.cursor.execute("""
                 DELETE
                 FROM conversations
                 WHERE session_id = %s
                 """, (session_id,)
            )
            self.mysql_client.connection.commit()
            self.logger.info(f"Session {session_id} history cleared")
            return True
        except pymysql.MySQLError as e:
            self.logger.error(f"Failed to clear session history: {e}")
            self.mysql_client.connection.rollback()
            return False

    def query(self, query, source_filter=None, session_id=None):
        """
        Function: query the integrated system with conversation history and streaming output.
        :param query:
        :param source_filter:
        :param session_id:
        :return:
        """
        start_time = time.time()
        self.logger.info(f"Handling query: '{query}' (session ID: {session_id})")

        history = self.get_session_history(session_id) if session_id else []
        answer, need_rag = self.bm25_search.search(query, threshold=0.85)

        if answer:
            self.logger.info(f"MySQL answer: {answer}")
            if session_id:
                self.update_session_history(session_id, query, answer)
            processing_time = time.time() - start_time
            self.logger.info(f"Query handling time {processing_time:.2f}s")
            yield answer, True
        elif need_rag:
            self.logger.info("No reliable MySQL answer; falling back to RAG")
            collected_answer = ""
            for token in self.rag_system.generate_answer(query, source_filter=source_filter, history=history):
                collected_answer += token
                yield token, False
            if session_id:
                self.update_session_history(session_id, query, collected_answer)
            processing_time = time.time() - start_time
            self.logger.info(f"Query handling time {processing_time:.2f}s")
            yield "", True
        else:
            self.logger.info("No answer found")
            processing_time = time.time() - start_time
            self.logger.info(f"Query handling time {processing_time:.2f}s")
            yield "No answer found", True


def main():
    qa_system = IntegratedQASystem()
    session_id = str(uuid.uuid4())
    print("\nWelcome to the integrated Q&A system.")
    print(f"Session ID: {session_id}")
    print(f"Supported subject categories: {qa_system.config.VALID_SOURCES}")
    print("Enter a query to ask a question, or type 'exit' to quit.")

    try:
        while True:
            query = input("\nEnter query: ").strip()
            if query.lower() == "exit":
                logger.info("Exiting the system")
                print("Goodbye. Thanks for using the system.")
                break
            source_filter = input(
                f"Enter a subject category ({'/'.join(qa_system.config.VALID_SOURCES)}) (press Enter to skip): "
            ).strip()
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:

                logger.warning(f"Invalid subject category '{source_filter}'; ignoring filter")

                source_filter = None

            print("\nAnswer: ", end="", flush=True)
            answer = ""
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                if token:
                    print(token, end="", flush=True)
                    answer += token
                if is_complete:
                    print()
                    break
            history = qa_system.get_session_history(session_id)
            print("\nRecent conversation history:")
            for idx, entry in enumerate(history, 1):
                print(f"{idx}. Q: {entry['question']}\n   A: {entry['answer']}")
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"An error occurred: {e}")
    finally:
        qa_system.mysql_client.close()

if __name__ == '__main__':
    main()
