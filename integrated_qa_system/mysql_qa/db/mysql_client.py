import pymysql
import pandas as pd
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(module_dir)
sys.path.insert(0, project_root)
from insurance_rag import config, logger

class MySQLClient:

    def __init__(self):
        self.logger = logger
        try:
            self.connection = pymysql.connect(
                host = config.MYSQL_HOST,
                user = config.MYSQL_USER,
                password = config.MYSQL_PASSWORD,
                database = config.MYSQL_DATABASE,
            )
            self.cursor = self.connection.cursor()
            self.logger.info("MySQL connection succeeded")
        except pymysql.MySQLError as e:
            self.logger.error(f"MySQL connection failed: {e}")
            raise

    def create_table(self):
        create_table_query = '''
            create table if not exists jpkb (
                id int primary key auto_increment,
                subject_name varchar(20),
                question varchar(1000),
                answer varchar(1000)
            )
        '''
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            self.logger.info("Table created successfully")
        except pymysql.MySQLError as e:
            self.logger.error(f"Table creation failed: {e}")
            raise

    def insert_data(self, csv_path):
        try:
            data = pd.read_csv(csv_path)
            for _, row in data.iterrows():
                insert_query = "insert into jpkb(subject_name, question, answer) values(%s, %s, %s)"
                self.cursor.execute(insert_query, (row['学科名称'], row['问题'], row['答案']))
            self.connection.commit()
            self.logger.info('MySQL data inserted successfully')
        except Exception as e:
            self.logger.error(f'Data insertion failed: {e}')
            self.connection.rollback()
            raise

    def fetch_questions(self):
        try:
            self.cursor.execute("select question from jpkb")
            results = self.cursor.fetchall()
            self.logger.info('Data query succeeded')
            return results
        except pymysql.MySQLError as e:
            self.logger.error(f'Data query failed: {e}')
            return []

    def fetch_answer(self, question):
        try:
            self.cursor.execute("select answer from jpkb where question=%s", (question,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except pymysql.MySQLError as e:
            self.logger.error(f'Failed to get answer: {e}')
            return None

    def close(self):
        try:
            self.connection.close()
            self.logger.info("MySQL connection closed")
        except pymysql.MySQLError as e:
            self.logger.error(f"Failed to close connection: {e}")

if __name__ == '__main__':
    mysql_client = MySQLClient()
    mysql_client.create_table()
    mysql_client.insert_data(csv_path='../data/JP_questions.csv')
    questions = mysql_client.fetch_questions()
    answer = mysql_client.fetch_answer('How to calculate function execution time using a context manager?')
    print(answer)
    mysql_client.close()
