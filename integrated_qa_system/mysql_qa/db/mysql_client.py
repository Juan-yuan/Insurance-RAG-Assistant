import pymysql
import pandas as pd
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(module_dir)

sys.path.insert(0, project_root)

from base import config, logger

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
        # 1. Define the SQL statement to create a table for subject knowledge Q&A data.
        create_table_query = '''
            create table if not exists jpkb (
                id int primary key auto_increment,    # Primary key (auto-increment)
                subject_name varchar(20),             # Subject name
                question varchar(1000),               # Question content
                answer varchar(1000)                  # Answer content      
            )
        '''

        # 2. Execute the SQL statement to create the table.
        try:
            # 3. Perform the table-creation action.
            self.cursor.execute(create_table_query)
            # 4. Commit the transaction -> MySQL requires a manual commit by default so the table creation takes effect.
            self.connection.commit()
            # 5. Write a log.
            self.logger.info("Table created successfully")
        except pymysql.MySQLError as e:
            # 6. Write an error log.
            self.logger.error(f"Table creation failed: {e}")
            raise

    def insert_data(self, csv_path):
        try:
            data = pd.read_csv(csv_path)
            for _, row in data.iterrows():
                # 2.1 Define the insert SQL statement, using %s placeholders to avoid SQL injection.
                insert_query = "insert into jpkb(subject_name, question, answer) values(%s, %s, %s)"
                # 2.2 Execute the SQL statement -> pass CSV row data (subject name, question, answer) into the placeholders.
                self.cursor.execute(insert_query, (row['学科名称'], row['问题'], row['答案']))

            # 3. Commit the transaction: ensure all insert operations take effect as a batch.
            self.connection.commit()
            # 4. Write an info log -> confirm the data was inserted successfully.
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
            # 1. Execute the SQL query.
            self.cursor.execute("select answer from jpkb where question=%s", (question,))       # Placeholder style, to avoid SQL injection.
            # 2. Fetch the query result.
            result = self.cursor.fetchone()         # fetchone(): fetch a single row.
            # 3. Output the query result.
            # print(f'result: {result}')
            # 4. Handle the result: if there is a match, return the answer (first element of the tuple); otherwise return None.
            return result[0] if result else None
        except pymysql.MySQLError as e:
            # 5. Write the error message.
            self.logger.error(f'Failed to get answer: {e}')
            # 6. Return None to avoid raising an exception.
            return None

    def close(self):
        # Close the database connection
        try:
            # Close the connection
            self.connection.close()
            # Log a successful close
            self.logger.info("MySQL connection closed")
        except pymysql.MySQLError as e:
            # Log a failed close
            self.logger.error(f"Failed to close connection: {e}")

if __name__ == '__main__':
    # 1. Instantiate MySQLClient -> create a database connection.
    mysql_client = MySQLClient()

    # 2. The following is test code for various features.
    mysql_client.create_table()     # Create the data table.

    # 3. Batch-insert data -> this function must not be run repeatedly, or it will append extra rows.
    mysql_client.insert_data(csv_path='../data/JP_questions.csv')

    # 4. Fetch all questions -> return a list of questions.
    questions = mysql_client.fetch_questions()
    # print(questions)

    # 5. Fetch the answer for a given question.
    answer = mysql_client.fetch_answer('How to calculate function execution time using a context manager?')
    print(answer)

    # 6. Close the database connection.
    mysql_client.close()
