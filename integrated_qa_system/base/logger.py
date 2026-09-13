# Custom logging utility class.

# Imports
import logging
import os
from config import Config


# todo 1. Compute the final save path of the log file.
# 1. Get the absolute path of the current script file.
module_path = os.path.abspath(__file__)
# print(f'module_path: {module_path}')

# 2. Get the path of the directory containing the current script.
base_path = os.path.dirname(module_path)
# print(f'base_path: {base_path}')

# 3. Get the path of the parent directory of the current script directory.
current_path = os.path.dirname(base_path)
# print(f'current_path: {current_path}')

# 4. Join the full path of the log file.   parent directory + logs folder + app.log file.
# log_file = os.path.join(current_path, 'logs/app.log')
log_file = os.path.join(current_path, Config().LOG_FILE)
# print(f'log_file: {log_file}')


# todo 2. Encapsulate the log configuration function -> reusable dual-output logging.
def setup_logger(log_file=log_file):
    """
    Function: create and return a logger that outputs logs to both console and file, and avoid adding handlers repeatedly.
    :param log_file: save path of the log file.
    :return: logger
    """
    # 1. Ensure the log directory exists.
    # arg1: get the directory of the log file.  
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 2. Create a logger object -> manage log handlers uniformly and set the log level.
    logger = logging.getLogger('Insurance RAG Assistant')
    logger.setLevel(logging.INFO)

    # 7. Add handlers to the logger. Key point: check whether handlers already exist to avoid adding them repeatedly when this function is called multiple times.
    if not logger.handlers:

        # 3. Create a console handler -> output logs to the console.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 4. Create a file handler -> output logs to a file.
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 5. Create the log output format, i.e.: timestamp - logger name - log level - log message
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 6. Bind the log format to both handlers.
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # 7. Add the handlers to the logger.
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # 8. Return the logger.
    return logger


# todo 3. Initialize the logger -> executed automatically when the project starts; other modules can import this logger and use it directly.
logger = setup_logger()

# todo 4. Main entry -> test logging. Running this file writes records to the log file.
if __name__ == '__main__':
    logger.info('Logger initialized successfully.')
    logger.info(f'Log file path: {log_file}')
    for handler in logger.handlers:
        handler.flush()
