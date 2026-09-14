import logging
import os
module_path = os.path.abspath(__file__)
base_path = os.path.dirname(module_path)
current_path = os.path.dirname(base_path)
log_file = os.path.join(current_path, 'logs/app.log')

def setup_logger(name, log_file=log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger

if __name__ == '__main__':
    logger = setup_logger('ai')
