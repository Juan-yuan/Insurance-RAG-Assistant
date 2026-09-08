import os
import sys
import jieba

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(module_dir)
sys.path.insert(0, project_root)

from base import logger


def preprocess_text(text):
    logger.info("Starting text preprocessing")
    try:
        return jieba.lcut(text.lower())
    except AttributeError as e:
        logger.error(f"Text preprocessing failed: {e}")
        return []


if __name__ == '__main__':
    print(preprocess_text("Heima Programmer"))
