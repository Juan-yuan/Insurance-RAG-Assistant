import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    REDIS_HOST = 'localhost'
    REDIS_PORT = 8379
    REDIS_PASSWORD = '1234'
    REDIS_DB = 0

if __name__ == '__main__':
    print(f'REDIS_HOST: {Config.REDIS_HOST}')
    conf = Config()
    print(f'REDIS_PORT: {conf.REDIS_PORT}')
    print(f'REDIS_PORT: {conf.REDIS_PORT}')
