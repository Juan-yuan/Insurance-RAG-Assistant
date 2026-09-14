from utils.logger import *
logger = setup_logger('AI')

def process_data(data):
    logger.debug(f'Start processing data: {data}')
    if not data:
        logger.error('Data is empty, unable to process')
        return None
    logger.info('Data processing completed')
    return data.upper()

def main():
    logger.info('Program started')
    result = process_data('Wishing everyone a happy Dragon Boat Festival!')
    if result:
        logger.info(f'Processing result: {result}')
    else:
        logger.warning('Processing failed')
    logger.info('Program ended')

if __name__ == '__main__':
    main()
