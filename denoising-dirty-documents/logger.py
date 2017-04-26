import logging
import sys

LOGGER_NAME = 'OUTPUT'

fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)

handlers = [logging.FileHandler(LOGGER_NAME + '.log'), logging.StreamHandler(sys.stdout)]
handlers[0].setFormatter(formatter)
handlers[1].setFormatter(formatter)

def getLogger(category = 'OUTPUT', level = logging.DEBUG):
  logger = logging.getLogger(category)
  # logger.addHandler(handlers[0])
  logger.addHandler(handlers[1])
  logger.setLevel(level)

  return logger

logger = getLogger()
