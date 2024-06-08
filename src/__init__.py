import logging

from cltl.brain import logger as brain_logger
from cltl.thoughts.thought_selection import logger as thoughts_logger
from user_model import logger as user_logger

brain_logger.setLevel(logging.ERROR)
thoughts_logger.setLevel(logging.INFO)
user_logger.setLevel(logging.INFO)
