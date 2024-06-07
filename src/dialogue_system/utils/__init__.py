import logging

from cltl.brain import logger as brain_logger

brain_logger.setLevel(logging.ERROR)

from cltl.thoughts.thought_selection import logger as thoughts_logger

thoughts_logger.setLevel(logging.INFO)
