import logging

from cltl.brain import logger as brain_logger
from cltl.reply_generation import logger as replier_logger
from cltl.thoughts.thought_selection import logger as thoughts_logger
from dialogue_system.rl_utils.memory import logger as memory_logger
from dialogue_system.rl_utils.hp_rdf_dataset import logger as dataset_logger
from user_model import logger as user_logger

brain_logger.setLevel(logging.ERROR)
thoughts_logger.setLevel(logging.ERROR)
replier_logger.setLevel(logging.ERROR)

dataset_logger.setLevel(logging.ERROR)
memory_logger.setLevel(logging.ERROR)
user_logger.setLevel(logging.ERROR)
