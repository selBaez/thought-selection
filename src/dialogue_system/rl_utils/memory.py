import pickle
import random
from collections import deque, defaultdict
from pathlib import Path

from dialogue_system import logger
from dialogue_system.rl_utils.rl_parameters import REPLAY_POOL_SIZE, BATCH_SIZE, TaggedTransition


class ReplayMemory(object):

    def __init__(self, capacity=REPLAY_POOL_SIZE, batch_size=BATCH_SIZE, prepopulate_path=None, add_reward_type=True):

        self._log = logger.getChild(self.__class__.__name__)
        self._log.info("Booted")

        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size

        if prepopulate_path:
            self.pre_populate(prepopulate_path, add_reward_type=add_reward_type)

        self._log.info(f"Memory size: {len(self)}")

    def push(self, *args):
        """Save a transition"""
        self.memory.append(TaggedTransition(*args))

    def sample(self, reward_type=None):
        if reward_type:
            filtered = [t for t in self.memory if getattr(t, 'reward_type', None) == reward_type]
        else:
            filtered = list(self.memory)

        if len(filtered) < self.batch_size:
            return None

        return random.sample(filtered, self.batch_size)

    def __len__(self):
        return len(self.memory)

    def pre_populate(self, experiments_path, add_reward_type=True):
        # get all pickle files in scenario folder
        pkl_files = list(experiments_path.rglob("*.pkl"))
        pkl_by_reward = defaultdict(list)

        # Sort them by reward
        for pkl_path in pkl_files:
            reward = pkl_path.relative_to(experiments_path).parts[1]  # reward is 3rd level
            pkl_by_reward[reward].append(pkl_path)

        # Read them into memory
        for reward_type, pkl_files in pkl_by_reward.items():
            for pkl_path in pkl_files:
                printable_path = Path(pkl_path).resolve().relative_to(experiments_path.resolve())
                try:
                    with open(pkl_path, 'rb') as file:
                        temp_memory = pickle.load(file)
                    for trans in temp_memory.memory:
                        if add_reward_type:
                            self.push(*trans, reward_type)
                        else:
                            self.push(*trans)
                    self._log.debug(f"Added {len(temp_memory)} transitions from {printable_path}")
                except:
                    self._log.debug(f"Error loading file: {printable_path}")
                    pass
