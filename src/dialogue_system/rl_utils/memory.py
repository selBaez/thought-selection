import random
from collections import deque

from dialogue_system.rl_utils.rl_parameters import REPLAY_POOL_SIZE, BATCH_SIZE, Transition


class ReplayMemory(object):

    def __init__(self, capacity=REPLAY_POOL_SIZE, batch_size=BATCH_SIZE):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self):
        if len(self.memory) < self.batch_size:
            return None

        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)
