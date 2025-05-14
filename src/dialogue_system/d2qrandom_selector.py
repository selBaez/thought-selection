import random
from pathlib import Path

import torch

from dialogue_system.d2q_selector import D2Q
from dialogue_system.rl_utils.rl_parameters import DEVICE, LR, EPSILON_INFO, GAMMA, ACTION_THOUGHTS, N_ACTION_TYPES


class D2QRandom(D2Q):
    def __init__(self, dataset, brain, reward="Total triples", trained_model=None, states_folder=Path("."),
                 learning_rate=LR, epsilon_info=EPSILON_INFO, gamma=GAMMA):
        """Initializes an instance of the Decomposed Deep Q-Network (D2Q) reinforcement learning algorithm.
        States are saved in different forms:

        as triple store => brain (only the current state)
        as trig files in states_folder => in a list
        as a calculated metric over the graph => in a list
        as embeddings => in Replay memory


        params

        returns:
        """
        super().__init__(dataset, brain, reward, trained_model, states_folder, learning_rate, epsilon_info, gamma)

    # Learning

    def _select_by_network(self):
        self.steps_done += 1

        # Random abstract action for exploration
        [sampled_action] = random.sample(ACTION_THOUGHTS.keys(), 1)
        action_tensor = torch.tensor([[sampled_action]], device=DEVICE, dtype=torch.long)
        action_name = ACTION_THOUGHTS[sampled_action]
        # Equal distribution for any subaction
        subaction_tensor = torch.full((1, N_ACTION_TYPES), 1 / N_ACTION_TYPES, device=DEVICE, dtype=torch.float)
        self._log.debug(f"Select random action")

        return action_name, action_tensor, subaction_tensor
