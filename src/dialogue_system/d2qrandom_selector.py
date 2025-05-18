import random
from pathlib import Path

import torch

from dialogue_system.d2q_selector import D2Q
from dialogue_system.rl_utils.rl_parameters import DEVICE, LR, EPSILON_INFO, GAMMA, ACTION_THOUGHTS, N_ACTION_TYPES


class D2QRandom(D2Q):
    def __init__(self, brain, memory, encoder, reward="Total triples",
                 trained_model=None,
                 states_folder=Path("."),
                 learning_rate=LR, epsilon_info=EPSILON_INFO, gamma=GAMMA):
        """Initializes an instance of the Decomposed Deep Q-Network (D2Q) reinforcement learning algorithm that learns
        abstract actions and chooses specific actions randomly


        params

        returns:
        """
        super().__init__(brain, memory, encoder, reward, trained_model, states_folder,
                         learning_rate, epsilon_info, gamma)

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
