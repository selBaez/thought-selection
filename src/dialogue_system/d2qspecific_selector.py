import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dialogue_system.d2q_selector import D2Q
from dialogue_system.rl_utils.rl_parameters import DEVICE, STATE_EMBEDDING_SIZE, DQN_HIDDEN_SIZE, LR, EPSILON_INFO, \
    GAMMA, ACTION_THOUGHTS, N_ACTION_TYPES, TaggedTransition


class D2QSpecific(D2Q):
    def __init__(self, brain, memory, encoder, reward="Total triples",
                 trained_model=None,
                 states_folder=Path("."),
                 learning_rate=LR, epsilon_info=EPSILON_INFO, gamma=GAMMA):
        """Initializes an instance of the Decomposed Deep Q-Network (D2Q) reinforcement learning algorithm that learns
        specific actions and chooses abstract actions randomly


        params

        returns:
        """
        super().__init__(brain, memory, encoder, reward, trained_model, states_folder,
                         learning_rate, epsilon_info, gamma)

        # D2Q infrastructure
        self.policy_net = DQNSpecific(STATE_EMBEDDING_SIZE, N_ACTION_TYPES).to(DEVICE)
        self.target_net = DQNSpecific(STATE_EMBEDDING_SIZE, N_ACTION_TYPES).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

    # Learning
    def _update_policy_network(self):
        """Updates the policy network by
        1) sampling a batch,
        2) computing the states Q-values using the policy network,
        3) computing the next states using the target network,
        4) computing the next states Q-values using the rewards and a GAMMA factor
        5) computing the loss,
        6) optimizing the policy network parameters according to this loss

        params
        str action:    selected action (with elements elem that are scored)
        float reward:  reward obtained after performing the action

        returns: None
        """
        transitions = self.memory.sample(reward_type=self._reward)
        if transitions:
            # Transpose the batch: convert batch-array of Transitions to Transition of batch-arrays
            batch = TaggedTransition(*zip(*transitions))
            state_batch = torch.cat([t.to(DEVICE) for t in batch.state])
            spe_action_batch = torch.cat([t.to(DEVICE) for t in batch.spe_action])
            next_state_batch = torch.cat([t.to(DEVICE) for t in batch.next_state])
            reward_batch = torch.cat([t.to(DEVICE) for t in batch.reward])

            # Compute action values based on the policy net: Q(s_t, a)
            state_action_values = self.policy_net(state_batch)

            # Select the columns of actions taken.
            state_spe_action_values = state_action_values.gather(1, spe_action_batch)

            # Compute action values for all next states based on the "older" target_net: V(s_{t+1})
            with torch.no_grad():
                next_state_action_values = self.target_net(next_state_batch)

            # Select based on the best reward
            next_state_spe_action_values = next_state_action_values.max(1).values

            # Compute the expected Q values
            expected_state_spe_action_values = (next_state_spe_action_values * self.gamma) + reward_batch
            expected_state_spe_action_values = expected_state_spe_action_values.unsqueeze(1)

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss_spe = criterion(state_spe_action_values, expected_state_spe_action_values)
            self._log.debug(f"Computing loss between policy and target net predicted action values")

            # Optimize the model
            self.optimizer.zero_grad()
            loss_spe.backward()
            self._log.debug(f"Optimizing the policy net")

            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()

    def _select_by_network(self):
        # compute probability of choosing random action
        sample = random.random()
        eps_threshold = self.epsilon_info["end"] + (self.epsilon_info["start"] - self.epsilon_info["end"]) * \
                        math.exp(-1. * self.steps_done / self.epsilon_info["decay"])
        self.steps_done += 1

        # Random abstract action
        [sampled_action] = random.sample(ACTION_THOUGHTS.keys(), 1)
        action_tensor = torch.tensor([[sampled_action]], device=DEVICE, dtype=torch.long)
        action_name = ACTION_THOUGHTS[sampled_action]
        self._log.debug(f"Select random abstract action")

        if sample > eps_threshold:
            # Use model to choose specific action
            with torch.no_grad():
                state_vector = self._state_history["embeddings"][-1].to(DEVICE)
                # Retrieve specific action values
                subaction_tensor = self.policy_net(state_vector)
                self._log.debug(f"Select specific action with policy network")

        else:
            # Equal distribution for any subaction
            subaction_tensor = torch.full((1, N_ACTION_TYPES), 1 / N_ACTION_TYPES, device=DEVICE, dtype=torch.float)
            self._log.debug(f"Select random specific action")

        return action_name, action_tensor, subaction_tensor


class DQNSpecific(nn.Module):

    def __init__(self, n_observations, n_spe_actions, hidden_size=DQN_HIDDEN_SIZE):
        super(DQNSpecific, self).__init__()
        # Shared learning
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)

        # Specific action learning (entity types in thoughts)
        self.layer3_spe = nn.Linear(hidden_size, n_spe_actions)
        self.softmax_spe = nn.Softmax(dim=1)

    def forward(self, x):
        # Called with either one element to determine next action, or a batch during optimization.
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        x_spe = self.layer3_spe(x)
        x_spe = self.softmax_spe(x_spe)

        return x_spe
