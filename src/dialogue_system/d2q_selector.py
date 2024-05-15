import math
import random
from collections import namedtuple, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cltl.commons.casefolding import (casefold_capsule)
from cltl.thoughts.api import ThoughtSelector
from cltl.thoughts.thought_selection.rl_selector import BrainEvaluator
from cltl.thoughts.thought_selection.utils.thought_utils import thoughts_from_brain
from src.dialogue_system.utils.encode_state import HarryPotterRDF, EncoderAttention
from src.dialogue_system.utils.helpers import download_from_triplestore

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################## STATE REPRESENTATION PARAMETERS ##################
STATE_HIDDEN_SIZE = 64  # original features per node is 87
STATE_EMBEDDING_SIZE = 16  # also n_observations

################## MEMORY PARAMETERS ##################
REPLAY_POOL_SIZE = 10000  # 5000 for DQN
BATCH_SIZE = 5  # 16 for D2Q, 128 tutorial

################## RL PARAMETERS ##################
DQN_HIDDEN_SIZE = 128  # 80 for DQN
LR = 1e-4  # 1e-4 for D2Q
EPSILON_INFO = {"start": 0.9, "end": 0.05, "decay": 1000}
GAMMA = 0.99
TAU = 0.005
ACTION_THOUGHTS = {0: '_complement_conflict', 1: '_negation_conflicts',
                   2: '_statement_novelty', 3: '_entity_novelty',
                   4: '_subject_gaps', 5: '_complement_gaps',
                   6: '_overlaps', 7: '_trust'}
N_ACTIONS_THOUGHTS = len(ACTION_THOUGHTS)

################## DATASET SPECIFIC PARAMETERS ##################
ACTION_TYPES = {0: 'character', 1: 'attribute'}
N_ACTION_TYPES = len(ACTION_TYPES)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class D2Q(ThoughtSelector):
    def __init__(self, brain, reward="Total triples", savefile=None, states_folder=Path("."),
                 dataset=HarryPotterRDF('.'), predict_only=True,
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
        super().__init__()

        # generic D2Q parameters
        self.epsilon_info = epsilon_info
        self.gamma = gamma
        self.steps_done = 0

        # D2Q infrastructure
        self.policy_net = DQN(STATE_EMBEDDING_SIZE, N_ACTIONS_THOUGHTS).to(DEVICE)
        self.target_net = DQN(STATE_EMBEDDING_SIZE, N_ACTIONS_THOUGHTS).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

        # State in different forms (brain, file)
        self._state = brain
        self._states_folder = states_folder.resolve()
        self._states_folder.mkdir(parents=True, exist_ok=True)

        # Create a state encoder
        self._state_encoder = StateEncoder(dataset)
        self._log.debug(f"Brain encoder ready")

        # Include rewards according to the state of the brain
        self._state_evaluator = BrainEvaluator(brain, reward)
        self._log.debug(f"Brain state evaluator ready")
        self._reward = reward
        self._log.info(f"Reward: {self._reward}")

        # infrastructure to keep track of selections.
        self.memory = ReplayMemory()
        self._state_history = {"trig_files": [], "metrics": [], "embeddings": []}
        self._update_states()
        self._reward_history = [0]

        # Load learned policy
        # self.load(savefile)
        self._log.debug(f"D2Q RL Selector ready")

    @property
    def state_history(self):
        return self._state_history

    @property
    def reward_history(self):
        return self._reward_history

    @property
    def state_evaluator(self):
        return self._state_evaluator

    # Utils

    def load(self, filename):
        """Reads trained model from file.

        params
        str filename: filename of file with utilities.

        returns: None
        """
        model_dict = self.policy_net.state_dict()
        modelCheckpoint = torch.load(filename)

        new_dict = {k: v for k, v in modelCheckpoint.items() if k in model_dict.keys()}
        model_dict.update(new_dict)

        self.policy_net.load_state_dict(model_dict)

    def save(self, filename):
        """Writes the trained model to a file.

        params
        str filename: filename of the output file.

        returns: None
        """
        torch.save(self.policy_net.state_dict(), filename)

    def _update_states(self):
        """
        Calculate new brain state (by trig file, metric and embedding) and add it to the history
        """
        # Calculate new state representations
        state_file = download_from_triplestore(self._state, self._states_folder)
        brain_state_metric = self._state_evaluator.evaluate_brain_state()
        encoded_state = self._state_encoder.encode(state_file)

        # add to history
        self._state_history["trig_files"].append(state_file)
        self._state_history["metrics"].append(self._state_evaluator.evaluate_brain_state())
        self._state_history["embeddings"].append(encoded_state)

        self._log.info(f"Brain state added from file {state_file.name}, with metric value: {brain_state_metric}")

    # Learning
    def _select_by_network(self):
        # compute probability of choosing random action
        sample = random.random()
        eps_threshold = self.epsilon_info["end"] + (self.epsilon_info["start"] - self.epsilon_info["end"]) * \
                        math.exp(-1. * self.steps_done / self.epsilon_info["decay"])
        self.steps_done += 1

        if sample > eps_threshold:
            # Use model to choose action
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of max element, so we pick action with the larger expected reward
                action_tensor = self.policy_net(self._state_history["embeddings"][-1]).max(1).indices.view(1, 1)
                action_name = ACTION_THOUGHTS[int(action_tensor[0])]
        else:
            # Random action for exploration
            [sampled_action] = random.sample(ACTION_THOUGHTS.keys(), 1)
            action_tensor = torch.tensor([[sampled_action]], device=DEVICE, dtype=torch.long)
            action_name = ACTION_THOUGHTS[sampled_action]

        # Random subaction for exploration
        [sampled_subaction] = random.sample(ACTION_TYPES.keys(), 1)
        subaction_tensor = torch.tensor([[sampled_subaction]], device=DEVICE, dtype=torch.long)
        subaction_name = ACTION_TYPES[sampled_subaction]

        return action_name, action_tensor, subaction_name, subaction_tensor

    def select(self, actions):
        """Selects an action from the set of available actions that maximizes
        the average observed reward, taking into account uncertainty.

        params
        list actions: List of actions from which to select

        returns: action
        """
        # TODO add entity types logic

        # Select thought type and make sure that type is available
        processed_actions = {}
        while len(processed_actions) == 0:
            # Select action
            action_name, action_tensor, subaction_name, subaction_tensor = self._select_by_network()

            # Safe processing, selecting the thought type already
            processed_actions = self._preprocess(actions, thought_options=[action_name])

        action_scores = []
        for action in processed_actions:
            # Compute score for each element of the action
            score = []
            for elem in action.split():
                # if entity type in selected entity type, add it
                pass

            # Convert element-scores into action score
            action_scores.append((action, np.mean(score)))

        # Greedy selection
        selected_action, _ = max(action_scores, key=lambda x: x[1])

        # Safe processing
        thought_type, thought_info = self._postprocess(processed_actions, selected_action)

        return {thought_type: thought_info}

    def _preprocess(self, brain_response, thought_options=None):
        # Manage types of capsules
        capsule_type = 'statement' if 'statement' in brain_response.keys() else 'mention'
        capsule_focus = 'triple' if 'statement' in brain_response.keys() else 'entity'

        # Quick check if there is anything to do here
        if not brain_response[capsule_type][capsule_focus]:
            return None

        # What types of thoughts will we phrase?
        if not thought_options:
            thought_options = ['_complement_conflict', '_negation_conflicts',
                               '_statement_novelty', '_entity_novelty',
                               '_subject_gaps', '_complement_gaps',
                               '_overlaps', '_trust'] if 'statement' in brain_response.keys() \
                else ['_entity_novelty', '_complement_gaps']
        self._log.debug(f'Thoughts options: {thought_options}')

        # # Casefold TODO creates None types for empty lists
        # thoughts = casefold_capsule(brain_response['thoughts'], format='natural')
        # utterance = casefold_capsule(brain_response[capsule_type], format='natural')

        # Extract thoughts from brain response
        # thoughts = decompose_thoughts(utterance, thoughts, filter=thought_options)
        thoughts = thoughts_from_brain(brain_response[capsule_type], brain_response['thoughts'], filter=thought_options)

        return thoughts

    def update_utility(self):
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
        transitions = self.memory.sample()
        if transitions:
            # Transpose the batch This converts batch-array of Transitions to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
            # These are the actions which would've been taken for each batch state according to policy_net
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions are computed based on the "older" target_net; selecting their best reward
            next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
            with torch.no_grad():
                next_state_values = self.target_net(state_batch).max(1).values

            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()

    def _target_net_update(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def reward_thought(self):
        """Rewards the last thought phrased by the replier by calculating the difference between brain states
        according to a specific graph metric (i.e. a reward).

        returns: None
        """
        # Calculate state representations
        self._update_states()

        # Reward last thought with R = S_brain(t) - S_brain(t-1)
        reward = 0
        if self._last_thought and len(self._state_history["metrics"]) > 1:
            self._log.info(f"Calculate reward")
            reward = self.state_evaluator.compare_brain_states(self._state_history["metrics"][-1],
                                                               self._state_history["metrics"][-2])

            # Store the transition in memory
            self._log.info("Pushing state transition to Memory Replay")
            self.memory.push(self._state_history["embeddings"][-2], self._last_thought,
                             self._state_history["embeddings"][-1], reward)

            # Perform one step of the optimization (on the policy network) and update target network
            self._log.info("Updating networks")
            self.update_utility()
            self._target_net_update()

        self.reward_history.append(reward)
        self._log.info(f"{reward} reward due to {self._last_thought}")

        return reward


# class DQN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(DQN, self).__init__()
#
#         self.adv = nn.Sequential(OrderedDict([('w1', nn.Linear(input_size, hidden_size)),
#                                               ('relu', nn.Tanh()),
#                                               ('w2', nn.Linear(hidden_size, output_size))]))
#         self.vf = nn.Sequential(OrderedDict([('w1', nn.Linear(input_size, hidden_size)),
#                                              ('relu', nn.Tanh()),
#                                              ('w2', nn.Linear(hidden_size, 4 + 2))]))
#
#         # self.w1 = torch.Tensor([1.0])
#         # self.w1.requires_grad
#         # self.w1 = 0.3
#
#     def forward(self, x):
#         v = self.vf(x)
#         adv = self.adv(x)
#
#         batch_comm_x = v[:, 0:4]  #### comm action
#         batch_inf_x = v[:, 4:5]  #### inform action
#         batch_req_x = v[:, 5:]  #### request action
#
#         self.w1 = 1.8
#
#         # movie
#         v_expand = torch.cat((batch_comm_x,
#                               batch_inf_x.repeat(1, 13),
#                               batch_req_x.repeat(1, 12)), 1)
#
#         adv = adv - adv.mean(-1).unsqueeze(1).expand(adv.size())
#
#         # return v_expand + adv
#         return self.w1 * v_expand + (2.0 - self.w1) * adv
#
#     def predict(self, x):
#         y = self.forward(x)
#         return torch.argmax(y, 1)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_size=DQN_HIDDEN_SIZE):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


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


class StateEncoder(object):
    def __init__(self, dataset, embedding_size=STATE_EMBEDDING_SIZE, hidden_size=STATE_HIDDEN_SIZE):
        self.dataset = dataset
        self.embedding_size = STATE_EMBEDDING_SIZE
        self.model_attention = EncoderAttention(self.dataset.NUM_FEATURES, hidden_size, embedding_size,
                                                self.dataset.NUM_RELATIONS)

    def encode(self, trig_file):
        # RGAT - Conv
        data = self.dataset.process_one_graph(trig_file)

        # Check if the graph is empty,so we return a zero tensor or the right dimensions
        if len(data.edge_type) > 0:
            encoded_state = self.model_attention(data.node_features.float(), data.edge_index, data.edge_type)
        else:
            encoded_state = torch.tensor(np.zeros([1, self.embedding_size]), dtype=torch.float)

        return encoded_state
