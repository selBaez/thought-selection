import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdflib import ConjunctiveGraph
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph

from cltl.thoughts.api import ThoughtSelector
from cltl.thoughts.thought_selection.utils.thought_utils import decompose_thoughts
from src.dialogue_system.metrics.graph_measures import get_avg_degree, get_sparseness, get_shortest_path
from src.dialogue_system.metrics.ontology_measures import get_avg_population
from src.dialogue_system.utils.encode_state import HarryPotterRDF, EncoderAttention
from src.dialogue_system.utils.helpers import download_from_triplestore
from src.dialogue_system.utils.plotting import separate_thought_elements, plot_action_counts, plot_cumulative_reward, plot_metrics_over_time
from src.dialogue_system.utils.rl_parameters import DEVICE, STATE_HIDDEN_SIZE, STATE_EMBEDDING_SIZE, REPLAY_POOL_SIZE, \
    BATCH_SIZE, DQN_HIDDEN_SIZE, LR, EPSILON_INFO, GAMMA, TAU, ACTION_THOUGHTS, N_ACTIONS_THOUGHTS, N_ACTION_TYPES, \
    ACTION_TYPES_REVERSED, Transition


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
        self.policy_net = DQN(STATE_EMBEDDING_SIZE, N_ACTIONS_THOUGHTS, N_ACTION_TYPES).to(DEVICE)
        self.target_net = DQN(STATE_EMBEDDING_SIZE, N_ACTIONS_THOUGHTS, N_ACTION_TYPES).to(DEVICE)
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
        self._action_history = [None]
        self._selection_history = [None]

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
    def action_history(self):
        return self._action_history

    @property
    def selection_history(self):
        return self._selection_history

    @property
    def state_evaluator(self):
        return self._state_evaluator

    @property
    def state_encoder(self):
        return self._state_encoder

    # Utils

    def load(self, filename):
        """Reads trained model from file.

        params
        Path filename: filename of file with utilities.

        returns: None
        """
        model_dict = self.policy_net.state_dict()
        modelCheckpoint = torch.load(filename)

        new_dict = {k: v for k, v in modelCheckpoint.items() if k in model_dict.keys()}
        model_dict.update(new_dict)

        self.policy_net.load_state_dict(model_dict)

        self._log.info(f"Loaded model from {filename.name}")

    def save(self, filename):
        """Writes the trained model to a file.

        params
        Path filename: filename of the output file.

        returns: None
        """
        torch.save(self.policy_net.state_dict(), filename)
        self._log.info(f"Saved model to {filename.name}")

    def _preprocess(self, brain_response, thought_options=None):
        # Manage types of capsules
        capsule_type = 'statement' if 'statement' in brain_response.keys() else 'mention'
        capsule_focus = 'triple' if 'statement' in brain_response.keys() else 'entity'

        # Quick check if there is anything to do here
        if not brain_response[capsule_type][capsule_focus]:
            return None

        # What types of thoughts will we phrase?
        self._log.debug(f'Thoughts options: {thought_options}')

        # Extract thoughts from brain response
        thoughts = decompose_thoughts(brain_response[capsule_type], brain_response['thoughts'], filter=thought_options)

        return thoughts

    def _postprocess(self, all_thoughts, selected_thought):
        # Keep track of selections
        self._last_thought = f"{selected_thought['thought_type']}:" \
                             f"{'-'.join(sorted(selected_thought['entity_types'].keys()))}"
        thought_type, thought_info = selected_thought["thought_type"], selected_thought
        self._log.info(f"Chosen thought type: {thought_type}")

        return thought_type, thought_info

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

        self._log.debug(f"Brain state added from file {state_file.name}, with metric value: {brain_state_metric}")

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
                full_tensor = self.policy_net(self._state_history["embeddings"][-1])
                # t.max(1) will pick action with the larger value
                action_tensor = full_tensor[:, :N_ACTIONS_THOUGHTS].max(1).indices.view(1, 1)
                action_name = ACTION_THOUGHTS[int(action_tensor[0])]

                subaction_tensor = full_tensor[:, N_ACTIONS_THOUGHTS:]
                self._log.debug(f"Select action with policy network")

        else:
            # Random action for exploration, equal distribution for any subaction
            [sampled_action] = random.sample(ACTION_THOUGHTS.keys(), 1)
            action_tensor = torch.tensor([[sampled_action]], device=DEVICE, dtype=torch.long)
            action_name = ACTION_THOUGHTS[sampled_action]

            subaction_tensor = torch.full((1, N_ACTION_TYPES), 1 / N_ACTION_TYPES, device=DEVICE, dtype=torch.float)
            self._log.debug(f"Select random action")

        return action_name, action_tensor, subaction_tensor

    def _target_net_update(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)
        self._log.info(f"Updating target net")

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
            next_state_batch = torch.cat(batch.next_state)
            reward_batch = torch.cat(batch.reward)

            # Compute action values based on the policy net: Q(s_t, a)
            # The model computes Q(s_t), then we select the columns of actions taken.
            # These are always valid action values, as it ignores subactions
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Compute action values for all next states based on the "older" target_net: V(s_{t+1})
            # The model computes the values of actions, then we select based on the best reward
            # Here we do need to set apart the subactions as they all come in the same vector from the model
            with torch.no_grad():
                next_state_values = self.target_net(next_state_batch)
                next_state_values = next_state_values[:, :N_ACTIONS_THOUGHTS].max(1).values

            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            expected_state_action_values = expected_state_action_values.unsqueeze(1)

            # Compute Huber loss, only on abstract actions
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values)
            self._log.debug(f"Computing loss between policy and target net predicted action values")

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self._log.info(f"Optimizing the policy net")

            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()

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
            self._log.debug(f"Calculate reward")
            reward = self.state_evaluator.compare_brain_states(self._state_history["metrics"][-1],
                                                               self._state_history["metrics"][-2])

            # Store the transition in memory
            self._log.debug("Pushing state transition to Memory Replay")
            self.memory.push(self._state_history["embeddings"][-2], self._action_history[-1],
                             self._state_history["embeddings"][-1],
                             torch.tensor([reward], device=DEVICE))

            # Perform one step of the optimization (on the policy network) and update target network
            self._log.debug("Updating networks")
            self.update_utility()
            self._target_net_update()

        self.reward_history.append(reward)
        self.selection_history.append(self._last_thought)
        self._log.info(f"{reward} reward due to {self._last_thought}")

        return reward

    def score_thoughts(self, processed_actions, subaction_tensor):
        action_scores = []
        for action in processed_actions:
            # Compute score for each element of the action
            score = []
            for typ, count in action["entity_types"].items():
                # Find index for the entity type
                subaction_idx = ACTION_TYPES_REVERSED.get(typ, -1)

                if subaction_idx >= 0:
                    # add to score
                    for i in range(count):
                        score.append(subaction_tensor[0, subaction_idx].item())
                else:
                    self._log.info(f"Entity type not in subaction vocabulary: {typ}")

            # Convert element-scores into action score
            action_scores.append((action, np.mean(score)))

        return action_scores

    def select(self, actions):
        """Selects an action from the set of available actions that maximizes
        the average observed reward, taking into account uncertainty.

        params
        list actions: List of actions from which to select

        returns: action
        """
        # Select thought type and make sure that type is available
        processed_actions = {}
        while len(processed_actions) == 0:
            # Select action
            action_name, action_tensor, subaction_tensor = self._select_by_network()

            # Safe processing, filter by selected action (thought type)
            processed_actions = self._preprocess(actions, thought_options=[action_name])

        # Score actions according to subactions (entity types)
        action_scores = self.score_thoughts(processed_actions, subaction_tensor)

        # Greedy selection
        selected_action, action_score = max(action_scores, key=lambda x: x[1])
        self._action_history.append(action_tensor)
        self._log.debug(f"Selected action {selected_action} with score {action_score}")

        # Safe processing
        thought_type, thought_info = self._postprocess(processed_actions, selected_action)

        return {thought_type: thought_info}

    def plot(self, episode_data, plots_folder):
        episode_data = pd.DataFrame.from_dict(episode_data)

        # Histogram: count per thought type and per entity type
        entity_types_counter, thought_types_counter = separate_thought_elements(episode_data["selections"])
        plot_action_counts(entity_types_counter, thought_types_counter, plots_folder)

        # Point plot: Cumulative reward
        plot_cumulative_reward(episode_data['rewards'], plots_folder)

        # State fluctuation
        plot_metrics_over_time(episode_data, plots_folder)


class BrainEvaluator(object):
    def __init__(self, brain, main_graph_metric):
        """ Create an object to evaluate the state of the brain according to different graph metrics.
        The graph can be evaluated by a single given metric, or a full set of pre established metrics
        """
        self._brain = brain
        self.metric = main_graph_metric

    def brain_as_graph(self):
        # Take brain from previous episodes
        graph = ConjunctiveGraph()
        graph.parse(data=self._brain._connection.export_repository(), format='trig')

        return graph

    def brain_as_netx(self):
        # Take brain from previous episodes
        netx = rdflib_to_networkx_multidigraph(self.brain_as_graph())

        return netx

    def evaluate_brain_state(self):
        brain_state = None

        ##### Group A #####
        if self.metric == 'Average degree':
            brain_state = get_avg_degree(self.brain_as_netx())
        elif self.metric == 'Sparseness':
            brain_state = get_sparseness(self.brain_as_netx())
        elif self.metric == 'Shortest path':
            brain_state = get_shortest_path(self.brain_as_netx())

        ##### Group B #####
        elif self.metric == 'Total triples':
            brain_state = self._brain.count_triples()
        elif self.metric == 'Average population':
            brain_state = get_avg_population(self.brain_as_graph())

        ##### Group C #####
        elif self.metric == 'Ratio claims to triples':
            brain_state = self._brain.count_statements() / self._brain.count_triples()
        elif self.metric == 'Ratio perspectives to claims':
            if self._brain.count_statements() != 0:
                brain_state = self._brain.count_perspectives() / self._brain.count_statements()
            else:
                brain_state = self._brain.count_perspectives() / 0.0000001
        elif self.metric == 'Ratio conflicts to claims':
            if self._brain.count_statements() != 0:
                brain_state = len(self._brain.get_all_negation_conflicts()) / self._brain.count_statements()
            else:
                brain_state = len(self._brain.get_all_negation_conflicts()) / 0.0000001

        return brain_state

    @staticmethod
    def compare_brain_states(current_state, prev_state):
        # TODO standardize according to metric
        if current_state is None or prev_state is None or prev_state == 0:
            reward = 0
        else:
            reward = current_state / prev_state

        return reward

    def calculate_brain_statistics(self, brain_response):
        # Grab the thoughts
        thoughts = brain_response['thoughts']

        # Gather basic stats
        stats = {
            'turn': brain_response['statement']['turn'],

            'cardinality conflicts': len(thoughts['_complement_conflict']) if thoughts['_complement_conflict'] else 0,
            'negation conflicts': len(thoughts['_negation_conflicts']) if thoughts['_negation_conflicts'] else 0,
            'subject gaps': len(thoughts['_subject_gaps']) if thoughts['_subject_gaps'] else 0,
            'object gaps': len(thoughts['_complement_gaps']) if thoughts['_complement_gaps'] else 0,
            'statement novelty': len(thoughts['_statement_novelty']) if thoughts['_statement_novelty'] else 0,
            'subject novelty': int(thoughts['_entity_novelty']['_subject']['value']),
            'object novelty': int(thoughts['_entity_novelty']['_complement']['value']),
            'overlaps subject-predicate': len(thoughts['_overlaps']['_subject'])
            if thoughts['_overlaps']['_subject'] else 0,
            'overlaps predicate-object': len(thoughts['_overlaps']['_complement'])
            if thoughts['_overlaps']['_complement'] else 0,
            'trust': thoughts['_trust'],

            'Total triples': self._brain.count_triples(),
            # 'Total classes': len(self._brain.get_classes()),
            # 'Total predicates': len(self._brain.get_predicates()),
            'Total claims': self._brain.count_statements(),
            'Total perspectives': self._brain.count_perspectives(),
            'Total conflicts': len(self._brain.get_all_negation_conflicts()),
            'Total sources': self._brain.count_friends(),
        }

        # Compute composite stats
        stats['Ratio claims to triples'] = stats['Total claims'] / stats['Total triples']
        stats['Ratio perspectives to triples'] = stats['Total perspectives'] / stats['Total triples']
        stats['Ratio conflicts to triples'] = stats['Total conflicts'] / stats['Total triples']
        stats['Ratio perspectives to claims'] = stats['Total perspectives'] / stats['Total claims']
        stats['Ratio conflicts to claims'] = stats['Total conflicts'] / stats['Total claims']

        return stats


class DQN(nn.Module):

    def __init__(self, n_observations, n_abs_actions, n_spe_actions, hidden_size=DQN_HIDDEN_SIZE):
        super(DQN, self).__init__()
        # Shared learning
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)

        # Abstract action learning (thought types)
        self.layer3_abs = nn.Linear(hidden_size, n_abs_actions)
        self.softmax_abs = nn.Softmax(dim=1)

        # Specific action learning (entity types in thoughts)
        self.layer3_spe = nn.Linear(hidden_size, n_spe_actions)
        self.softmax_spe = nn.Softmax(dim=1)

        # self.layer3 = nn.Linear(hidden_size, n_abs_actions)

    def forward(self, x):
        # Called with either one element to determine next action, or a batch during optimization.
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        x_abs = self.layer3_abs(x)
        x_abs = self.softmax_abs(x_abs)

        x_spe = self.layer3_spe(x)
        x_spe = self.softmax_abs(x_spe)

        y = torch.cat((x_abs, x_spe), 1)

        return y

        # return self.layer3(x)


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
        with torch.no_grad():  # TODO change this if we do train the encoder
            # RGAT - Conv
            data = self.dataset.process_one_graph(trig_file)

            # Check if the graph is empty,so we return a zero tensor or the right dimensions
            if len(data.edge_type) > 0:
                encoded_state = self.model_attention(data.node_features.float(), data.edge_index, data.edge_type)
            else:
                encoded_state = torch.tensor(np.zeros([1, self.embedding_size]), dtype=torch.float)

        return encoded_state
