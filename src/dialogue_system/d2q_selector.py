import math
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cltl.thoughts.api import ThoughtSelector
from cltl.thoughts.thought_selection.utils.rl_utils import BrainEvaluator
from cltl.thoughts.thought_selection.utils.thought_utils import decompose_thoughts
from dialogue_system.rl_utils.rl_parameters import DEVICE, STATE_EMBEDDING_SIZE, DQN_HIDDEN_SIZE, LR, REPLAY_PER_TURN, \
    EPSILON_INFO, GAMMA, TAU, ACTION_THOUGHTS, N_ACTIONS_THOUGHTS, N_ACTION_TYPES, ACTION_TYPES_REVERSED, TaggedTransition
from dialogue_system.utils.helpers import download_from_triplestore, select_entity_type
from results_analysis.utils.plotting import separate_thought_elements, plot_action_counts, plot_cumulative_reward, \
    plot_metrics_over_time


class D2Q(ThoughtSelector):
    def __init__(self, brain, memory, encoder, reward="Total triples",
                 trained_model=None,
                 states_folder=Path("."),
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
        self.prediction_mode = (trained_model is not None)
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
        self._state_encoder = encoder
        self._log.debug(f"Brain encoder ready")

        # Include rewards according to the state of the brain
        self._state_evaluator = BrainEvaluator(brain, reward)
        self._log.debug(f"Brain state evaluator ready")
        self._reward = reward
        self._log.info(f"Reward: {self._reward}")

        # infrastructure to keep track of selections.
        self.memory = memory
        self._state_history = {"trig_files": [], "metrics": [], "embeddings": []}
        self._update_states()
        self._reward_history = [0]
        self._abstract_action_history = [None]
        self._specific_action_history = [None]
        self._selection_history = [None]

        # Load learned policy
        if trained_model:
            self.load(trained_model)
        self._log.debug(f"D2Q RL Selector ready")

    @property
    def state_history(self):
        return self._state_history

    @property
    def reward_history(self):
        return self._reward_history

    @property
    def abstract_action_history(self):
        return self._abstract_action_history

    @property
    def specific_action_history(self):
        return self._specific_action_history

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
        Path filename: filename of file with trained model

        returns: None
        """
        if type(filename) == str:
            filename = Path(filename)

        # Load trained model
        model_dict = self.policy_net.state_dict()
        model_checkpoint = torch.load(filename)
        new_dict = {k: v for k, v in model_checkpoint.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        self.policy_net.load_state_dict(model_dict)

        # Load memory replay
        memory_filename = filename.parent / "memory.pkl"
        with open(memory_filename, 'rb') as file:
            self.memory = pickle.load(file)

        self._log.info(f"Loaded model from {filename} and memory from {memory_filename}")

    def save(self, filename):
        """Writes the trained model to a file.

        params
        Path filename: filename of the output file for the model

        returns: None
        """
        # Save trained model
        torch.save(self.policy_net.state_dict(), filename)
        self._log.info(f"Saved model to {filename.name}")

        # Save memory replay
        memory_filename = filename.parent / "memory.pkl"
        with open(memory_filename, 'wb') as file:
            pickle.dump(self.memory, file)
        self._log.info(f"Saved memory to {memory_filename.name}")

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
        encoded_state.to(DEVICE)

        # add to history
        self._state_history["trig_files"].append(state_file)
        self._state_history["metrics"].append(self._state_evaluator.evaluate_brain_state())
        self._state_history["embeddings"].append(encoded_state)

        self._log.debug(f"Brain state added from file {state_file.name}, with metric value: {brain_state_metric}")

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
            state_batch = torch.cat(batch.state).to(DEVICE)
            abs_action_batch = torch.cat(batch.abs_action).to(DEVICE)
            spe_action_batch = torch.cat(batch.spe_action).to(DEVICE)
            next_state_batch = torch.cat(batch.next_state).to(DEVICE)
            reward_batch = torch.cat(batch.reward).to(DEVICE)

            # Compute action values based on the policy net: Q(s_t, a)
            state_action_values = self.policy_net(state_batch)

            # Select the columns of actions taken.
            state_abs_action_values = state_action_values[:, :N_ACTIONS_THOUGHTS]
            state_abs_action_values = state_abs_action_values.gather(1, abs_action_batch)
            state_spe_action_values = state_action_values[:, N_ACTIONS_THOUGHTS:]
            state_spe_action_values = state_spe_action_values.gather(1, spe_action_batch)

            # Compute action values for all next states based on the "older" target_net: V(s_{t+1})
            with torch.no_grad():
                next_state_action_values = self.target_net(next_state_batch)

            # Select based on the best reward
            next_state_abs_action_values = next_state_action_values[:, :N_ACTIONS_THOUGHTS].max(1).values
            next_state_spe_action_values = next_state_action_values[:, N_ACTIONS_THOUGHTS:].max(1).values

            # Compute the expected Q values
            expected_state_abs_action_values = (next_state_abs_action_values * self.gamma) + reward_batch
            expected_state_abs_action_values = expected_state_abs_action_values.unsqueeze(1)
            expected_state_spe_action_values = (next_state_spe_action_values * self.gamma) + reward_batch
            expected_state_spe_action_values = expected_state_spe_action_values.unsqueeze(1)

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss_abs = criterion(state_abs_action_values, expected_state_abs_action_values)
            loss_spe = criterion(state_spe_action_values, expected_state_spe_action_values)
            loss = loss_abs + loss_spe
            self._log.debug(f"Computing loss between policy and target net predicted action values")

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self._log.debug(f"Optimizing the policy net")

            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()

    def _update_target_network(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)
        self._log.debug(f"Updating target net")

    def _select_by_network(self):
        # compute probability of choosing random action
        sample = random.random()
        eps_threshold = self.epsilon_info["end"] + (self.epsilon_info["start"] - self.epsilon_info["end"]) * \
                        math.exp(-1. * self.steps_done / self.epsilon_info["decay"])
        self.steps_done += 1

        if sample > eps_threshold:
            # Use model to choose action
            with torch.no_grad():
                state_vector = self._state_history["embeddings"][-1].to(DEVICE)
                full_tensor = self.policy_net(state_vector)

                # Pick abstract action with the larger value
                action_tensor = full_tensor[:, :N_ACTIONS_THOUGHTS].max(1).indices.view(1, 1)
                action_name = ACTION_THOUGHTS[int(action_tensor[0])]

                # Retrieve specific action values
                subaction_tensor = full_tensor[:, N_ACTIONS_THOUGHTS:]
                self._log.debug(f"Select action with policy network")

        else:
            # Random abstract action for exploration
            [sampled_action] = random.sample(ACTION_THOUGHTS.keys(), 1)
            action_tensor = torch.tensor([[sampled_action]], device=DEVICE, dtype=torch.long)
            action_name = ACTION_THOUGHTS[sampled_action]

            # Equal distribution for any subaction
            subaction_tensor = torch.full((1, N_ACTION_TYPES), 1 / N_ACTION_TYPES, device=DEVICE, dtype=torch.float)
            self._log.debug(f"Select random action")

        return action_name, action_tensor, subaction_tensor

    def _score_thoughts(self, processed_actions, subaction_tensor):
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
                    self._log.error(f"Entity type not in subaction vocabulary: {typ}")

            # Convert element-scores into action score
            action_scores.append((action, np.mean(score)))

        return action_scores

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
            if reward < 0:
                self._log.debug(f"Negative reward! {reward}")

            # Store the transition in memory
            self._log.debug("Pushing state transition to Memory Replay")
            self.memory.push(self._state_history["embeddings"][-2],
                             self._abstract_action_history[-1], self._specific_action_history[-1],
                             self._state_history["embeddings"][-1],
                             torch.tensor([reward], device=DEVICE),
                             self._reward)

            if not self.prediction_mode:  # TODO: Instead of 1 update per turn, do K updates from the replay buffer (e.g., 5–10)
                # Perform one step of the optimization (on the policy network) and update target network
                self._log.debug("Updating networks")
                for update in range(REPLAY_PER_TURN):
                    self._update_policy_network()
                self._update_target_network()

        self.reward_history.append(reward)
        self.selection_history.append(self._last_thought)
        self._log.info(f"{reward} reward due to {self._last_thought}")

        return reward

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
        action_scores = self._score_thoughts(processed_actions, subaction_tensor)

        # Greedy selection
        selected_action, action_score = max(action_scores, key=lambda x: x[1])
        most_important_type = select_entity_type(selected_action)
        subaction_tensor = torch.tensor(ACTION_TYPES_REVERSED.get(most_important_type, 24)).view(1, 1)
        self._log.debug(f"Selected action: {selected_action['thought_type']} - "
                        f"{'/'.join(selected_action['entity_types'].keys())} "
                        f"with score {action_score}")

        # Keep track of action values
        self._abstract_action_history.append(action_tensor)
        self._specific_action_history.append(subaction_tensor)

        # Safe processing
        thought_type, thought_info = self._postprocess(processed_actions, selected_action)

        return {thought_type: thought_info}

    @staticmethod
    def plot(episode_data, plots_folder):
        episode_data = pd.DataFrame.from_dict(episode_data)

        # Histogram: count per thought type and per entity type
        entity_types_counter, thought_types_counter = separate_thought_elements(episode_data["selections"])
        plot_action_counts(entity_types_counter, thought_types_counter, plots_folder)

        # Point plot: Cumulative reward
        plot_cumulative_reward(episode_data['rewards'], plots_folder)

        # State fluctuation
        plot_metrics_over_time(episode_data, plots_folder)


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

    def forward(self, x):
        # Called with either one element to determine next action, or a batch during optimization.
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        x_abs = self.layer3_abs(x)
        x_abs = self.softmax_abs(x_abs)

        x_spe = self.layer3_spe(x)
        x_spe = self.softmax_spe(x_spe)

        y = torch.cat((x_abs, x_spe), 1)

        return y
