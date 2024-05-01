import numpy as np

from cltl.thoughts.api import ThoughtSelector
from cltl.thoughts.thought_selection.rl_selector import BrainEvaluator


class D2Q(ThoughtSelector):
    def __init__(self, brain, reward="Total triples", savefile=None, c=2, tmax=1e10):
        """Initializes an instance of the Upper Confidence Bound
        (UCB) reinforcement learning algorithm.

        params
        float c:          controls level of exploration
        float tmax:       number of timesteps in which uncertainty of choices
                          are taken into account (exploitation when t > tmax)

        float t:          timestep
        float decay:      decay rate of exploration constant c
        float Q:          stores the estimate of the expected reward for each action
        float N:          stores the number of updates performed on each action

        returns: UCB object
        """
        super().__init__()

        # generic D2Q parameters

        # Include rewards according to the state of the brain
        self._state_evaluator = BrainEvaluator(brain)
        self._log.debug(f"Brain state evaluator ready")
        self._reward = reward
        self._log.info(f"Reward: {self._reward}")

        # infrastructure to keep track of selections
        self._state_history = [self._state_evaluator.evaluate_brain_state(self._reward)]
        self._reward_history = [0]

        # Load learned policy
        self.load(savefile)
        self._log.debug(f"UCB RL Selector ready")

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
        """Reads utility values from file.

        params
        str filename: filename of file with utilities.

        returns: None
        """
        pass

    def save(self, filename):
        """Writes the value and uncertainty tables to a JSON file.

        params
        str filename: filename of the ouput file.

        returns: None
        """
        pass

    def select(self, actions):
        """Selects an action from the set of available actions that maximizes
        the average observed reward, taking into account uncertainty.

        params
        list actions: List of actions from which to select

        returns: action
        """
        # Safe processing
        actions = self._preprocess(actions)

        action_scores = []
        for action in actions:

            # Compute UCB score for each element of the action
            score = []
            for elem in action.split():
                pass

            # Convert element-scores into action score
            action_scores.append((action, np.mean(score)))

        # Greedy selection
        selected_action, _ = max(action_scores, key=lambda x: x[1])

        # Safe processing
        thought_type, thought_info = self._postprocess(actions, selected_action)

        return {thought_type: thought_info}

    def update_utility(self, action, reward):
        """Updates the action-value table (Q) by incrementally updating the
        reward estimate of the action elements with the observed reward.

        params
        str action:    selected action (with elements elem that are scored)
        float reward:  reward obtained after performing the action

        returns: None
        """
        pass

    def reward_thought(self):
        """Rewards the last thought phrased by the replier by updating its
        utility estimate with the relative improvement of the brain as
        a result of the user response (i.e. a reward).

        returns: None
        """
        pass
