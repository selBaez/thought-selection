from cltl.commons.casefolding import (casefold_capsule)
from cltl.reply_generation.rl_replier import RLReplier
from cltl.reply_generation.utils.thought_utils import thoughts_from_brain

from src.dialogue_system.utils.thoughts_utils import structure_correct_thought


class RLCapsuleReplier(RLReplier):
    def __init__(self, brain, savefile=None, reward=None):
        """Creates a reinforcement learning-based replier to respond to questions
        and statements by the user. Statements are replied to by phrasing a
        thought; Selection of the thoughts are learnt by the UCB algorithm.

        params
        object brain: the brain of Leolani
        str savefile: file with stored utility values in JSON format

        returns: None
        """
        super(RLCapsuleReplier, self).__init__(brain, savefile, reward)

        self._statistics_history = []

    @property
    def statistics_history(self):
        return self._state_history

    def reply_to_statement(self, brain_response, persist=False, thought_options=None):
        """Selects a Thought from the brain response to verbalize, and produces a template capsule for the user .

        params
        dict brain_response: brain response from brain.update() converted to JSON

        returns:
        str reply: a string representing a verbalized thought
        dicr capsule_user: template for user to respond back
        """
        # Quick check if there is anything to do here
        if not brain_response['statement']['triple']:
            return None

        # What types of thoughts will we phrase?
        if not thought_options:
            thought_options = ['_complement_conflict', '_negation_conflicts', '_statement_novelty', '_entity_novelty',
                               '_complement_gaps', '_subject_gaps', '_overlaps']
        self._log.debug(f'Thoughts options: {thought_options}')

        # Casefold
        utterance = casefold_capsule(brain_response['statement'], format='natural')
        thoughts = casefold_capsule(brain_response['thoughts'], format='natural')

        # Extract thoughts from brain response
        thoughts = thoughts_from_brain(utterance, thoughts, filter=thought_options)

        # Select thought
        self._last_thought = self._thought_selector.select(thoughts.keys())
        thought_type, thought_info = thoughts[self._last_thought]
        self._log.info(f"Chosen thought type: {thought_type}")

        # Preprocess thought_info and utterance (triples)
        thought_info = {"thought": thought_info}
        thought_info = casefold_capsule(thought_info, format="natural")
        thought_info = thought_info["thought"]

        # Generate reply as capsule
        reply, capsule_user = structure_correct_thought(brain_response['statement'], thought_type, thought_info)

        # Calculate brain state
        profile = self._state_evaluator.calculate_brain_statistics(brain_response)
        self.statistics_history.append(profile)

        return reply, capsule_user
