from cltl.commons.casefolding import (casefold_capsule)
from cltl.reply_generation.rl_replier import RLReplier
from cltl.reply_generation.utils.thought_utils import thoughts_from_brain
from rdflib import ConjunctiveGraph
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph

from src.chatbot.utils.thoughts_utils import structure_correct_thought


# from cltl.dialogue_evaluation.metrics.ontology_measures import get_avg_population
# from cltl.dialogue_evaluation.metrics.graph_measures import get_avg_degree, get_sparseness, get_shortest_path


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
        super(RLCapsuleReplier, self).__init__(brain, savefile)

        self.brain_stats = []
        self._reward_function = reward
        self._log.info(f"Reward: {self._reward_function}")

        self._brain_as_graph = None
        self._brain_as_netx = None

    def transform_brain(self):
        # Take brain from previous episodes
        self._brain_as_graph = ConjunctiveGraph()
        self._brain_as_graph.parse(data=self._brain._connection.export_repository(), format='trig')
        self._brain_as_netx = rdflib_to_networkx_multidigraph(self._brain_as_graph)

    def _score_brain(self, brain_response):
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
            'subject novelty': thoughts['_entity_novelty']['_subject'],
            'object novelty': thoughts['_entity_novelty']['_complement'],
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

        self.brain_stats.append(stats)

    def _evaluate_brain_state(self):
        brain_state = None

        self._log.info(f"Calculate reward")

        self.transform_brain()

        # if self._reward_function == 'Average degree':
        #     brain_state = get_avg_degree(self._brain_as_netx)
        # elif self._reward_function == 'Sparseness':
        #     brain_state = get_sparseness(self._brain_as_netx)
        # elif self._reward_function == 'Shortest path':
        #     brain_state = get_shortest_path(self._brain_as_netx)

        if self._reward_function == 'Total triples':
            brain_state = self._brain.count_triples()
        # elif self._reward_function == 'Average population':
        #     brain_state = get_avg_population(self._brain_as_graph)
        # elif self._reward_function == 'Total classes':
        #     brain_state = len(self._brain.get_classes())
        # elif self._reward_function == 'Total predicates':
        #     brain_state = len(self._brain.get_predicates())
        # elif self._reward_function == 'Total claims':
        #     brain_state = self._brain.count_statements()
        # elif self._reward_function == 'Total perspectives':
        #     brain_state = self._brain.count_perspectives()
        # elif self._reward_function == 'Total conflicts':
        #     brain_state = len(self._brain.get_all_negation_conflicts())
        # elif self._reward_function == 'Total sources':
        #     brain_state = self._brain.count_friends()

        elif self._reward_function == 'Ratio claims to triples':
            brain_state = self._brain.count_statements() / self._brain.count_triples()
        # elif self._reward_function == 'Ratio perspectives to triples':
        #     brain_state = self._brain.count_perspectives() / self._brain.count_triples()
        # elif self._reward_function == 'Ratio conflicts to triples':
        #     brain_state = len(self._brain.get_all_negation_conflicts()) / self._brain.count_triples()
        elif self._reward_function == 'Ratio perspectives to claims':
            brain_state = self._brain.count_perspectives() / self._brain.count_statements()
        elif self._reward_function == 'Ratio conflicts to claims':
            brain_state = len(self._brain.get_all_negation_conflicts()) / self._brain.count_statements()

        return brain_state

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
        self._score_brain(brain_response)

        return reply, capsule_user
