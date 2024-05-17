import copy
import random
from typing import Optional, Tuple

from cltl.commons.casefolding import casefold_capsule, casefold_text
from cltl.reply_generation.simplenlg_phraser import SimplenlgPhraser
from cltl.reply_generation.utils.phraser_utils import clean_overlaps
from src.dialogue_system.utils.global_variables import BASE_CAPSULE, ONTOLOGY_DETAILS

def label_in_say(label, say):
    if say:
        return label.lower() in say.lower()
    else:
        return False

class TriplePhraser(SimplenlgPhraser):

    def __init__(self):
        # type: () -> None
        """
        Generate response as structured data (thought to capsule)

        Parameters
        ----------
        """

        super(SimplenlgPhraser, self).__init__()

    def phrase_correct_thought(self, utterance, thought_type, thought_info, fallback=False) -> Tuple[
        Optional[str], Optional[dict]]:
        reply = None
        response_template = None

        if thought_type == "_complement_conflict":
            reply, response_template = self._phrase_cardinality_conflicts(thought_info, utterance)

        elif thought_type == "_negation_conflicts":
            reply, response_template = self._phrase_negation_conflicts(thought_info, utterance)

        elif thought_type == "_statement_novelty":
            reply, response_template = self._phrase_statement_novelty(thought_info, utterance)

        elif thought_type == "_entity_novelty":
            reply, response_template = self._phrase_type_novelty(thought_info, utterance)

        elif thought_type == "_complement_gaps":
            reply, response_template = self._phrase_complement_gaps(thought_info, utterance)

        elif thought_type == "_subject_gaps":
            reply, response_template = self._phrase_subject_gaps(thought_info, utterance)

        elif thought_type == "_overlaps":
            reply, response_template = self._phrase_overlaps(thought_info, utterance)

        elif thought_type == "_trust":
            reply, response_template = self._phrase_trust(thought_info, utterance)

        if fallback and reply is None:  # Fallback strategy
            reply, response_template = self.phrase_fallback()

        # Formatting
        if reply:
            reply = casefold_text(reply, format='natural')

        return reply, response_template

    @staticmethod
    def _phrase_cardinality_conflicts(conflicts: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser)._phrase_cardinality_conflicts(conflicts, utterance)

        # There is no conflict, so nothing
        if not conflicts:
            return say, capsule_user

        # There is a conflict, so we phrase it
        else:
            conflict = random.choice(conflicts)

            # Capsule with other conflicting triple, user should set the polarity and certainty
            capsule_user['subject'] = utterance['subject']
            capsule_user['predicate'] = utterance['predicate']
            capsule_user["object"] = {"label": conflict['_complement']['_label'],
                                      "type": conflict['_complement']['_type'],
                                      'uri': conflict['_complement']['_id']}

        return say, capsule_user

    @staticmethod
    def _phrase_negation_conflicts(conflicts: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser)._phrase_negation_conflicts(conflicts, utterance)

        # There is no conflict, so no response
        if not conflicts or len(conflicts) < 2:
            return say, capsule_user

        # There is conflict entries
        else:
            affirmative_conflict = [item for item in conflicts if item['_polarity_value'] == 'POSITIVE']
            negative_conflict = [item for item in conflicts if item['_polarity_value'] == 'NEGATIVE']

            # There is a conflict, so we phrase it
            if affirmative_conflict and negative_conflict:
                # Capsule with original triple, user should set the polarity and certainty
                capsule_user['subject'] = utterance['subject']
                capsule_user['predicate'] = utterance['predicate']
                capsule_user['object'] = utterance['object']

        return say, capsule_user

    @staticmethod
    def _phrase_statement_novelty(novelties: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser)._phrase_statement_novelty(novelties, utterance)

        novelties = novelties["provenance"]

        # I do not know this before, so be happy to learn
        if not novelties:
            if label_in_say(utterance['subject']['label'], say):
                # Capsule with original subject & predicate, user should change object so we keep learning similar facts
                capsule_user['subject'] = utterance['subject']
                capsule_user['predicate'] = utterance['predicate']

            else:
                # Capsule with some original parts, user should change subject so we keep learning similar facts
                capsule_user['predicate'] = utterance['predicate']
                capsule_user['object'] = utterance['object']

        # I already knew this
        else:
            novelties = [n for n in novelties if label_in_say(n['_provenance']['_author']['_label'], say)]
            novelty = random.choice(novelties)
            # Capsule with author as triple subject, user should say something about that author
            capsule_user["subject"] = {"label": novelty['_provenance']['_author']['_label'],
                                       "type": novelty['_provenance']['_author']['_types'],
                                       "uri": novelty['_provenance']['_author']['_id']}

        return say, capsule_user

    @staticmethod
    def _phrase_type_novelty(novelties: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser)._phrase_type_novelty(novelties, utterance)

        # There is no novelty information, so no response
        if not novelties:
            return say, capsule_user

        novelty = novelties['_subject'] if label_in_say(utterance['subject']['label'], say) else novelties['_complement']

        # never heard it
        if not novelty:
            # Capsule with original triple subject or object, user should add predicate and other entity to keep learning
            capsule_user['subject'] = utterance['subject'] if label_in_say(utterance['subject']['label'], say) \
                else capsule_user['subject']
            capsule_user['object'] = utterance['object'] if not label_in_say(utterance['subject']['label'], say) \
                else capsule_user['object']

        # I know this
        else:
            # Capsule with original triple predicate, user should add other entities to keep learning
            capsule_user['predicate'] = utterance['predicate']
            capsule_user['subject'] = utterance['subject'] if label_in_say(utterance['subject']['label'], say) \
                else capsule_user['subject']
            capsule_user['object'] = utterance['object'] if not label_in_say(utterance['subject']['label'], say) \
                else capsule_user['object']

        return say, capsule_user

    @staticmethod
    def _phrase_subject_gaps(all_gaps: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser)._phrase_subject_gaps(all_gaps, utterance)

        # There is no gaps, so no response
        if not all_gaps:
            return say, capsule_user

        gaps = all_gaps['_subject'] if label_in_say(utterance['subject']['label'], say) else all_gaps['_complement']

        if not gaps:
            return say, capsule_user

        gap = random.choice(gaps)

        if label_in_say(utterance['subject']['label'], say):
            # Capsule with original triple subject + gap info, user should add object label
            capsule_user['subject'] = utterance['subject']
            capsule_user['predicate'] = {"label": gap["_predicate"]["_label"], "uri": gap["_predicate"]["_id"]}
            capsule_user['object'] = {"label": None, "type": gap["_target_entity_type"]["_types"], "uri": None}


        else:
            # Capsule with original triple object + gap info, user should add subject label
            capsule_user['subject'] = {"label": None, "type": gap['_target_entity_type']['_types'], 'uri': None}
            capsule_user['predicate'] = {"label": gap['_predicate']['_label'], 'uri': gap['_predicate']['_id']}
            capsule_user['object'] = utterance['subject']

        return say, capsule_user

    @staticmethod
    def _phrase_complement_gaps(all_gaps: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser)._phrase_complement_gaps(all_gaps, utterance)

        # There is no gaps, so no response
        if not all_gaps:
            return say, capsule_user

        # random choice between object or subject
        gaps = all_gaps['_subject'] if label_in_say(utterance['subject']['label'], say) else all_gaps['_complement']

        if not gaps:
            return say, capsule_user

        if label_in_say(utterance['subject']['label'], say):
            # Capsule with original object as subject + gap info, user should add object label
            capsule_user['subject'] = utterance['object']
            capsule_user['predicate'] = {"label": gaps[0]['_predicate']['_label'], 'uri': gaps[0]['_predicate']['_id']}
            capsule_user['object'] = {"label": None, "type": gaps[0]['_target_entity_type']['_types'], 'uri': None}

        else:
            # Capsule with original triple object + gap info, user should add subject label
            capsule_user['subject'] = {"label": None, "type": gaps[0]['_target_entity_type']['_types'], 'uri': None}
            capsule_user['predicate'] = {"label": gaps[0]['_predicate']['_label'], 'uri': gaps[0]['_predicate']['_id']}
            capsule_user['object'] = utterance['object']

        return say, capsule_user

    @staticmethod
    def _phrase_overlaps(all_overlaps: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser)._phrase_overlaps(all_overlaps, utterance)

        if not all_overlaps:
            return say, capsule_user

        overlaps = all_overlaps['_subject'] if label_in_say(utterance['subject']['label'], say) else all_overlaps['_complement']

        if not overlaps:
            return say, capsule_user

        overlaps = clean_overlaps(overlaps)

        if len(overlaps) < 2 and label_in_say(utterance['subject']['label'], say):
            overlap = random.choice(overlaps)
            # Capsule with original triple subject + overlap info, user should add perspective
            capsule_user['subject'] = utterance['subject']
            capsule_user['predicate'] = utterance['predicate']
            capsule_user['object'] = {"label": overlap['_entity']['_label'],
                                      "type": overlap['_entity']['_types'],
                                      'uri': overlap['_entity']['_id']}

        elif len(overlaps) < 2 and not label_in_say(utterance['subject']['label'], say):
            overlap = random.choice(overlaps)
            # Capsule with original triple object + overlap info, user should add perspective
            capsule_user['subject'] = {"label": overlap['_entity']['_label'],
                                       "type": overlap['_entity']['_types'],
                                       'uri': overlap['_entity']['_id']}
            capsule_user['predicate'] = utterance['predicate']
            capsule_user['object'] = utterance['object']

        # More than two cases, can we generalize?
        elif label_in_say(utterance['subject']['label'], say):
            # Capsule with original triple, user should add object
            capsule_user['subject'] = utterance['subject']
            capsule_user['predicate'] = utterance['predicate']

        elif not label_in_say(utterance['subject']['label'], say):
            # Capsule with original triple, user should add subject
            capsule_user['predicate'] = utterance['predicate']
            capsule_user['object'] = utterance['object']

        return say, capsule_user

    @staticmethod
    def _phrase_trust(trust: dict, utterance: dict) -> Tuple[Optional[str], Optional[dict]]:
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser)._phrase_trust(trust, utterance)

        # Capsule with speaker trusts entity
        capsule_user['subject'] = utterance['author']
        capsule_user['predicate'] = {'label': 'trust', 'uri': ONTOLOGY_DETAILS['namespace'] + 'trust'}

        return say, capsule_user

    @staticmethod
    def phrase_fallback() -> Tuple[Optional[dict], Optional[dict]]:
        # Capsule with empty triple, redirect conversation
        capsule_user, say = copy.deepcopy(BASE_CAPSULE), \
                            super(TriplePhraser, TriplePhraser).phrase_fallback()

        return say, capsule_user

    def reply_to_statement(self, brain_response, persist=False):
        """
        Phrase a thought based on the brain response
        Parameters
        ----------
        brain_response: output of the brain
        persist: Call fallback

        Returns
        -------

                """
        # Quick check if there is anything to do here
        if not brain_response['statement']['triple']:
            return copy.deepcopy(BASE_CAPSULE), None

        # # Casefold
        # utterance = casefold_capsule(brain_response['statement'], format='natural')
        # thoughts = casefold_capsule(brain_response['thoughts'], format='natural')

        # Generate reply
        (thought_type, thought_info) = list(brain_response['thoughts'].items())[0]
        reply = self.phrase_correct_thought(brain_response['statement'], thought_type, thought_info, fallback=persist)

        if persist and reply[0] is None and reply[1] is None:
            reply = self.phrase_fallback()

        return reply
