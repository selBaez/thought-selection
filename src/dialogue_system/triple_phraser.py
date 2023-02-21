import copy
import random
from cltl.commons.language_data.sentences import NEW_KNOWLEDGE, EXISTING_KNOWLEDGE, CONFLICTING_KNOWLEDGE, \
    CURIOSITY, HAPPY
from cltl.commons.triple_helpers import filtered_types_names
from typing import Optional, Tuple, Union

from cltl.reply_generation.api import Phraser
from src.dialogue_system.utils.global_variables import BASE_CAPSULE


class TriplePhraser(Phraser):

    def __init__(self):
        # type: () -> None
        """
        Generate response as structured data

        Parameters
        ----------
        """

        super(Phraser, self).__init__()

    @staticmethod
    def _phrase_cardinality_conflicts(conflicts: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = None, None

        # There is no conflict, so nothing
        if not conflicts:
            return say, capsule_user

        # There is a conflict, so we phrase it
        else:
            say = random.choice(CONFLICTING_KNOWLEDGE)
            conflict = random.choice(conflicts)
            x = 'you' if conflict['_provenance']['_author']['_label'] == utterance['author'] \
                else conflict['_provenance']['_author']['_label']
            y = 'you' if utterance['triple']['_subject']['_label'] == conflict['_provenance']['_author']['_label'] \
                else utterance['triple']['_subject']['_label']

            say += ' %s told me in %s that %s %s %s, but now you tell me that %s %s %s' \
                   % (x, conflict['_provenance']['_date'],
                      y, utterance['triple']['_predicate']['_label'], conflict['_complement']['_label'],
                      y, utterance['triple']['_predicate']['_label'], utterance['triple']['_complement']['_label'])

            # Capsule with other conflicting triple, user should set the polarity and certainty
            capsule_user = copy.deepcopy(BASE_CAPSULE)
            capsule_user['subject'] = utterance['subject']
            capsule_user['predicate'] = utterance['predicate']
            capsule_user["object"] = {"label": conflict['_complement']['_label'],
                                      "type": conflict['_complement']['_type'],
                                      'uri': conflict['_complement']['_id']}

        return say, capsule_user

    @staticmethod
    def _phrase_negation_conflicts(conflicts: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = None, None

        # There is conflict entries
        if conflicts and conflicts[0]:
            affirmative_conflict = [item for item in conflicts if item['_polarity_value'] == 'POSITIVE']
            negative_conflict = [item for item in conflicts if item['_polarity_value'] == 'NEGATIVE']

            # There is a conflict, so we phrase it
            if affirmative_conflict and negative_conflict:
                say = random.choice(CONFLICTING_KNOWLEDGE)

                affirmative_conflict = random.choice(affirmative_conflict)
                negative_conflict = random.choice(negative_conflict)

                say += ' %s told me in %s that %s %s %s, but in %s %s told me that %s did not %s %s' \
                       % (affirmative_conflict['_provenance']['_author']['_label'],
                          affirmative_conflict['_provenance']['_date'],
                          utterance['triple']['_subject']['_label'], utterance['triple']['_predicate']['_label'],
                          utterance['triple']['_complement']['_label'],
                          negative_conflict['_provenance']['_date'],
                          negative_conflict['_provenance']['_author']['_label'],
                          utterance['triple']['_subject']['_label'], utterance['triple']['_predicate']['_label'],
                          utterance['triple']['_complement']['_label'])

                # Capsule with original triple, user should set the polarity and certainty
                capsule_user = copy.deepcopy(BASE_CAPSULE)
                capsule_user['subject'] = utterance['subject']
                capsule_user['predicate'] = utterance['predicate']
                capsule_user['object'] = utterance['object']

        return say, capsule_user

    @staticmethod
    def _phrase_statement_novelty(novelties: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = None, None

        # I do not know this before, so be happy to learn
        if not novelties:
            entity_role = random.choice(['subject', 'object'])
            say = random.choice(NEW_KNOWLEDGE)

            if entity_role == 'subject':
                if 'person' in filtered_types_names(utterance['triple']['_complement']['_types']):
                    any_type = 'anybody'
                elif 'location' in filtered_types_names(utterance['triple']['_complement']['_types']):
                    any_type = 'anywhere'
                else:
                    any_type = 'anything'

                say += ' I did not know %s that %s %s' % (any_type, utterance['triple']['_subject']['_label'],
                                                          utterance['triple']['_predicate']['_label'])

                # Capsule with original predicate and object, user should change subject so we keep learning similar facts
                capsule_user = copy.deepcopy(BASE_CAPSULE)
                capsule_user['predicate'] = utterance['predicate']
                capsule_user['object'] = utterance['object']

            elif entity_role == 'object':
                say += ' I did not know anybody who %s %s' % (utterance['triple']['_predicate']['_label'],
                                                              utterance['triple']['_complement']['_label'])

                # Capsule with original subject and predicate, user should change object so we keep learning similar facts
                capsule_user = copy.deepcopy(BASE_CAPSULE)
                capsule_user['subject'] = utterance['subject']
                capsule_user['predicate'] = utterance['predicate']

        # I already knew this
        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            novelty = random.choice(novelties)

            say += ' %s told me about it in %s' % (novelty['_provenance']['_author']['_label'],
                                                   novelty['_provenance']['_date'])

            # Capsule with author as triple subject, user should say something about that author
            capsule_user = copy.deepcopy(BASE_CAPSULE)
            capsule_user["subject"] = {"label": novelty['_provenance']['_author']['_label'],
                                       "type": novelty['_provenance']['_author']['_types'],
                                       "uri": novelty['_provenance']['_author']['_id']}

        return say, capsule_user

    @staticmethod
    def _phrase_type_novelty(novelties: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = None, None

        entity_role = random.choice(['subject', 'object'])
        entity_label = utterance['triple']['_subject']['_label'] \
            if entity_role == 'subject' else utterance['triple']['_complement']['_label']
        novelty = novelties['_subject'] if entity_role == 'subject' else novelties['_complement']

        if not novelty:
            say = random.choice(NEW_KNOWLEDGE)
            if entity_label != 'you':
                say += ' I had never heard about %s before!' % entity_label
            else:
                say += ' I am excited to get to know about %s!' % entity_label

            # Capsule with original triple subject or object, user should add predicate and other entity to keep learning
            capsule_user = copy.deepcopy(BASE_CAPSULE)
            capsule_user['subject'] = utterance['subject'] if entity_role == 'subject' else capsule_user['subject']
            capsule_user['object'] = utterance['object'] if entity_role == 'object' else None

        else:
            say = random.choice(EXISTING_KNOWLEDGE)
            if entity_label != 'you':
                say += ' I have heard about %s before' % entity_label
            else:
                say += ' I love learning more and more about %s!' % entity_label

            # Capsule with original triple predicate, user should add other entities to keep learning
            capsule_user = copy.deepcopy(BASE_CAPSULE)
            capsule_user['predicate'] = utterance['predicate']
            capsule_user['subject'] = utterance['subject'] if entity_role == 'subject' else capsule_user['subject']
            capsule_user['object'] = utterance['object'] if entity_role == 'object' else None

        return say, capsule_user

    @staticmethod
    def _phrase_subject_gaps(all_gaps: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = None, None

        entity_role = random.choice(['subject', 'object'])
        gaps = all_gaps['_subject'] if entity_role == 'subject' else all_gaps['_complement']

        if entity_role == 'subject':
            say = random.choice(CURIOSITY)

            if not gaps:
                say += ' What types can %s %s' % (utterance['triple']['_subject']['_label'],
                                                  utterance['triple']['_predicate']['_label'])

                # Capsule with original triple, user should add object type # TODO(carl - ? - ?)
                capsule_user = copy.deepcopy(BASE_CAPSULE)
                capsule_user['subject'] = utterance["subject"]
                capsule_user['predicate'] = utterance["predicate"]
                capsule_user['object'] = {"label": None, "type": utterance["object"]["type"], "uri": None}

            else:
                gap = random.choice(gaps)
                if 'is ' in gap['_predicate']['_label'] or ' is' in gap['_predicate']['_label']:
                    say += ' Is there a %s that %s %s?' % (filtered_types_names(gap['_entity']['_types']),
                                                           gap['_predicate']['_label'],
                                                           utterance['triple']['_subject']['_label'])

                elif ' of' in gap['_predicate']['_label']:
                    say += ' Is there a %s that %s is %s?' % (filtered_types_names(gap['_entity']['_types']),
                                                              utterance['triple']['_subject']['_label'],
                                                              gap['_predicate']['_label'])

                elif ' ' in gap['_predicate']['_label']:
                    say += ' Is there a %s that is %s %s?' % (filtered_types_names(gap['_entity']['_types']),
                                                              gap['_predicate']['_label'],
                                                              utterance['triple']['_subject']['_label'])
                else:
                    say += ' Has %s %s %s?' % (utterance['triple']['_subject']['_label'],
                                               gap['_predicate']['_label'],
                                               filtered_types_names(gap['_entity']['_types']))

                # Capsule with original triple subject + gap info, user should add object label
                capsule_user = copy.deepcopy(BASE_CAPSULE)
                capsule_user['subject'] = utterance['subject']
                capsule_user['predicate'] = {"label": gap["_predicate"]["_label"], "uri": gap["_predicate"]["_id"]}
                capsule_user['object'] = {"label": None, "type": gap["_entity"]["_types"], "uri": None}

        elif entity_role == 'object':
            say = random.choice(CURIOSITY)

            if not gaps:
                say += ' What kinds of things can %s a %s like %s' % (utterance['triple']['_predicate']['_label'],
                                                                      utterance['triple']['_complement']['_label'],
                                                                      utterance['triple']['_subject']['_label'])

                # Capsule with original triple, user should add subject type # TODO(? - ? - carl)
                capsule_user = copy.deepcopy(BASE_CAPSULE)
                capsule_user['subject'] = {"label": None, "type": utterance["subject"]["type"], "uri": None}
                capsule_user['predicate'] = utterance["predicate"]
                capsule_user['object'] = utterance["object"]

            else:
                gap = random.choice(gaps)
                if '#' in filtered_types_names(gap['_entity']['_types']):
                    say += ' What is %s %s?' % (utterance['triple']['_subject']['_label'],
                                                gap['_predicate']['_label'])
                elif ' ' in gap['_predicate']['_label'] or '-' in gap['_predicate']['_label']:
                    say += ' Has %s ever %s %s?' % (filtered_types_names(gap['_entity']['_types']),
                                                    gap['_predicate']['_label'],
                                                    utterance['triple']['_subject']['_label'])

                else:
                    say += ' Has %s ever %s a %s?' % (utterance['triple']['_subject']['_label'],
                                                      gap['_predicate']['_label'],
                                                      filtered_types_names(gap['_entity']['_types']))

                # Capsule with original triple object + gap info, user should add subject label
                capsule_user = copy.deepcopy(BASE_CAPSULE)
                capsule_user['subject'] = {"label": None, "type": gap['_entity']['_types'], 'uri': None}
                capsule_user['predicate'] = {"label": gap['_predicate']['_label'], 'uri': gap['_predicate']['_id']}
                capsule_user['object'] = utterance['subject']

        return say, capsule_user

    @staticmethod
    def _phrase_complement_gaps(all_gaps: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = None, None

        # random choice between object or subject
        entity_role = random.choice(['subject', 'object'])
        gaps = all_gaps['_subject'] if entity_role == 'subject' else all_gaps['_complement']

        if entity_role == 'subject':
            say = random.choice(CURIOSITY)

            if not gaps:
                say += ' What types can %s %s' % (utterance['triple']['_subject']['_label'],
                                                  utterance['triple']['_predicate']['_label'])

                # Capsule with original triple, user should add object type # TODO(pills - ? - ?)
                capsule_user = copy.deepcopy(BASE_CAPSULE)
                capsule_user['subject'] = utterance["subject"]
                capsule_user['predicate'] = utterance["predicate"]
                capsule_user['object'] = {"label": None, "type": utterance["object"]["type"], "uri": None}

            else:
                gap = random.choice(gaps)
                if ' in' in gap['_predicate']['_label']:  # ' by' in gap['_predicate']['_label']
                    say += ' Is there a %s %s %s?' % (filtered_types_names(gap['_entity']['_types']),
                                                      gap['_predicate']['_label'],
                                                      utterance['triple']['_complement']['_label'])
                else:
                    say += ' Has %s %s by a %s?' % (utterance['triple']['_complement']['_label'],
                                                    gap['_predicate']['_label'],
                                                    filtered_types_names(gap['_entity']['_types']))

                # Capsule with original object as subject + gap info, user should add object label
                capsule_user = copy.deepcopy(BASE_CAPSULE)
                capsule_user['subject'] = utterance['subject']
                capsule_user['predicate'] = {"label": gap['_predicate']['_label'], 'uri': gap['_predicate']['_id']}
                capsule_user['object'] = {"label": None, "type": gap['_entity']['_types'], 'uri': None}

        elif entity_role == 'object':
            say = random.choice(CURIOSITY)

            if not gaps:
                otypes = filtered_types_names(utterance['triple']['_complement']['_types'])
                otypes = otypes if otypes != '' else 'things'
                stypes = filtered_types_names(utterance['triple']['_subject']['_types'])
                stypes = stypes if stypes != '' else 'actors'
                say += ' What types of %s like %s do %s usually %s' % (otypes,
                                                                       utterance['triple']['_complement']['_label'],
                                                                       stypes,
                                                                       utterance['triple']['_predicate']['_label'])

                # Capsule with original predicate and complement type, user should add object label # TODO(? - ? - pills)
                capsule_user = copy.deepcopy(BASE_CAPSULE)
                capsule_user['subject'] = {"label": None, "type": utterance["subject"]["type"], "uri": None}
                capsule_user['predicate'] = utterance["predicate"]
                capsule_user['object'] = {"label": None, "type": utterance["object"]["type"], "uri": None}

            else:
                gap = random.choice(gaps)
                if '#' in filtered_types_names(gap['_entity']['_types']):
                    say += ' What is %s %s?' % (utterance['triple']['_complement']['_label'],
                                                gap['_predicate']['_label'])
                elif ' by' in gap['_predicate']['_label']:
                    say += ' Has %s ever %s a %s?' % (utterance['triple']['_complement']['_label'],
                                                      gap['_predicate']['_label'],
                                                      filtered_types_names(gap['_entity']['_types']))
                else:
                    say += ' Has a %s ever %s %s?' % (filtered_types_names(gap['_entity']['_types']),
                                                      gap['_predicate']['_label'],
                                                      utterance['triple']['_complement']['_label'])

                # Capsule with original triple object + gap info, user should add subject label
                capsule_user = copy.deepcopy(BASE_CAPSULE)
                capsule_user['subject'] = {"label": None, "type": gap['_entity']['_types'], 'uri': None}
                capsule_user['predicate'] = {"label": gap['_predicate']['_label'], 'uri': gap['_predicate']['_id']}
                capsule_user['object'] = utterance['object']

        return say, capsule_user

    @staticmethod
    def _phrase_overlaps(all_overlaps: dict, utterance: dict) -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = None, None

        entity_role = random.choice(['subject', 'object'])
        overlaps = all_overlaps['_subject'] if entity_role == 'subject' else all_overlaps['_complement']

        if not overlaps:
            return say, capsule_user

        elif len(overlaps) < 2 and entity_role == 'subject':
            say = random.choice(HAPPY)
            overlap = random.choice(overlaps)

            say += ' Did you know that %s also %s %s' % (utterance['triple']['_subject']['_label'],
                                                         utterance['triple']['_predicate']['_label'],
                                                         overlap['_entity']['_label'])

            # Capsule with original triple subject + overlap info, user should add perspective
            capsule_user = copy.deepcopy(BASE_CAPSULE)
            capsule_user['subject'] = utterance['subject']
            capsule_user['predicate'] = utterance['predicate']
            capsule_user['object'] = {"label": overlap['_entity']['_label'],
                                      "type": overlap['_entity']['_types'],
                                      'uri': overlap['_entity']['_id']}

        elif len(overlaps) < 2 and entity_role == 'object':
            say = random.choice(HAPPY)
            overlap = random.choice(overlaps)

            say += ' Did you know that %s also %s %s' % (overlap['_entity']['_label'],
                                                         utterance['triple']['_predicate']['_label'],
                                                         utterance['triple']['_complement']['_label'])

            # Capsule with original triple object + overlap info, user should add perspective
            capsule_user = copy.deepcopy(BASE_CAPSULE)
            capsule_user['subject'] = {"label": overlap['_entity']['_label'],
                                       "type": overlap['_entity']['_types'],
                                       'uri': overlap['_entity']['_id']}
            capsule_user['predicate'] = utterance['predicate']
            capsule_user['object'] = utterance['object']

        # More than two cases, can we generalize?
        elif entity_role == 'subject':
            say = random.choice(HAPPY)
            sample = random.sample(overlaps, 2)

            entity_0 = sample[0]['_entity']['_label']
            entity_1 = sample[1]['_entity']['_label']

            say += ' Now I know %s items that %s %s, like %s and %s' % (len(overlaps),
                                                                        utterance['triple']['_subject']['_label'],
                                                                        utterance['triple']['_predicate']['_label'],
                                                                        entity_0, entity_1)

            # Capsule with original triple, user should add object
            capsule_user = copy.deepcopy(BASE_CAPSULE)
            capsule_user['subject'] = utterance['subject']
            capsule_user['predicate'] = utterance['predicate']

        elif entity_role == 'object':
            say = random.choice(HAPPY)
            sample = random.sample(overlaps, 2)
            types = filtered_types_names(sample[0]['_entity']['_types']) if sample[0]['_entity']['_types'] else 'things'
            say += ' Now I know %s %s that %s %s, like %s and %s' % (len(overlaps), types,
                                                                     utterance['triple']['_predicate']['_label'],
                                                                     utterance['triple']['_complement']['_label'],
                                                                     sample[0]['_entity']['_label'],
                                                                     sample[1]['_entity']['_label'])

            # Capsule with original triple, user should add subject
            capsule_user = copy.deepcopy(BASE_CAPSULE)
            capsule_user['predicate'] = utterance['predicate']
            capsule_user['object'] = utterance['object']

        return say, capsule_user

    @staticmethod
    def _phrase_trust(trust: float, utterance: dict) -> Tuple[Optional[str], Optional[dict]]:
        capsule_user, say = None, super(Phraser)._phrase_trust(trust)

        # Capsule with speaker trusts entity
        capsule_user = copy.deepcopy(BASE_CAPSULE)
        capsule_user['subject'] = {"label": utterance['author'], "type": [], "uri": None}
        capsule_user['predicate'] = {'label': 'trust', 'uri': 'http://cltl.nl/leolani/n2mu/trust'}

        return say, capsule_user

    @staticmethod
    def phrase_fallback() -> Tuple[Optional[dict], Optional[dict]]:
        capsule_user, say = None, super(Phraser).phrase_fallback()

        # Capsule with empty triple, redirect conversation
        capsule_user = copy.deepcopy(BASE_CAPSULE)

        return say, capsule_user

    def phrase_correct_thought(self, utterance, thought_type, thought_info, fallback=False) -> Tuple[Optional[str], Optional[dict]]:
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
            reply = reply.replace("-", " ").replace("  ", " ")

        return reply, response_template

    # def phrase_correct_thought(self, utterance, thought_type, thought_info, fallback=False) -> Tuple[Optional[dict], Optional[dict]]:
    #     capsule_user, say = None, super(Phraser).phrase_correct_thought(utterance, thought_type, thought_info, fallback=True)
    #
    #     # Capsule with empty triple, redirect conversation
    #     capsule_user = copy.deepcopy(BASE_CAPSULE)
    #
    #     return say, capsule_user
