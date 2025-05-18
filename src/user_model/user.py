from datetime import datetime
from pathlib import Path
from random import choice

from rdflib import ConjunctiveGraph, URIRef

from cltl.brain.infrastructure.rdf_builder import RdfBuilder
from cltl.brain.utils.helper_functions import brain_response_to_json
from cltl.brain.utils.helper_functions import hash_claim_id
from cltl.commons.discrete import UtteranceType, Certainty, Polarity, Sentiment
from cltl.reply_generation.llama_phraser import LlamaPhraser
from dialogue_system.utils.global_variables import HARRYPOTTER_NS, HARRYPOTTER_PREFIX, RESOURCES_PATH, \
    RAW_VANILLA_USER_PATH, USER_PATH
from user_model import logger


def uri_to_capsule_triple(uri, response_template, role='subject'):
    if role == 'subject':
        entity_type = "character"
    elif role == 'object':
        entity_type = "attribute"
    else:
        entity_type = None

    response_template[role] = {'label': uri.split('/')[-1], 'type': [entity_type], 'uri': uri}
    return response_template


class User(object):
    def __init__(self, kb_filepath=RAW_VANILLA_USER_PATH, ontology_filepath=RESOURCES_PATH + "hp_data/ontology.ttl"):
        """Sets up a user with a triple database.

        returns: None
        """

        self._log = logger.getChild(self.__class__.__name__)
        self._log.info("Booted")

        self._graph = ConjunctiveGraph()
        self._rdf_builder = RdfBuilder(ontology_details={"filepath": ontology_filepath,
                                                         "namespace": HARRYPOTTER_NS, "prefix": HARRYPOTTER_PREFIX})

        self._replier = LlamaPhraser()

        # parse data and namespaces
        self._graph.parse(kb_filepath)
        self._log.info(f"Parsed file {Path(kb_filepath).resolve().relative_to(Path(USER_PATH).resolve())}, "
                       f"size of graph is {len(self._graph)}")

    def init_capsule(self, args, chatbot):
        capsule = {
            "chat": chatbot.chat_id,
            "turn": chatbot.turns,
            "author": {
                "label": chatbot.speaker.lower(),
                "type": ["person"],
                "uri": f"http://cltl.nl/leolani/world/{chatbot.speaker.lower()}"
            },
            "utterance": "James was a female",
            "utterance_type": "STATEMENT",
            "position": '0-25',
            'subject': {'label': 'james', 'type': ['person'],
                        'uri': 'http://harrypotter.org/james'},
            'predicate': {'label': 'gender', 'uri': 'http://harrypotter.org/gender'},
            'object': {'label': 'female', 'type': ['attribute'],
                       'uri': 'http://harrypotter.org/female'},
            'perspective': {'certainty': 1, 'polarity': 1, 'sentiment': 0},
            'timestamp': datetime.now(), 'context_id': args.context_id}

        # Choose a random triple
        action = "random triple"
        capsule = self._random_triple(capsule)

        # Prepare capsule as brain response
        capsule = self._prepare_capsule(capsule)

        return action, capsule

    def query_database(self, response_template):
        # Full triple in response template, just looking for perspective
        if (response_template['subject']['uri'] is not None) and \
                (response_template['predicate']['uri'] is not None) and \
                (response_template['object']['uri'] is not None):

            action = "response perspective"
            response_template = self._response_perspective(response_template)

            return action, response_template

        # Missing triple element, query
        else:
            filter_clause = ''

            # Subject
            if response_template['subject']['uri'] is not None:
                subject_uri = URIRef(response_template['subject']['uri'])
            else:
                subject_uri = '?s'
                filter_clause += '?s a gaf:Instance .\n'

            # predicate
            if response_template['predicate']['uri'] is not None:
                predicate_uri = URIRef(response_template['predicate']['uri'])
            else:
                predicate_uri = '?p'
                filter_clause += 'FILTER (STRSTARTS(STR(?p), STR(hp:)))\n'

            # object
            if response_template['object']['uri'] is not None:
                object_uri = URIRef(response_template['object']['uri'])
            else:
                object_uri = '?o'
                filter_clause += '?o a gaf:Instance .\n'

            query = f"""
            SELECT ?s ?p ?o WHERE {{ {subject_uri} {predicate_uri} {object_uri} .
            {filter_clause} }}"""

            self._log.debug(f"Query from given template")
            query = self._replace_namespace(query)
            response = self._graph.query(query)

            if len(list(response)) > 0:
                action = "response triple"
                response_template = self._response_triple(response_template, response)

            else:
                action = "random triple"
                response_template = self._random_triple(response_template)

        # Prepare capsule as brain response
        response_template = self._prepare_capsule(response_template)

        return action, response_template

    def _prepare_capsule(self, response_template):
        response_template['triple'] = self._rdf_builder.fill_triple(response_template['subject'],
                                                                    response_template['predicate'],
                                                                    response_template['object'])
        response_template['perspective'] = self._rdf_builder.fill_perspective(response_template['perspective']) \
            if 'perspective' in response_template.keys() else self._rdf_builder.fill_perspective({})
        response_template['utterance_type'] = UtteranceType[response_template['utterance_type']] \
            if type(response_template['utterance_type']) == str else response_template['utterance_type']
        response_template = brain_response_to_json(response_template)
        response_template['utterance'] = self._replier.phrase_triple(response_template)

        return response_template

    def _replace_namespace(self, txt):
        txt = txt.replace(HARRYPOTTER_NS, f'{HARRYPOTTER_PREFIX}:')

        prefixes = ""
        for prefix, namespace in self._rdf_builder.namespaces.items():
            if namespace in txt:
                txt = txt.replace(namespace, f'{prefix.lower()}:')
                prefixes = prefixes + f"prefix {prefix.lower()}: <{namespace}>\n"

        return prefixes + txt

    def _response_perspective(self, response_template):
        # perspective
        response_template = self._query_perspective_on_claim(response_template)

        # utterance
        response_template['utterance'] = self._replier.phrase_triple(response_template)

        return response_template

    def _response_triple(self, response_template, response):
        selection = choice(list(response))

        # Subject
        if response_template['subject']['uri'] is None:
            response_template = uri_to_capsule_triple(selection[0], response_template, role='subject')

        # predicate
        if response_template['predicate']['uri'] is None:
            response_template = uri_to_capsule_triple(selection[1], response_template, role='predicate')

        # object
        if response_template['object']['uri'] is None:
            response_template = uri_to_capsule_triple(selection[2], response_template, role='object')

        # perspective
        response_template = self._query_perspective_on_claim(response_template)

        # utterance
        response_template['utterance'] = self._replier.phrase_triple(response_template)

        return response_template

    def _random_triple(self, response_template):
        query = f"""SELECT ?s ?p ?o  WHERE {{ ?s ?p ?o . ?s a gaf:Instance . ?o a gaf:Instance . 
        FILTER (STRSTARTS(STR(?p), STR(hp:))) }}"""

        self._log.debug(f"Query for random triple")
        query = self._replace_namespace(query)
        response = self._graph.query(query)

        if len(list(response)) > 0:
            response = choice(response.bindings)

        # Fill triple
        response_template = uri_to_capsule_triple(response['s'], response_template, role='subject')
        response_template = uri_to_capsule_triple(response['p'], response_template, role='predicate')
        response_template = uri_to_capsule_triple(response['o'], response_template, role='object')

        # perspective
        response_template = self._query_perspective_on_claim(response_template)

        return response_template

    def _query_perspective_on_claim(self, response_template):
        claim = "http://cltl.nl/leolani/world/" + hash_claim_id([response_template["subject"]["label"],
                                                                 response_template["predicate"]["label"],
                                                                 response_template["object"]["label"]])

        query = f"""SELECT distinct ?certainty ?polarity ?sentiment WHERE {{ 
                ?certainty rdf:type graspf:CertaintyValue . 
                ?polarity rdf:type graspf:PolarityValue . 
                ?sentiment rdf:type grasps:SentimentValue . 

                ?mention gaf:denotes {claim} . 
                ?mention grasp:hasAttribution ?attribution .
                ?attribution rdf:value ?certainty .
                ?attribution rdf:value ?polarity .
                ?attribution rdf:value ?sentiment .  }}"""

        self._log.debug(f"Query to obtain perspective")
        query = self._replace_namespace(query)
        response = self._graph.query(query)

        # Fact is known and has a perspective
        if len(list(response)) > 0:
            response = choice(response.bindings)

            # Fill perspective
            response_template['perspective'] = {'certainty': Certainty.from_str(response["certainty"].split('#')[-1]),
                                                'polarity': Polarity.from_str(response["polarity"].split('#')[-1]),
                                                'sentiment': Sentiment.from_str(response["sentiment"].split('#')[-1])}

        # Fact is not known or has no perspective attached, need another random triple
        else:
            response_template = self._random_triple(response_template)

        return response_template
