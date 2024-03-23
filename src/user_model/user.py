from datetime import datetime
from random import choice

from cltl.brain.infrastructure.rdf_builder import RdfBuilder
from rdflib import Graph, URIRef

from src.dialogue_system.utils.global_variables import HARRYPOTTER_NS, HARRYPOTTER_PREFIX


def init_capsule(args, chatbot):
    # TODO pick a random triple

    init_cap = {
        "chat": chatbot.chat_id,
        "turn": chatbot.turns,
        "author": {
            "label": chatbot.speaker.lower(),
            "type": ["person"],
            "uri": f"http://cltl.nl/leolani/world/{chatbot.speaker.lower()}"
        },
        "utterance": "James was an adult",
        "utterance_type": "STATEMENT",
        "position": '0-25',
        'subject': {'label': 'james', 'type': ['person'],
                    'uri': 'http://harrypotter.org/james'},
        'predicate': {'label': 'age', 'uri': 'http://harrypotter.org/age'},
        'object': {'label': 'adult', 'type': ['age_range'],
                   'uri': 'http://harrypotter.org/adult'},
        'perspective': {'certainty': 1, 'polarity': 1, 'sentiment': 0},
        'timestamp': datetime.now(), 'context_id': args.context_id}

    return init_cap


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
    def __init__(self, kb_filepath='/Users/sbaez/Documents/PhD/data/harry potter dataset/Data/EN-data/all.ttl'):
        """Sets up a user with a triple database.

        returns: None
        """

        self._graph = Graph()
        self._rdf_builer = RdfBuilder(ontology_details={"filepath": kb_filepath.rsplit('/', 1)[0] + 'ontology.ttl',
                                                        "namespace": HARRYPOTTER_NS, "prefix": HARRYPOTTER_PREFIX})

        # parse data and namespaces
        # TODO parse correct type and instance of user
        self._graph.parse(kb_filepath)

    def replace_namespace(self, txt):
        txt = txt.replace(HARRYPOTTER_NS, f'{HARRYPOTTER_PREFIX}:')

        prefixes = ""
        for prefix, namespace in self._rdf_builer.namespaces.items():
            if namespace in txt:
                txt = txt.replace(namespace, f'{prefix.lower()}:')
                prefixes = prefixes + f"prefix {prefix.lower()}: <{namespace}>\n"

        return prefixes + txt

    def query_database(self, response_template):
        filter_clause = ''

        # Subject
        if response_template['subject']['uri'] is not None:
            subject_uri = URIRef(response_template['subject']['uri'])
        else:
            subject_uri = '?s'

        # predicate 
        if response_template['predicate']['uri'] is not None:
            predicate_uri = URIRef(response_template['predicate']['uri'])
        else:
            predicate_uri, filter_clause = '?p', 'FILTER (?p != rdf:type)'

        # object 
        if response_template['object']['uri'] is not None:
            object_uri = URIRef(response_template['object']['uri'])
        else:
            object_uri = '?o'

        # TODO query except types
        query = f"""
        SELECT ?s ?p ?o WHERE {{ {subject_uri} {predicate_uri} {object_uri} 
        {filter_clause} }}"""

        query = self.replace_namespace(query)
        response = self._graph.query(query)

        if len(response) > 0:
            response_template = self.response_triple(response_template, response)

        else:
            response_template = self.random_triple(response_template)

        return response_template

    def response_triple(self, response_template, response):
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

        # perspective TODO: Adapt querying to retrieve this and put it in capsule
        response_template['perspective'] = {'certainty': 1, 'polarity': 1, 'sentiment': 0}

        # utterance
        response_template['utterance'] = f"{response_template['subject']['label']} " \
                                         f"{response_template['predicate']['label']} " \
                                         f"{response_template['object']['label']}"

        return response_template

    def random_triple(self, response_template):
        query = f"""SELECT ?s ?p ?o  WHERE {{ ?s ?p ?o FILTER (?p != rdf:type) }}"""
        query = self.replace_namespace(query)
        response = self._graph.query(query)

        if len(response) > 0:
            response = choice(response.bindings)

        # Fill triple
        response_template = uri_to_capsule_triple(response['s'], response_template, role='subject')
        response_template = uri_to_capsule_triple(response['p'], response_template, role='predicate')
        response_template = uri_to_capsule_triple(response['o'], response_template, role='object')

        # perspective
        response_template['perspective'] = {'certainty': 1, 'polarity': 1, 'sentiment': 0}

        # utterance
        response_template['utterance'] = f"{response_template['subject']['label']} " \
                                         f"{response_template['predicate']['label']} " \
                                         f"{response_template['object']['label']}"

        return response_template
