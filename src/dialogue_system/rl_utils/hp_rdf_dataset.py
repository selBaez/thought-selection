from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import torch
from iribaker import to_iri
from rdflib import URIRef
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import HeteroData, InMemoryDataset

from cltl.brain.utils.helper_functions import hash_claim_id
from dialogue_system import logger
from dialogue_system.utils.global_variables import RAW_USER_PATH, RAW_VANILLA_USER_PATH, PROCESSED_USER_PATH, \
    TYPE_CERTAINTYVALUE, TYPE_POLARITYVALUE, TYPE_SENTIMENTVALUE, SIMPLE_ATTRELS, SIMPLE_ATTVALS, ATTVALS_TO_ATTRELS
from dialogue_system.utils.helpers import build_graph, get_all_characters, get_all_predicates, get_all_attributes

TEST = True
PROCESS_FOR_GRAPH_CLASSIFIER = False


def one_hot_feature(entity, entities_dict):
    """
    Initialize a vector with zeros, search for the index of an entity and assign one in that index
    """
    # Initialize zero vector
    one_hot_vector = np.zeros(len(entities_dict))

    if entity:
        # Determine what type of node this is, and treat grasp perspectives in a special way
        if entity.startswith("http://groundedannotationframework.org/grasp/"):
            prefix = ""
            entity = str(ATTVALS_TO_ATTRELS[URIRef(entity)])
        else:
            prefix = "http://harrypotter.org/"

        # Assign value
        index = entities_dict[prefix + entity]
        one_hot_vector[index] = 1

    return one_hot_vector


# noinspection PyAttributeOutsideInit
class HarryPotterRDF(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):

        self._log = logger.getChild(self.__class__.__name__)
        self._log.info("Booted")

        # build vocabulary from vanilla graph
        og_graph = build_graph()
        og_graph.parse(RAW_VANILLA_USER_PATH, format="trig")
        self._log.debug(f"Read dataset in {Path(RAW_VANILLA_USER_PATH).resolve()}: {len(og_graph)}")

        # build resources that we only need once (IDs, features and node types)
        self.build_vocabulary(og_graph)
        self.compute_node_features()
        self.compute_node_types()

        # check if data has been processed
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return RAW_USER_PATH

    @property
    def processed_dir(self) -> str:
        return PROCESSED_USER_PATH

    @property
    def raw_file_names(self) -> list:
        files = list(Path(self.raw_dir).glob('*.trig'))
        files = sorted(files, reverse=True)
        return files

    @property
    def processed_file_names(self):
        files = []
        for file in self.raw_file_names:
            files.append(f"{self.processed_dir}/{file.stem}.pt")

        return files

    def get_claim_node_id(self, claim):
        return self.claims_dict.get(str(claim), -1)

    def get_perspective_node_id(self, perspective):
        # offset objects by subjects, to have all nodes have a unique id
        return self.perspectives_dict[str(perspective)] + len(self.claims_dict)

    def ingest_claims_and_perspectives(self, og_graph):
        # Generate query for assertions only
        query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX gaf: <http://groundedannotationframework.org/gaf#>
        PREFIX grasp: <http://groundedannotationframework.org/grasp#>
        PREFIX graspf: <http://groundedannotationframework.org/grasp/factuality#> 
        PREFIX grasps: <http://groundedannotationframework.org/grasp/sentiment#> 

        select distinct ?claim ?certainty ?polarity ?sentiment  where {{
            ?certainty rdf:type graspf:CertaintyValue . 
            ?polarity rdf:type graspf:PolarityValue . 
            ?sentiment rdf:type grasps:SentimentValue . 

            ?mention gaf:denotes ?claim . 
            ?mention grasp:hasAttribution ?attribution .
            ?attribution rdf:value ?certainty .
            ?attribution rdf:value ?polarity .
            ?attribution rdf:value ?sentiment .
        }}"""
        response = og_graph.query(query)
        self._log.debug(f"Query dataset, obtained {len(response)} claims")

        # Turn into rdf
        graph = build_graph()
        for el in response:
            subject = URIRef(to_iri(el["claim"]))
            certainty = URIRef(to_iri(el["certainty"]))
            graph.add((subject, TYPE_CERTAINTYVALUE, certainty))

            polarity = URIRef(to_iri(el["polarity"]))
            graph.add((subject, TYPE_POLARITYVALUE, polarity))

            sentiment = URIRef(to_iri(el["sentiment"]))
            graph.add((subject, TYPE_SENTIMENTVALUE, sentiment))
        self._log.debug(f"New simple datasets contains {len(graph)} triples")

        return graph

    def build_vocabulary(self, full_graph):
        """
        Create dictionaries to get node/entities and relation/predicate IDs.
        Nodes and relations: related to the main graph, with claims and their perspectives.
                         These are used for message passing.
        Entities and predicates: related to the knowledge inside the claims, connecting characters and their attributes.
                                 These are used as features of the main nodes.
        """
        # Get everything from the instance graph in the oracle user
        all_characters = [str(node["character"].replace("_", "")) for node in get_all_characters(full_graph)]
        all_attributes = [str(node["attribute"].replace("_", "")) for node in get_all_attributes(full_graph)]
        all_predicates = [str(rel["predicate"].replace("_", "")) for rel in get_all_predicates(full_graph)]
        self.characters_dict = {node: i for i, node in enumerate(all_characters)}
        self.attributes_dict = {node: i for i, node in enumerate(all_attributes)}
        self.predicates_dict = {rel: i for i, rel in enumerate(all_predicates)}

        # Create all possible claims from the combinations of the known nodes
        all_claims = list(product(all_characters, all_predicates, all_attributes))
        all_claims = ["http://cltl.nl/leolani/world/" + hash_claim_id([s.split('/')[-1],
                                                                       p.split('/')[-1],
                                                                       o.split('/')[-1]]) for (s, p, o) in all_claims]
        self.claims_dict = {str(node): i for i, node in enumerate(all_claims)}
        self.perspectives_dict = {str(node): i for i, node in enumerate(SIMPLE_ATTVALS)}
        self.rels_dict = {str(rel): i for i, rel in enumerate(SIMPLE_ATTRELS)}

        # Set dimensions
        self.NUM_NODES = len(self.claims_dict) + len(self.perspectives_dict)
        self.NUM_RELATIONS = len(self.rels_dict)

        self.NUM_ENTITIES = len(self.characters_dict) + len(self.attributes_dict)
        self.NUM_PREDICATES = len(self.predicates_dict)
        self.NUM_FEATURES = self.NUM_ENTITIES + self.NUM_PREDICATES + self.NUM_RELATIONS

    def compute_node_features(self):
        """
        Create feature vector in one hot vectors, concatenated
         Claims: encoding original subject, predicate, object (character, predicate, attribute)
         Perspectives: encode their type (factuality, sentiment or emotion)
        """
        # Initialize matrix
        node_features = np.zeros([self.NUM_NODES, self.NUM_FEATURES])

        # Representation for claims
        for claim in self.claims_dict.keys():
            # Get original SPO values in claim
            try:
                sub, pred, obj = claim.split("/")[-1].split("_")
            except:
                # Too many underscores
                sub, pred, obj = claim.split("/")[-1].split("_", 2)
                obj = obj.replace("_", "")

            # one hot encode each SPO part
            subject_one_hot = one_hot_feature(sub, self.characters_dict)
            predicates_one_hot = one_hot_feature(pred, self.predicates_dict)
            object_one_hot = one_hot_feature(obj, self.attributes_dict)
            type_one_hot = one_hot_feature(None, self.rels_dict)

            # Concatenate for final claim representation, assign to node index
            claim_rep = np.concatenate((subject_one_hot, predicates_one_hot, object_one_hot, type_one_hot), axis=0)
            index = self.get_claim_node_id(claim)
            node_features[index] = claim_rep

        # Representation for perspectives
        for perspective in self.perspectives_dict.keys():
            # one hot their type
            subject_one_hot = one_hot_feature(None, self.characters_dict)
            predicates_one_hot = one_hot_feature(None, self.predicates_dict)
            object_one_hot = one_hot_feature(None, self.attributes_dict)
            type_one_hot = one_hot_feature(perspective, self.rels_dict)

            # Concatenate for final type representation, assign to node index
            perspective_rep = np.concatenate((subject_one_hot, predicates_one_hot, object_one_hot, type_one_hot),
                                             axis=0)
            index = self.get_perspective_node_id(perspective)
            node_features[index] = perspective_rep

        self.node_features = node_features

    def compute_node_types(self):
        """
        Create node types matrix
        """
        # These are claims, not characters! same for attributions values
        types = [["claim"] for _ in self.claims_dict.keys()]
        types.extend([[o.split("/")[-1].split("#")[0]] for o in self.perspectives_dict.keys()])
        mlb = MultiLabelBinarizer()
        node_types = mlb.fit_transform(types)

        self.node_types = node_types

    def process(self):
        data_list = []
        if PROCESS_FOR_GRAPH_CLASSIFIER:
            for raw_file, processed_file in zip(self.raw_file_names, self.processed_file_names):
                data = self.process_one_graph(raw_file)
                torch.save(self.collate([data]), processed_file)
                data_list.append(data)

                if TEST:
                    break
        return data_list

    def process_one_graph(self, file):
        # Read raw data
        og_graph = build_graph()
        og_graph.parse(file, format="trig")
        self._log.debug(f"Read dataset in {file.name}: {len(og_graph)}")

        # Ingest claims and perspectives
        graph = self.ingest_claims_and_perspectives(og_graph)

        # Create edge representation
        edge_data = self.create_edge_representation(graph)

        # # Create node type representation
        # node_types = self.create_node_type_representation(graph)

        # Create node feature representation
        node_features = self.create_node_feature_representation(graph)

        # Save data
        data = HeteroData(
            edge_index=torch.tensor(edge_data['edge_index'], dtype=torch.long).t().contiguous(),
            edge_type=torch.tensor(edge_data['edge_type'], dtype=torch.long),
            # node_type=torch.tensor(node_types, dtype=torch.float)
            node_features=torch.tensor(node_features, dtype=torch.long)
        )

        return data

    def create_edge_representation(self, graph):
        """
        Create edge index and edge types matrices
        """
        # edge_data = {"edge_index": [[0,0]], "edge_type": []}
        edge_data = defaultdict(list)

        # Loop through triples in graphs, all claims with their perspectives
        for s, p, o in graph.triples((None, None, None)):
            # get indices of each part, from their own dictionaries #]
            src, dst, rel = self.get_claim_node_id(s), \
                            self.get_perspective_node_id(o), \
                            self.rels_dict[str(p)]

            # register claim, if it exists (some claims might not exist if subjects and objects are scrambled)
            if src != -1:
                edge_data['edge_index'].append([src, dst])
                edge_data['edge_type'].append(rel)

        return edge_data

    def create_node_type_representation(self, graph):
        """
        Create node types vector
        """
        # Initialize matrix
        node_types = np.zeros([self.NUM_NODES])

        # Select existing nodes
        to_keep = [self.get_claim_node_id(claim) for claim in set(graph.subjects())]
        to_keep.extend([self.get_perspective_node_id(perspective) for perspective in set(graph.objects())])

        # assign pre-coputed features
        for index in to_keep:
            node_types[index] = self.node_types[index]

        return node_types

    def create_node_feature_representation(self, graph):
        """
        Create feature vector in one hot vectors, concatenated
         Claims: encoding original subject, predicate, object (character, predicate, attribute)
         Perspectives: encode their type (factuality, sentiment or emotion)
        """
        # Initialize matrix
        node_features = np.zeros([self.NUM_NODES, self.NUM_FEATURES])

        # Select existing nodes
        to_keep = [self.get_claim_node_id(claim) for claim in set(graph.subjects())]
        to_keep.extend([self.get_perspective_node_id(perspective) for perspective in set(graph.objects())])

        # assign pre-coputed features
        for index in to_keep:
            node_features[index] = self.node_features[index]

        return node_features

