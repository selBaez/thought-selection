import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path

import torch
from iribaker import to_iri
from rdflib import URIRef
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import HeteroData, InMemoryDataset

from cltl.brain.utils.helper_functions import hash_claim_id
from src.user_model.utils.helpers import build_graph, get_all_characters, get_all_attributes, get_all_predicates, \
    RAW_USER_PATH, PROCESSED_USER_PATH, SIMPLE_ATTRRELS, SIMPLE_ATTVALS


def ingest_claims_and_perspectives(og_graph):
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
    print(f"QUERY DATASET, OBTAINED {len(response)} CLAIMS")

    # Turn into rdf
    graph = build_graph()
    for el in response:
        subject = URIRef(to_iri(el["claim"]))

        certainty_predicate = URIRef("http://groundedannotationframework.org/grasp/factuality#CertaintyValue")
        certainty = URIRef(to_iri(el["certainty"]))
        graph.add((subject, certainty_predicate, certainty))

        polarity_predicate = URIRef("http://groundedannotationframework.org/grasp/factuality#PolarityValue")
        polarity = URIRef(to_iri(el["polarity"]))
        graph.add((subject, polarity_predicate, polarity))

        sentiment_predicate = URIRef("http://groundedannotationframework.org/grasp/sentiment#SentimentValue")
        sentiment = URIRef(to_iri(el["sentiment"]))
        graph.add((subject, sentiment_predicate, sentiment))
    print(f"NEW SIMPLE DATASET CONTAINS {len(graph)} TRIPLES")

    return graph


class HarryPotterRDF(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        # build vocabulary
        og_graph = build_graph()
        og_graph.parse(self.raw_paths[0])
        print(f"READ DATASET in {self.raw_paths[0]}: {len(og_graph)}")
        self.build_vocabulary(og_graph)

        # check if data has been processed
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return RAW_USER_PATH

    @property
    def processed_dir(self) -> str:
        return PROCESSED_USER_PATH

    @property
    def raw_file_names(self) -> list:
        return ["vanilla.trig"]

    @property
    def processed_file_names(self):
        return ["vanilla.pt"]

    def download(self):
        """
        Method to get data from a graphDB reporsitory as opposed to a file
        """
        # Download to `self.raw_dir`.

        # get everything
        response = self.chatbot.brain._connection.export_repository()
        graph_data = build_graph()
        graph_data.parse(response)
        graph_data.serialize(destination=self.raw_dir + self.chatbot.scenario_folder.split('/')[-1])

    def process(self):
        # Get all files to process
        files = list(Path(self.raw_dir).glob('*.trig'))

        for file in files:
            # Read raw data
            og_graph = build_graph()
            og_graph.parse(file)
            print(f"READ DATASET in {file}: {len(og_graph)}")

            # Ingest claims and perspectives
            graph = ingest_claims_and_perspectives(og_graph)

            # Create edge representation
            edge_data = self.create_edge_representation(graph)

            # Create node representation
            node_types = self.create_node_representation(graph)

            # Save data
            data = HeteroData(
                edge_index=torch.tensor(edge_data['edge_index'], dtype=torch.long).t().contiguous(),
                edge_type=torch.tensor(edge_data['edge_type'], dtype=torch.long),
                node_type=torch.tensor(node_types, dtype=torch.float)
            )
            processed_filename = self.processed_dir + f"/{file.stem}.pt"
            torch.save(self.collate([data]), processed_filename)

        print("here")

    def build_vocabulary(self, full_graph):
        """
        Create dictionaries to get node and edges IDs
        """
        # Get everything from the instance graph in the oracle user
        all_characters = [str(node["character"]) for node in get_all_characters(full_graph)]
        all_attributes = [str(node["attribute"]) for node in get_all_attributes(full_graph)]
        all_predicates = [str(rel["predicate"]) for rel in get_all_predicates(full_graph)]
        self.subjects_dict = {node: i for i, node in enumerate(all_characters)}
        self.objects_dict = {node: i for i, node in enumerate(all_attributes)}
        self.predicates_dict = {rel: i for i, rel in enumerate(all_predicates)}

        # Create all possible claims from the combinations of the known nodes
        all_claims = list(product(all_characters, all_predicates, all_attributes))
        all_claims = ["http://cltl.nl/leolani/world/" + hash_claim_id([s.split('/')[-1],
                                                                       p.split('/')[-1],
                                                                       o.split('/')[-1]]) for (s, p, o) in all_claims]
        self.claims_dict = {node: i for i, node in enumerate(all_claims)}
        self.attvals_dict = {str(node): i for i, node in enumerate(SIMPLE_ATTVALS)}
        self.attrels_dict = {str(rel): i for i, rel in enumerate(SIMPLE_ATTRRELS)}

    def create_edge_representation(self, graph):
        """
        Create edge index and edge types matrices
        """
        edge_data = defaultdict(list)
        for s, p, o in graph.triples((None, None, None)):
            src, dst, rel = self.claims_dict.get(str(s), -1), self.attvals_dict[str(o)], self.attrels_dict[str(p)]
            if src != -1:
                edge_data['edge_index'].append([src, dst])
                edge_data['edge_type'].append(rel)

        return edge_data

    def create_node_representation(self, graph):
        """
        Create node types matrix
        """
        types = [["character"] for s in set(graph.subjects())]
        types.extend([["attribute"] for o in set(graph.objects())])
        mlb = MultiLabelBinarizer()
        node_types = mlb.fit_transform(types)

        # TODO check that these indices correspond to the node ids in the dictionary
        return node_types


def main(args):
    dataset = HarryPotterRDF('.')

    dataset.process()

    print("here")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)
