import json
from pathlib import Path

from iribaker import to_iri
from rdflib import ConjunctiveGraph, URIRef, Namespace
from nltk import word_tokenize, pos_tag

DATA_PATHS = ["/Users/sbaez/Downloads/harry potter dataset/Data/EN-data/test_set_en/",
              "/Users/sbaez/Downloads/harry potter dataset/Data/EN-data/train_set_en/"]

CHARACTERS_NS = Namespace("http://harrypotter.org/characters/")
PREDICATES_NS = Namespace("http://harrypotter.org/predicates/")
ATTRIBUTES_NS = Namespace("http://harrypotter.org/attributes/")


def clean_string(text):
    """
    Get a string and remove special characters and empty spaces.
    Casefolding to first letter capitalized (arbitrary decision)
    Remove conjunctions/articles at the beginning
    """
    for punct in ["(", ")"]:
        text = text.replace(punct, "")
    text = text.strip()
    text = text.capitalize()

    # # POS tagging
    # words = word_tokenize(text)
    # poss = pos_tag(words)
    # if poss[0][1] in ["CC", "DT"]:
    #     # TODO: remove first word. Think? do we need to remove all conjuctions or just the "and"
    #     pass

    if text.startswith("And "):
        text = text[4:]
        text = text.capitalize()

    return text


def build_graph():
    """
    Build graph to put dataset in
    """
    graph_data = ConjunctiveGraph()
    graph_data.bind("hpCharacters", CHARACTERS_NS)
    graph_data.bind("hpPredicates", PREDICATES_NS)
    graph_data.bind("hpAttributes", ATTRIBUTES_NS)

    return graph_data


def get_all_files():
    """
    Get all JSON files that belong to the dataset
    """
    files = []
    for folder in DATA_PATHS:
        files.extend(list(Path(folder).glob('*.json')))

    return files


# iterate through JSONS
graph_data = build_graph()
files = get_all_files()
for file in files:
    # Read JSON file
    with open(file) as json_file:
        data = json.load(json_file)

    # Iterate through dialogues, and attributes
    for _, session in data.items():
        for character, attributes in session["attributes"].items():
            # Make subject node
            character = clean_string(character)
            character_node = URIRef(to_iri(CHARACTERS_NS + character))

            for relation, value in attributes.items():
                # Make predicate resource
                relation = clean_string(relation)
                predicate = URIRef(to_iri(PREDICATES_NS + relation))

                # Split the messy items
                value = value.split(",")

                for val in value:
                    # Clean string
                    val = clean_string(val)

                    # Corner cases
                    if val in [None, "None", ""] or val.lower() == relation.lower():
                        continue

                    # Make object node
                    value_node = URIRef(to_iri(ATTRIBUTES_NS + val))

                    # Add to graph
                    graph_data.add((character_node, predicate, value_node))

    # Report and save per dialogue
    triples_file = file.with_suffix(".ttl")
    graph_data.serialize(destination=triples_file)
    print(f"\n\nSIZE OF DATASET: {len(graph_data)}")

# Report and save
triples_file = file.parent.parent / "all.ttl"
graph_data.serialize(destination=triples_file)
print(f"\n\nFINAL SIZE OF DATASET: {len(graph_data)}")
