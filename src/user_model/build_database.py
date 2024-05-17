import json
from copy import deepcopy
from datetime import datetime

from iribaker import to_iri

from cltl.brain.long_term_memory import LongTermMemory
from cltl.commons.discrete import UtteranceType, Certainty, Polarity, Sentiment
from src.dialogue_system.utils.global_variables import ONTOLOGY_DETAILS, RAW_USER_PATH, HARRYPOTTER_NS
from src.dialogue_system.utils.helpers import get_all_characters, get_all_attributes
from src.user_model.utils.constants import CONTEXT_ID, START_DATE, HP_CONTEXT_CAPSULE
from src.user_model.utils.helpers import *

TEST = True
NUM_USERS_PER_TYPE = 10 if TEST else 100


def add_triple(book, chapter, position, counter, name, character, relation, val, brain):
    capsule = {
        "chat": book,  # book
        "turn": chapter,  # chapter
        "author": {"label": "JK Rowling", "type": ["person", "author"],
                   'uri': "http://cltl.nl/leolani/friends/jk-rowling"},
        "utterance": "",
        "utterance_type": UtteranceType.STATEMENT,
        "position": f"{position}-{int(position) + counter}",
        "subject": {"label": name, "type": ["character"],
                    'uri': to_iri(HARRYPOTTER_NS + character)},
        "predicate": {"label": relation, "uri": to_iri(HARRYPOTTER_NS + relation)},
        "object": {"label": val, "type": ["attribute"],
                   'uri': to_iri(HARRYPOTTER_NS + val)},
        "perspective": {"certainty": Certainty.CERTAIN, "polarity": Polarity.POSITIVE,
                        "sentiment": Sentiment.NEUTRAL},
        "timestamp": datetime.combine(START_DATE.replace(year=START_DATE.year + int(book)),
                                      datetime.now().time()),
        "context_id": CONTEXT_ID
    }
    _ = brain.capsule_statement(capsule, reason_types=False, create_label=True, return_thoughts=False)
    print(f"\t{capsule['triple']}")


def process_session(book, session, brain):
    chapter = session["position"].split("-")[0][7:]
    position = session["position"].split("-")[1]
    counter = 0

    for charactr, attributes in session["attributes"].items():
        # Split the messy items
        subjs = break_list(charactr)

        for character in subjs:
            # Make subject node
            character = clean_string(character)

            for relation, value in attributes.items():
                # Make predicate resource
                relation = clean_string(relation)
                if relation == "character":
                    relation = "personality"

                # Split the messy items
                value = value.split(",")

                for val in value:
                    # Clean string
                    val = clean_string(val)

                    # Corner cases
                    if val in [None, "none", ""] or val.lower() == relation.lower() or relation in ["name", "nickname"]:
                        continue

                    else:
                        # Add to graph
                        add_triple(book, chapter, position, counter, attributes["name"], character, relation, val,
                                   brain)
                        counter += 1

                    # Future work: add relations to harry


def process_file(file):
    # Create brain connection
    brain = LongTermMemory(address="http://localhost:7200/repositories/harryPotter",
                           log_dir=Path("/Users/sbaez/Documents/PhD/research/thought-selection/resources/users"),
                           ontology_details=ONTOLOGY_DETAILS,
                           clear_all=True)

    # Create context
    _ = brain.capsule_context(HP_CONTEXT_CAPSULE)

    # Read JSON file
    print(f"\nFILE: {file}")
    with open(file) as json_file:
        data = json.load(json_file)

    # Iterate through dialogues, and attributes
    for context, session in data.items():
        book = get_book_number(file)
        process_session(book, session, brain)

    # Report and save per dialogue
    save_graph(filepath=file.with_suffix(".trig"), graph_data=brain.dataset)


def remove_claim(claim, graph_data):
    # Find attribution and delete direct references to it
    att = get_all_attribution_of_claim(graph_data, claim)
    graph_data.remove((att, None, None))  # (1) types + (1) label + (x) mentions + (4) values
    graph_data.remove((None, None, att))  # (x) mentions

    # Find mentions
    all_men = get_all_mentions_of_claim(graph_data, claim)
    for men in all_men:
        # Delete direct references to mentions
        men = men["mention"]
        graph_data.remove((men, None, None))  # (2) types + (1) label + (2) instances + (1) claim + (1?) attribution
        #                                     + (1) source + (1) timestamp + (1) value + (1) utterance
        graph_data.remove((None, None, men))  # (2) instances +  (1) claim + (1?) attribution

    # Delete direct references to claim
    graph_data.remove((None, None, None, claim))  # (1) data inside the claim
    graph_data.remove((claim, None, None))  # (2) types + (1) label + (x?) mentions
    graph_data.remove((None, None, claim))  # (x?) mentions

    return graph_data


def doubt_claim(claim, graph_data):
    # Find attributions and delete
    att = get_all_attribution_of_claim(graph_data, claim)
    graph_data.remove((att, None, None))  # (1) types + (1) label + (x) mentions + (4) values
    graph_data.remove((None, None, att))  # (x) mentions

    # Make new attribution, with low certainty
    graph_data, new_att = lower_certainty(graph_data, att)

    # Find mentions
    all_men = get_all_mentions_of_claim(graph_data, claim)
    for men in all_men:
        graph_data = link_mention_to_att(graph_data, men, new_att)

    return graph_data


def negate_claim(claim, graph_data):
    # Find attributions and delete
    att = get_all_attribution_of_claim(graph_data, claim)
    graph_data.remove((att, None, None))  # (1) types + (1) label + (x) mentions + (4) values
    graph_data.remove((None, None, att))  # (x) mentions

    # Make new attribution, with negative polarity
    graph_data, new_att = negate_polarity(graph_data, att)

    # Find mentions and link new att
    all_men = get_all_mentions_of_claim(graph_data, claim)
    for men in all_men:
        graph_data = link_mention_to_att(graph_data, men, new_att)

    return graph_data


def corrupt_claim(claim, graph_data, all_characters, all_attributes):
    # Make new claim
    new_claim, elems, new_elems = create_new_claim(graph_data, claim, all_characters, all_attributes)

    # Make new attribution
    graph_data, new_att = create_new_att(graph_data, new_claim)

    # Find mentions and link new claim and att
    all_men = get_all_mentions_of_claim(graph_data, claim)
    for men in all_men:
        # Delete old links
        graph_data.remove((men["mention"], GAF_CONTAINSDEN, elems["s"], PERSPECTIVE_GRAPH))
        graph_data.remove((men["mention"], GAF_CONTAINSDEN, elems["o"], PERSPECTIVE_GRAPH))
        graph_data.remove((elems["s"], GAF_DENOTEDIN, men["mention"], INSTANCE_GRAPH))
        graph_data.remove((elems["o"], GAF_DENOTEDIN, men["mention"], INSTANCE_GRAPH))

        # Add new links
        graph_data = link_mention_to_claim(graph_data, men, new_claim, new_elems)
        graph_data = link_mention_to_att(graph_data, men, new_att)

    # Find old attribution and delete
    att = get_all_attribution_of_claim(graph_data, claim)
    graph_data.remove((att, None, None))  # (1) types + (1) label + (x) mentions + (4) values
    graph_data.remove((None, None, att))  # (x) mentions

    # Delete direct references to old claim
    graph_data.remove((None, None, None, claim))  # (1) data inside the claim
    graph_data.remove((claim, None, None))  # (2) types + (1) label + (x?) mentions
    graph_data.remove((None, None, claim))  # (x?) mentions

    return graph_data


def create_users(graph_data):
    all_claims = get_all_claims(graph_data)
    all_characters = get_all_characters(graph_data)
    all_attributes = get_all_attributes(graph_data)

    # VANILLA: all data
    print(f"\nCREATING USER TYPE: vanilla")
    save_graph(filepath=Path(RAW_USER_PATH) / "vanilla.trig", graph_data=graph_data)

    # AMATEUR: remove 50% of claims (including all their attributions and links to the mentions)
    print(f"\nCREATING USER TYPE: amateur")
    for u in range(NUM_USERS_PER_TYPE):
        u_graph = deepcopy(graph_data)

        # Sample claims and remove them
        claims_to_delete = sample_claims(all_claims)
        print(f"DELETING {len(claims_to_delete)} CLAIMS")
        for claim in claims_to_delete:
            u_graph = remove_claim(claim["claim"], u_graph)
        save_graph(filepath=Path(RAW_USER_PATH) / f"amateur{u}.trig", graph_data=u_graph)

    # DOUBTFUL: lower certainty of 50% of claims
    print(f"\nCREATING USER TYPE: doubtful")
    for u in range(NUM_USERS_PER_TYPE):
        # Add probable certainty
        u_graph = deepcopy(graph_data)
        u_graph = create_low_certainty(u_graph)

        # Sample claims and edit them
        claims_to_doubt = sample_claims(all_claims)
        print(f"DOUBTING {len(claims_to_doubt)} CLAIMS")
        for claim in claims_to_doubt:
            u_graph = doubt_claim(claim["claim"], u_graph)
        save_graph(filepath=Path(RAW_USER_PATH) / f"doubtful{u}.trig", graph_data=u_graph)

    # INCOHERENT: negate 50% of claims
    print(f"\nCREATING USER TYPE: incoherent")
    for u in range(NUM_USERS_PER_TYPE):
        # Add negative polarity
        u_graph = deepcopy(graph_data)
        u_graph = create_neg_polairty(u_graph)

        # Sample claims and edit them
        claims_to_negate = sample_claims(all_claims)
        print(f"NEGATING {len(claims_to_negate)} CLAIMS")
        for claim in claims_to_negate:
            u_graph = negate_claim(claim["claim"], u_graph)
        save_graph(filepath=Path(RAW_USER_PATH) / f"incoherent{u}.trig", graph_data=u_graph)

    # CONFUSED: switch an element in the triple for another (valid: existing and equivalent) element
    print(f"\nCREATING USER TYPE: confused")
    for u in range(NUM_USERS_PER_TYPE):
        u_graph = deepcopy(graph_data)

        # Sample claims and edit them
        claims_to_corrupt = sample_claims(all_claims)
        print(f"CORRUPTING {len(claims_to_corrupt)} CLAIMS")
        for claim in claims_to_corrupt:
            u_graph = corrupt_claim(claim["claim"], u_graph, all_characters, all_attributes)
        save_graph(filepath=Path(RAW_USER_PATH) / f"confused{u}.trig", graph_data=u_graph)


def main():
    print("---------------------------- Ingest triples per book  ----------------------------")
    # iterate through JSONs
    files = get_all_files(extension="json")
    for file in files:
        process_file(file)

    print("---------------------------- Make users  ----------------------------")
    # Make types of users
    graph_data = merge_all_graphs(TEST)
    create_users(graph_data)


if __name__ == "__main__":
    main()
