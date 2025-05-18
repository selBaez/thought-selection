from datetime import datetime
from pathlib import Path
from random import choice

from rdflib import ConjunctiveGraph

from dialogue_system.utils.global_variables import RESOURCES_PATH, RAW_USER_PATH, HARRYPOTTER_PREFIX, HARRYPOTTER_NS
from user_model.utils.constants import user_model_names


def cast_actions_to_json(actions):
    action_history = []
    for el in actions:
        if el:
            action_history.append(int(el[0][0]))
        else:
            action_history.append(None)

    return action_history


def select_entity_type(selected_action):
    entity_types = selected_action['entity_types']
    max_val = max(entity_types.values())
    candidates = [k for k, v in entity_types.items() if v == max_val]
    most_important_type = choice(candidates)

    return most_important_type


def search_session_folder(experiment_id, run_id, context_id, reward, chat_id):
    prev_sess = create_session_folder(experiment_id, f"{run_id}", context_id, reward, chat_id, '*', make_folder=False)
    prev_sess = list(prev_sess.parent.glob(prev_sess.name))[0]

    return prev_sess


def create_session_folder(experiment_id, run_id, context_id, reward, chat_id, speaker, make_folder=True):
    # Create folder to store session
    session_folder = Path(f"{RESOURCES_PATH}"
                          f"experiments/"
                          f"{experiment_id}/"
                          f"{reward.replace(' ', '-')}/"
                          f"{run_id}/"
                          f"{reward.replace(' ', '-')}_"
                          f"{chat_id}_"
                          f"{speaker.replace(' ', '-')}"
                          f"({context_id})/")
    if make_folder:
        session_folder.mkdir(parents=True, exist_ok=True)

    return session_folder


def download_from_triplestore(brain, directory=RAW_USER_PATH):
    """
    Method to get data from a graphDB repository as opposed to a file.
    Returns the filename where the data gets stored
    """
    # get everything
    response = brain._connection.export_repository()
    graph_data = build_graph()
    graph_data.parse(data=response, format="trig")

    # save
    raw_file = directory / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.trig"
    graph_data.serialize(destination=raw_file, format="trig")
    return raw_file


def build_graph():
    """
    Build graph to put dataset in
    """
    graph_data = ConjunctiveGraph()
    graph_data.bind(HARRYPOTTER_PREFIX, HARRYPOTTER_NS)

    return graph_data


def get_all_characters(graph_data):
    """
    Query graph for characters (subjects in triples)
    """
    q_characters = """SELECT distinct ?character  WHERE {{ ?character rdf:type hp:character . }}"""
    all_characters = graph_data.query(q_characters)
    all_characters = [c for c in all_characters]

    print(f"\tCHARACTERS IN DATASET: {len(all_characters)}")

    return all_characters


def get_all_predicates(graph_data):
    """
    Query graph for predicates (relations in triples)
    """
    q_predicates = """SELECT distinct ?predicate  WHERE {{ ?s ?predicate ?o . 
                        FILTER(STRSTARTS(STR(?predicate), STR(hp:))) . }}"""
    all_predicates = graph_data.query(q_predicates)
    all_predicates = [c for c in all_predicates]

    print(f"\tPREDICATES IN DATASET: {len(all_predicates)}")

    return all_predicates


def get_all_attributes(graph_data):
    """
    Query graph for attributes (objects in triples)
    """
    q_attributes = """SELECT distinct ?attribute  WHERE {{ ?attribute rdf:type hp:attribute . }}"""
    all_attributes = graph_data.query(q_attributes)
    all_attributes = [c for c in all_attributes]

    print(f"\tATTRIBUTES IN DATASET: {len(all_attributes)}")

    return all_attributes


def replace_user_name(user_model):
    x = user_model.split("/")[-1].split(".")[0]
    for user_type, user_name in user_model_names.items():
        x = x.replace(user_type, user_name)

    return x
