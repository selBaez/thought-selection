import os
from pathlib import Path

import requests
from cltl.brain.long_term_memory import LongTermMemory
from rdflib import Graph

from src.metrics.ontology_measures import *

# Read scenario from file
ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = ABSOLUTE_PATH + f'/../logs'
INPUT_FOLDER = DATA_FOLDER + f'/2022-02-20-20-22/'
OUTPUT_FOLDER = DATA_FOLDER + f'/processed/'

scenario_file_name = 'g1-piek/2021-12-07-16_26_14/rdf/2021-12-07-16-26/brain_log_2021-12-07-16-51-15.trig'
scenario_file = INPUT_FOLDER + scenario_file_name


def evaluate_brain():
    # Create Brains

    # Brain object in Python, fresh Dataset
    brain = LongTermMemory(address="http://localhost:7200/repositories/thought-selection",
                           log_dir=Path("../logs"),
                           clear_all=False)

    # Final log from corresponding convo
    brain_2 = Graph()
    brain_2.parse(scenario_file, format='trig')

    # From triple store
    get_url = "http://localhost:7200/repositories/thought-selection" + "/statements?infer=false"
    response = requests.get(get_url, headers={'Content-Type': 'application/x-' + 'trig'})
    brain_3 = Graph()
    brain_3.parse(data=response.text, format='trig')

    clsss = brain.get_classes()
    num_class = len(clsss)
    num_class_1 = get_number_classes(brain.dataset)

    num_class_2 = get_number_classes(brain_2)
    num_class_3 = get_number_classes(brain_3)

    print(f"Different methods, same brain: {num_class, num_class_1}")
    print(f"Same methods, different brain: {num_class_2, num_class_2}")


if __name__ == "__main__":
    evaluate_brain()

    print('ALL DONE')
