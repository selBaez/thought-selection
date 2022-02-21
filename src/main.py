import json
import os
from pathlib import Path

from cltl.brain.long_term_memory import LongTermMemory
from cltl.brain.utils.helper_functions import brain_response_to_json
from cltl.combot.backend.api.discrete import UtteranceType
from cltl.reply_generation.rl_replier import RLReplier
from tqdm import tqdm

# Read scenario from file
ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = ABSOLUTE_PATH + f'/../data'
INPUT_FOLDER = DATA_FOLDER + f'/combots_convos_capsules/'
OUTPUT_FOLDER = DATA_FOLDER + f'/processed/'

scenario_file_name = '0.json'
scenario_json_file = INPUT_FOLDER + scenario_file_name


def converse():
    f = open(scenario_json_file, )
    scenario = json.load(f)

    # Create objects
    brain = LongTermMemory(address="http://localhost:7200/repositories/thought-selection",
                           log_dir=Path("../logs"),
                           clear_all=True)

    replier = RLReplier(brain, Path(ABSOLUTE_PATH + "/../resources/thoughts.json"))

    # Recreate conversation through ingesting capsules
    for capsule in tqdm(scenario):
        # Add information to the brain
        print(f"\n\n---------------------------------------------------------------\n")

        # STATEMENT
        if capsule["utterance_type"] in [UtteranceType.STATEMENT, 'STATEMENT']:
            # Update Brain -> communicate a thought
            brain_response = brain.update(capsule, reason_types=True, create_label=True)
            brain_response = brain_response_to_json(brain_response)

            # Romanas chimp here between brain before and after update,
            # get data as trig from triple store

            # Reward chosen thought
            replier.reward_thought()
            reply = replier.reply_to_statement(brain_response)

            # Report
            print(f"\n{capsule['triple']}\n")
            print(f"{capsule['author']}: {capsule['utterance']}")
            print(f"Leolani: {reply}")
            print(f"Brain states: {replier.brain_states}")


if __name__ == "__main__":
    converse()

    print('ALL DONE')
