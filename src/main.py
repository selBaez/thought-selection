""" Filename:     main.py
    Author(s):    Thomas Bellucci
    Description:  This file contains the interaction loop for the Chatbot defined in chatbots.py. By
                  typing 'plot' the chatbot will plot a graph with learnt thought statistics (if mode='RL')
                  and 'quit' ends the interaction.
    Date created: Nov. 11th, 2021
"""
import argparse
import json
import os
from datetime import date
from random import getrandbits

import requests
from cltl.brain.utils.base_cases import statements

from src.chatbot.chatbots import Chatbot

ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__))
RESOURCES_PATH = ABSOLUTE_PATH + "/../resources/"
THOUGHTS_FILE = RESOURCES_PATH + "thoughts.json"

context_id = getrandbits(8)
place_id = getrandbits(8)
location = requests.get("https://ipinfo.io").json()

carl_scenario = [
    {
        "chat": 1,
        "turn": 1,
        "author": "carl",  # speaker of scenario
        "utterance": "I need to take my cough syrup, but I cannot find it.",  # sequence of mention
        "utterance_type": "STATEMENT",
        "position": "0-25",  # segment of the annotation
        "subject": {"label": "carl", "type": ["person"],
                    'uri': "http://cltl.nl/leolani/world/carl-1"},  # annotations of type NER
        "predicate": {"label": "see", "uri": "http://cltl.nl/leolani/n2mu/see"},  # annotation of type x
        "object": {"label": "cough syrup", "type": ["object", "medicine"],
                   'uri': "http://cltl.nl/leolani/world/cough-syrup"},  # annotations of type NER
        "perspective": {"certainty": 1, "polarity": -1, "sentiment": -1},  # annotation of type x (still to be done)
        "context_id": context_id,
        "date": date(2021, 3, 12).isoformat(),  # we take them from the temporal container of scenario
        "place": "Carl's room",
        "place_id": place_id,
        "country": location['country'],
        "region": location['region'],
        "city": location['city'],
        "objects": [{'type': 'chair', 'confidence': 0.68, 'id': 1},
                    {'type': 'table', 'confidence': 0.79, 'id': 1}],  # Usually come from Vision
        "people": [{'name': 'Carl', 'confidence': 0.94, 'id': 1}]  # Usually come from Vision
    },
    {
        "chat": 1,
        "turn": 2,
        "author": "leolani",
        "utterance": "I found them. They are under the table.",
        "utterance_type": "STATEMENT",
        "position": "0-25",
        "subject": {"label": "pills", "type": ["object"], 'uri': "http://cltl.nl/leolani/world/pills"},
        "predicate": {"label": "located under", "uri": "http://cltl.nl/leolani/n2mu/located-under"},
        "object": {"label": "table", "type": ["object"], 'uri': "http://cltl.nl/leolani/world/table"},
        "perspective": {"certainty": 1, "polarity": 1, "sentiment": 0},
        "context_id": context_id,
        "date": date(2021, 3, 12).isoformat(),
        "place": "Carl's room",
        "place_id": place_id,
        "country": location['country'],
        "region": location['region'],
        "city": location['city'],
        "objects": [{'type': 'chair', 'confidence': 0.56, 'id': 1},
                    {'type': 'table', 'confidence': 0.87, 'id': 1},
                    {'type': 'pillbox', 'confidence': 0.92, 'id': 1}],
        "people": []
    },
    {
        "chat": 1,
        "turn": 3,
        "author": "carl",
        "utterance": "Oh! Got it. Thank you.",
        "utterance_type": "STATEMENT",
        "position": "0-25",
        "subject": {"label": "carl", "type": ["person"], 'uri': "http://cltl.nl/leolani/world/carl-1"},
        "predicate": {"label": "see", "uri": "http://cltl.nl/leolani/n2mu/see"},
        "object": {"label": "pills", "type": ["object"], 'uri': "http://cltl.nl/leolani/world/pills"},
        "perspective": {"certainty": 1, "polarity": 1, "sentiment": 1},
        "context_id": context_id,
        "date": date(2021, 3, 12).isoformat(),
        "place": "Carl's room",
        "place_id": place_id,
        "country": location['country'],
        "region": location['region'],
        "city": location['city'],
        "objects": [{'type': 'chair', 'confidence': 0.59, 'id': 1},
                    {'type': 'table', 'confidence': 0.73, 'id': 1},
                    {'type': 'pillbox', 'confidence': 0.32, 'id': 1}],
        "people": [{'name': 'Carl', 'confidence': 0.98, 'id': 1}]
    }
]


def main(args):
    """Runs the main interaction loop of the chatbot."""
    # Sets up chatbot with a Lenka-, RL- or NSPReplier
    chatbot = Chatbot(args.speaker, args.mode, args.savefile)
    print("\nBot:", chatbot.greet)

    # Interaction loop
    for capsule in statements:
        # while True:
        #     capsule = input("\nYou: ")
        #     json.loads(capsule)

        if capsule == "quit":
            break

        if capsule == "plot":
            chatbot.replier._RLReplier__thought_selector.plot(filename=RESOURCES_PATH)
            continue

        try:
            say, capsule_user, brain_response = chatbot.respond(capsule)
            print("\nCarl:", json.dumps(brain_response['statement'], indent=2))
            print("\nBot:", say)
            print("\nCarl desired:", json.dumps(capsule_user, indent=2))
        except:
            break

    # Farewell + update savefile
    print("\nBot:", chatbot.farewell)
    chatbot.replier._RLReplier__thought_selector.plot(filename=RESOURCES_PATH)
    chatbot.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speaker", default="john", type=str, help="Name of the speaker (e.g. 'john')"
    )
    parser.add_argument(
        "--mode",
        default="RL",
        type=str,
        choices=["RL", "NSP", "Lenka"],
        help="Thought selection method: {'RL', 'NSP', 'Lenka'}",
    )
    parser.add_argument(
        "--savefile",
        default=THOUGHTS_FILE,
        type=str,
        help="Path to BERT for NSP (--method=NSP) or RL JSON (--method=RL)",
    )
    args = parser.parse_args()
    print(
        "\nUsing mode=%s with %s and speaker %s.\n"
        % (args.mode, args.savefile, args.speaker)
    )

    main(args)
