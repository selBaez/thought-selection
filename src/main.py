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
import traceback

from cltl.brain.utils.base_cases import statements

from src.chatbot.chatbots import Chatbot

ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__))
RESOURCES_PATH = ABSOLUTE_PATH + "/../resources/"
THOUGHTS_FILE = RESOURCES_PATH + "thoughts.json"


def main(args):
    """Runs the main interaction loop of the chatbot."""
    # Sets up chatbot with a Lenka-, RL- or NSPReplier
    chatbot = Chatbot(args.chat_id, args.speaker, args.mode, args.reward, args.savefile)
    print("\nBot:", chatbot.greet)

    # Interaction loop
    for capsule in statements:
        # while True:
        #     capsule = input("\nYou: ")
        #     json.loads(capsule)

        if capsule == "quit":
            break

        if capsule == "plot":
            chatbot.replier.thought_selector.plot(filename=RESOURCES_PATH)
            continue

        try:
            say, capsule_user, brain_response = chatbot.respond(capsule)
            print("\nCarl:", json.dumps(brain_response['statement'], indent=2))
            print("\nBot:", say)
            print("\nCarl desired:", json.dumps(capsule_user, indent=2))
        except:
            print(traceback.format_exc())
            break

    # Farewell + update savefile
    print("\nBot:", chatbot.farewell)
    chatbot.replier.thought_selector.plot(filename=RESOURCES_PATH)
    chatbot.close_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chat_id", default=1, type=int, help="Id for chat (e.g. 1)"
    )
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
        "--reward",
        default="Total triples",
        type=str,
        choices=["Total triples", "Total classes", "Total predicates"],
        help="Reward for RL algorithm",
    )
    parser.add_argument(
        "--savefile",
        default=THOUGHTS_FILE,
        type=str,
        help="Path to BERT for NSP (--method=NSP) or RL JSON (--method=RL)",
    )
    args = parser.parse_args()
    print(
        "\n%s chat id and speaker %s.\nUsing mode=%s with reward %s and %s\n"
        % (args.mode, args.reward, args.savefile, args.chat_id, args.speaker)
    )

    main(args)
