import argparse
import json
from cltl.brain.utils.base_cases import chat_1
from cltl.brain.utils.helper_functions import brain_response_to_json
from datetime import datetime

from src.dialogue_system.chatbot import Chatbot
from src.dialogue_system.utils.capsule_utils import template_to_statement_capsule
from src.dialogue_system.utils.global_variables import CONTEXT_ID, PLACE_ID, PLACE_NAME, LOCATION
from src.dialogue_system.utils.helpers import create_session_folder


def create_context_capsule(args):
    capsule = {}

    # context
    capsule["context_id"] = args.context_id
    capsule["date"] = datetime.strftime(args.context_date, "%Y-%m-%d")

    # place
    capsule["place"] = args.place_label
    capsule["place_id"] = args.place_id
    capsule["country"] = args.country
    capsule["region"] = args.region
    capsule["city"] = args.city

    return capsule


def print_bot(bot_utterance):
    print("\nBot:", bot_utterance)


def main(args):
    """Runs the main interaction loop of the chatbot."""
    # Sets up chatbot
    chatbot = Chatbot()

    # Create folder to store session
    session_folder = create_session_folder(args.reward, args.chat_id, args.speaker)

    # Create dialogue_system
    chatbot.begin_session(args.chat_id, args.speaker, args.reward, session_folder)

    # Situate chat
    capsule_for_context = create_context_capsule(args)
    chatbot.situate_chat(capsule_for_context)

    # Generate greeting for starting chat
    print_bot(chatbot.greet)

    # Interaction loop
    context, capsules = chat_1
    for capsule in capsules:
        if capsule == "quit":
            break

        try:
            # process with brain, get template for response
            say, response_template, brain_response = chatbot.respond(capsule)
            response_template = template_to_statement_capsule(response_template, chatbot)

            # arrange all response info to be saved
            capsule["last_reward"] = chatbot.replier.reward_history[-1]
            capsule["brain_state"] = chatbot.replier.state_history[-1]
            capsule["statistics_history"] = chatbot.replier.statistics_history[-1]
            capsule["reply"] = say
            chatbot.capsules_submitted.append(brain_response_to_json(capsule))
            chatbot.say_history.append(say)
            chatbot.capsules_suggested.append(response_template)

            # say, response_template, brain_response = chatbot.respond(capsule)
            print(f"\n{capsule['author']['label']}: {capsule['utterance']}")
            print(f"\nBot: {say}")
            print(f"\nResponse template: {json.dumps(response_template, indent=2)}")

        except:
            break

    # Farewell + update savefile
    print("\nBot:", chatbot.farewell)
    chatbot.close_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speaker", default="john", type=str, help="Name of the speaker (e.g. 'john')"
    )
    parser.add_argument(
        "--chat_id", default=42, type=int, help="ID for a chat",
    )
    parser.add_argument(
        "--reward", default="Total triples", type=str, help="Graph metric to use as reward",
        choices=['Average degree', 'Sparseness', 'Shortest path',
                 'Total triples', 'Average population',
                 'Ratio claims to triples',
                 'Ratio perspectives to claims', 'Ratio conflicts to claims'
                 ]
    )

    parser.add_argument(
        "--context_id", default=CONTEXT_ID, type=int, help="ID for a situated context",
    )
    parser.add_argument(
        "--context_date", default=datetime.today(), type=str, help="Date of a situated context",
    )
    parser.add_argument(
        "--place_id", default=PLACE_ID, type=int, help="ID for a physical location",
    )
    parser.add_argument(
        "--place_label", default=PLACE_NAME, type=str, help="Name for a physical location",
    )
    parser.add_argument(
        "--country", default=LOCATION["country"], type=str, help="Country of a physical location",
    )
    parser.add_argument(
        "--region", default=LOCATION["region"], type=str, help="Region of a physical location",
    )
    parser.add_argument(
        "--city", default=LOCATION["city"], type=str, help="City of a physical location",
    )

    args = parser.parse_args()
    main(args)
