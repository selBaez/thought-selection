import argparse
from cltl.brain.utils.base_cases import chat_1
from cltl.brain.utils.helper_functions import brain_response_to_json
from datetime import datetime

from src.dialogue_system.chatbot import Chatbot
from src.dialogue_system.utils.global_variables import CONTEXT_ID, PLACE_ID, PLACE_NAME, LOCATION
from src.user_model.user import User, init_capsule


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
    # Sets up user model
    user_model = User()

    # Create dialogue_system
    chatbot = Chatbot()
    chatbot.begin_session(args.chat_id, args.speaker, args.reward)

    # Situate chat
    capsule_for_context = create_context_capsule(args)
    chatbot.situate_chat(capsule_for_context)

    # Interaction loop
    context, capsules = chat_1 # TODO replace with query to user model
    capsule = init_capsule()
    for index, _ in enumerate(capsules):  # TODO fix number of turns
        # process with brain, get template for response
        say, response_template, brain_response = chatbot.respond(capsule)

        print(f"\n{capsule['author']['label']}: {capsule['utterance']}")
        print(f"\nBot: {say}")
        # print(f"\nResponse template: {json.dumps(brain_response_to_json(response_template), indent=2)}")

        # use response template to query HP dataset
        response_template = brain_response_to_json(response_template)
        capsule = user_model.query_database(response_template)

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
