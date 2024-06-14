import argparse
from datetime import datetime

from cltl.brain.utils.helper_functions import brain_response_to_json
from dialogue_system.chatbot import Chatbot
from dialogue_system.utils.global_variables import CONTEXT_ID, PLACE_ID, PLACE_NAME, LOCATION, RAW_VANILLA_USER_PATH
from user_model.user import User


def create_context_capsule(args):
    return {"context_id": args.context_id,
            "date": datetime.strftime(args.context_date, "%Y-%m-%d"),
            "place": args.place_label,
            "place_id": args.place_id,
            "country": args.country,
            "region": args.region,
            "city": args.city}


def print_bot(bot_utterance):
    print("Bot:", bot_utterance)


def print_user_model(capsule):
    print(f"\nTURN:{capsule['turn']}\n{capsule['author']['label']}: {capsule['utterance']}")


def print_template(response_template):
    print(f"\tResponse template: {response_template['subject']['label']} "
          f"{response_template['predicate']['label']} {response_template['object']['label']}")


def main(args):
    """Runs the main interaction loop of the chatbot."""
    # Sets up user model
    user_model = User(args.user_model)

    # Create dialogue_system
    chatbot = Chatbot()
    chatbot.begin_session(args.experiment_id, args.run_id,
                          args.context_id, args.chat_id, args.speaker, args.reward, args.init_brain, args.test_model)

    # Situate chat
    capsule_for_context = create_context_capsule(args)
    chatbot.situate_chat(capsule_for_context)

    # Interaction loop
    capsule = user_model.init_capsule(args, chatbot)
    for index in range(args.turn_limit):
        # process with brain, get template for response
        say, response_template, brain_response = chatbot.respond(capsule)

        print_user_model(capsule)
        print_bot(say)
        print_template(response_template)

        # use response template to query HP dataset
        response_template = brain_response_to_json(response_template)
        capsule = user_model.query_database(response_template)

    # Farewell + update savefile
    print("\nBot:", chatbot.farewell)
    chatbot.close_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", default="e1", type=str, help="ID for an experiment")
    parser.add_argument("--run_id", default="run1", type=str, help="ID for a run")
    parser.add_argument("--chat_id", default=42, type=int, help="ID for a chat")
    parser.add_argument("--init_brain", default="None", type=str, help="trig file to read if swapping brains")
    parser.add_argument("--user_model", default=RAW_VANILLA_USER_PATH, type=str,
                        help="Filepath of the user model (e.g. 'vanilla.trig')")
    parser.add_argument("--speaker", default="john", type=str, help="Name of the speaker (e.g. 'john')")
    parser.add_argument("--reward", default="Total triples", type=str, help="Graph metric to use as reward",
                        choices=['Average degree', 'Sparseness', 'Shortest path',
                                 'Total triples', 'Average population',
                                 'Ratio claims to triples',
                                 'Ratio perspectives to claims', 'Ratio conflicts to claims'
                                 ])
    parser.add_argument("--context_id", default=CONTEXT_ID, type=int, help="ID for a situated context")
    parser.add_argument("--context_date", default=datetime.today(), type=str, help="Date of a situated context")
    parser.add_argument("--place_id", default=PLACE_ID, type=int, help="ID for a physical location")
    parser.add_argument("--place_label", default=PLACE_NAME, type=str, help="Name for a physical location")
    parser.add_argument("--country", default=LOCATION["country"], type=str, help="Country of a physical location")
    parser.add_argument("--region", default=LOCATION["region"], type=str, help="Region of a physical location")
    parser.add_argument("--city", default=LOCATION["city"], type=str, help="City of a physical location")
    parser.add_argument("--turn_limit", default=15, type=int, help="Number of turns for this interaction")

    parser.add_argument("--test_model", default=None, type=str, help="Use trained network and freeze learning")

    args = parser.parse_args()
    main(args)
