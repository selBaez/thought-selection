import argparse
from datetime import datetime

from cltl.brain.utils.helper_functions import brain_response_to_json
from src.dialogue_system.chatbot import Chatbot
from src.dialogue_system.utils.global_variables import CONTEXT_ID, PLACE_ID, PLACE_NAME, LOCATION, RAW_VANILLA_USER_PATH
from src.user_model.user import User, init_capsule


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
    user_model = User(RAW_VANILLA_USER_PATH)

    # Create dialogue_system
    chatbot = Chatbot()
    chatbot.begin_session(args.chat_id, args.speaker, args.reward)

    # Situate chat
    capsule_for_context = create_context_capsule(args)
    chatbot.situate_chat(capsule_for_context)

    # Interaction loop
    capsule = init_capsule(args, chatbot)
    for index in range(args.turn_limit):
        # process with brain, get template for response
        say, response_template, brain_response = chatbot.respond(capsule)

        print_user_model(capsule)
        print_bot(say)
        print_template(response_template)

        # use response template to query HP dataset
        response_template = brain_response_to_json(response_template)
        capsule = user_model.query_database(response_template)

        # every 100 turns log the policy
        if index % 100 == 0:
            tempfile = chatbot.scenario_folder / f"thoughts_{index}.pt"
            chatbot.selector.save(tempfile)
            # chatbot.selector.plot(filename=chatbot.scenario_folder / f"results_{index}.png")
            print(f"\t\t\t\t\t\t\t\tSaving policy at {index + 1} turns")

    # Farewell + update savefile
    print("\nBot:", chatbot.farewell)
    chatbot.close_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", default="john", type=str, help="Name of the speaker (e.g. 'john')")
    parser.add_argument("--chat_id", default=42, type=int, help="ID for a chat")
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

    args = parser.parse_args()
    main(args)
