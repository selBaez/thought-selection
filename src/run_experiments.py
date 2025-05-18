import argparse
import logging
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from random import shuffle, choice

from cltl.brain import logger as brain_logger
from cltl.reply_generation import logger as replier_logger
from cltl.thoughts.thought_selection import logger as thoughts_logger
from dialogue_system.rl_utils.hp_rdf_dataset import HarryPotterRDF
from dialogue_system.rl_utils.memory import ReplayMemory
from dialogue_system.rl_utils.rl_parameters import SHUFFLE_FREQUENCY, RESET_FREQUENCY, METRICS
from dialogue_system.rl_utils.state_encoder import StateEncoder
from dialogue_system.utils.global_variables import RESOURCES_PATH, RAW_USER_PATH, RAW_VANILLA_USER_PATH, LOCATION
from dialogue_system.utils.helpers import create_session_folder, search_session_folder, replace_user_name
from simulated_interaction import main as simulate_interaction_main

brain_logger.setLevel(logging.ERROR)
thoughts_logger.setLevel(logging.ERROR)
replier_logger.setLevel(logging.ERROR)


# dataset_logger.setLevel(logging.ERROR)
# memory_logger.setLevel(logging.ERROR)
# user_logger.setLevel(logging.ERROR)


def collect_and_shuffle_cumulative_graphs(experiment_id, run_id, chat_id, speaker, switch_users):
    brains = []
    for reward, setting_id in METRICS.items():
        context_id = (run_id * 1000) + (chat_id * 100) + setting_id

        if not switch_users:
            # We know the speaker is always the same, so path is known
            prev_sess = create_session_folder(experiment_id, f"run{run_id}", context_id - 100, reward, chat_id - 1,
                                              speaker)

        else:
            # We do not know the name of the previous speaker, so we look for it according to ID on the right folder
            prev_sess = search_session_folder(experiment_id, f"run{run_id}", context_id - 100, reward, chat_id - 1)

        # Find the right folder, and select the trig file with the latest cumulative state
        prev_sess = Path(f"{prev_sess}/cumulative_states").resolve()
        states = sorted(f for f in prev_sess.iterdir())

        brains.append(states[-1])

    shuffle(brains)

    return brains


def get_user_models(users_path):
    users_path = Path(users_path).resolve()
    users_pool = sorted(f for f in users_path.iterdir() if f.name != "vanilla.trig")

    return users_pool


def main(args):
    # Read dataset once to avoid loading several times
    hp_dataset = HarryPotterRDF('.')

    # Create share state encoder
    shared_encoder = StateEncoder(hp_dataset)

    # Create and pre-populate memory from prev experiments
    shared_memory = ReplayMemory()
    if (Path(RESOURCES_PATH) / "2025_experiments_small").exists():
        shared_memory.pre_populate(Path(RESOURCES_PATH) / "2025_experiments_small" , add_reward_type=True)
    if (Path(RESOURCES_PATH) / "2025_experiments_medium").exists():
        shared_memory.pre_populate(Path(RESOURCES_PATH) / "2025_experiments_medium" , add_reward_type=True)
    if (Path(RESOURCES_PATH) / "2025_experiments_fail").exists():
        shared_memory.pre_populate(Path(RESOURCES_PATH) / "2025_experiments_fail" , add_reward_type=True)
    if (Path(RESOURCES_PATH) / "2025_experiments_fail_taggedTransitions").exists():
        shared_memory.pre_populate(Path(RESOURCES_PATH) / "2025_experiments_fail_taggedTransitions" , add_reward_type=False)

    shared_memory._log.info(f"Memory size: {len(shared_memory)}")

    # Get list of users
    if args.switch_users:
        users_pool = get_user_models(args.user_model)

    for run_id in range(1, args.num_runs + 1):
        r = run_id * 1000

        for chat_id in range(1, args.num_chats + 1):
            c = chat_id * 100
            print("\n")

            # Select user model if needed
            if args.switch_users:
                user_model = str(choice(users_pool))
            else:
                user_model = args.user_model

            # Determine whether we are resetting or shuffling
            shuffle_brain = (chat_id % SHUFFLE_FREQUENCY == 0) and (chat_id != 1)
            resetting_brain = (chat_id - 1) % RESET_FREQUENCY == 0
            if shuffle_brain:
                # Collect and shuffle (cumulative graphs)
                print(f"\n################ SHUFFLING BRAINS ################")
                brains = collect_and_shuffle_cumulative_graphs(args.experiment_id, run_id, chat_id,
                                                               replace_user_name(user_model), args.switch_users)
            elif resetting_brain:
                # Check if we are resetting
                print(f"\n################ RESETTING BRAINS ################")

            # Run process
            for idx, (reward, setting_id) in enumerate(METRICS.items()):
                brain = brains[idx] if shuffle_brain else "None"
                printable_brain = brain.parents[1].name if shuffle_brain else "None"
                printable_user_model = replace_user_name(user_model)
                print(f"REWARD: {reward}, \t\tRUN: {run_id}, \t\tCHAT: {chat_id}, "
                      f"\t\tBRAIN: {printable_brain},\t\tUSER: {printable_user_model}")

                context_id = r + c + setting_id

                chat_args = Namespace(
                    # Experiment variables
                    experiment_id=args.experiment_id,
                    run_id=f"run{run_id}",
                    # Chat variables
                    chat_id=chat_id,
                    turn_limit=args.num_turns,
                    context_id=context_id,
                    context_date=datetime.today(),
                    place_id=44,
                    place_label="bookstore",
                    country=LOCATION["country"],
                    region=LOCATION["region"],
                    city=LOCATION["city"],
                    speaker=printable_user_model,
                    user_model=user_model,
                    # RL variables
                    init_brain=brain,
                    reward=reward,
                    dm_model=args.dm_model,
                    test_model=False
                )

                simulate_interaction_main(chat_args, memory=shared_memory, encoder=shared_encoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # test
    # parser.add_argument("--num_turns", default=5, type=int, help="Number of turns for this experiment")
    # parser.add_argument("--num_chats", default=3, type=int, help="Number of chats for this experiment")
    # parser.add_argument("--num_runs", default=1, type=int, help="Number of runs for this experiment")

    # real
    parser.add_argument("--num_turns", default=10, type=int, help="Number of turns for this experiment")
    parser.add_argument("--num_chats", default=8, type=int, help="Number of chats for this experiment")
    parser.add_argument("--num_runs", default=3, type=int, help="Number of runs for this experiment")

    # Parameters for experiment 1 (vanilla user)
    parser.add_argument("--experiment_id", default="e1 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    parser.add_argument("--switch_users", default=False, action='store_true', help="Switch users between chats")
    parser.add_argument("--user_model", default=RAW_VANILLA_USER_PATH, type=str, help="File or folder of user model")
    parser.add_argument("--dm_model", default="rl(full)", type=str, help="Type of selector to use",
                        choices=["rl(full)", "rl(abstract)", "rl(specific)", "random"])

    # # Parameters for experiment 2 (mixed users)
    # parser.add_argument("--experiment_id", default="e2 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    # parser.add_argument("--switch_users", default=True, action='store_true', help="Switch users between chats")
    # parser.add_argument("--user_model", default=RAW_USER_PATH, type=str, help="File or folder of user model")
    # parser.add_argument("--dm_model", default="rl(full)", type=str, help="Type of selector to use",
    #                     choices=["rl(full)", "rl(abstract)", "rl(specific)", "random"])

    # # Parameters for experiment 3 (baseline: random specific)
    # parser.add_argument("--experiment_id", default="e3 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    # parser.add_argument("--switch_users", default=False, action='store_true', help="Switch users between chats")
    # parser.add_argument("--user_model", default=RAW_VANILLA_USER_PATH, type=str, help="File or folder of user model")
    # parser.add_argument("--dm_model", default="rl(abstract)", type=str, help="Type of selector to use",
    #                     choices=["rl(full)", "rl(abstract)", "rl(specific)", "random"])

    # # Parameters for experiment 4 (baseline: random abstract)
    # parser.add_argument("--experiment_id", default="e4 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    # parser.add_argument("--switch_users", default=False, action='store_true', help="Switch users between chats")
    # parser.add_argument("--user_model", default=RAW_VANILLA_USER_PATH, type=str, help="File or folder of user model")
    # parser.add_argument("--dm_model", default="rl(specific)", type=str, help="Type of selector to use",
    #                     choices=["rl(full)", "rl(abstract)", "rl(specific)", "random"])

    # # Parameters for experiment 5 (baseline: random)
    # parser.add_argument("--experiment_id", default="e5 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    # parser.add_argument("--switch_users", default=False, action='store_true', help="Switch users between chats")
    # parser.add_argument("--user_model", default=RAW_VANILLA_USER_PATH, type=str, help="File or folder of user model")
    # parser.add_argument("--dm_model", default="random", type=str, help="Type of selector to use",
    #                     choices=["rl(full)", "rl(abstract)", "rl(specific)", "random"])

    args = parser.parse_args()
    main(args)
