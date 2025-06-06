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
from dialogue_system.rl_utils.rl_parameters import SHUFFLE_FREQUENCY, RESET_FREQUENCY, METRICS_TOINCLUDE, \
    EXPERIENCE_POOL_SIZE, EXPERIENCE_BATCH_SIZE
from dialogue_system.rl_utils.state_encoder import StateEncoder
from dialogue_system.utils.global_variables import RAW_USER_PATH, RAW_VANILLA_USER_PATH, LOCATION
from dialogue_system.utils.helpers import create_session_folder, search_session_folder, replace_user_name, \
    populate_replay_memory
from simulated_interaction import main as simulate_interaction_main

brain_logger.setLevel(logging.ERROR)
thoughts_logger.setLevel(logging.ERROR)
replier_logger.setLevel(logging.ERROR)


# dataset_logger.setLevel(logging.ERROR)
# memory_logger.setLevel(logging.ERROR)
# user_logger.setLevel(logging.ERROR)


def collect_and_shuffle_cumulative_graphs(experiment_id, run_id, chat_id, speaker, switch_users):
    brains = []
    for reward, setting_id in METRICS_TOINCLUDE.items():
        context_id = (run_id * 1000) + (chat_id * 100) + setting_id

        if not switch_users:
            # We know the speaker is always the same, so path is known
            speaker = replace_user_name(speaker)
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


def assign_brains(chat_id, run_id, user_model):
    # Determine whether we are resetting or shuffling
    shuffle_brain = (chat_id % SHUFFLE_FREQUENCY == 0) and (chat_id != 1)
    resetting_brain = (chat_id - 1) % RESET_FREQUENCY == 0

    brains = []
    if shuffle_brain:
        # Collect and shuffle (cumulative graphs)
        print(f"\n################ SHUFFLING BRAINS ################")
        brains = collect_and_shuffle_cumulative_graphs(args.experiment_id, run_id, chat_id, user_model,
                                                       args.switch_users)
    elif resetting_brain:
        # Check if we are resetting
        print(f"\n################ RESETTING BRAINS ################")
        brains = ["None" for reward in METRICS_TOINCLUDE.keys()]

    return brains


def get_user_models(users_path):
    users_path = Path(users_path).resolve()
    users_pool = sorted(f for f in users_path.iterdir() if f.name != "vanilla.trig")

    return users_pool


def main(args):
    # Build full experiment id
    args.experiment_id = f"{args.experiment_id} ({args.num_turns}turns_{args.num_chats}chats_{args.num_runs}runs)"

    # Read dataset once to avoid loading several times
    hp_dataset = HarryPotterRDF('.')

    # Create share state encoder
    shared_encoder = StateEncoder(hp_dataset)

    # Get list of users
    if args.switch_users:
        users_pool = get_user_models(RAW_USER_PATH)
    else:
        users_pool = [Path(RAW_VANILLA_USER_PATH).resolve()]

    for run_id in range(1, args.num_runs + 1):
        # Initialize memories
        run_memory = populate_replay_memory()  # Independent per run but shared across chats
        metric_memories = {reward: ReplayMemory(capacity=EXPERIENCE_POOL_SIZE, batch_size=EXPERIENCE_BATCH_SIZE)
                           for reward in METRICS_TOINCLUDE.keys()}

        for chat_id in range(1, args.num_chats + 1):
            # Select user model
            user_model = str(choice(users_pool))

            for idx, (reward, setting_id) in enumerate(METRICS_TOINCLUDE.items()):
                # Determine id
                r = run_id * 1000
                c = chat_id * 100
                context_id = r + c + setting_id

                # Assign brains if we reset or reshuffle
                brains = assign_brains(chat_id, run_id, user_model)

                # Run process
                brain = brains[idx]
                printable_brain = brain.parents[1].name if brain != "None" else "None"
                printable_user_model = replace_user_name(user_model)

                print(f"REWARD: {reward}, \t\tRUN: {run_id}, \t\tCHAT: {chat_id}, "
                      f"\t\tBRAIN: {printable_brain},\t\tUSER: {printable_user_model}")

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

                simulate_interaction_main(chat_args, replay_memory=run_memory, shared_memory=metric_memories[reward],
                                          encoder=shared_encoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # # test
    # parser.add_argument("--num_turns", default=2, type=int, help="Number of turns for this experiment")
    # parser.add_argument("--num_chats", default=3, type=int, help="Number of chats for this experiment")
    # parser.add_argument("--num_runs", default=1, type=int, help="Number of runs for this experiment")

    # real
    parser.add_argument("--num_turns", default=10, type=int, help="Number of turns for this experiment")
    parser.add_argument("--num_chats", default=5, type=int, help="Number of chats for this experiment")
    parser.add_argument("--num_runs", default=3, type=int, help="Number of runs for this experiment")

    # # Parameters for experiment 1 (vanilla user)
    # parser.add_argument("--experiment_id", default="e1", type=str, help="ID for an experiment")
    # parser.add_argument("--switch_users", default=False, action='store_true', help="Switch users between chats")
    # parser.add_argument("--dm_model", default="rl(full)", type=str, help="Type of selector to use",
    #                     choices=["rl(full)", "rl(abstract)", "rl(specific)", "rl(random)", "random"])

    # # Parameters for experiment 2 (mixed users)
    # parser.add_argument("--experiment_id", default="e2", type=str, help="ID for an experiment")
    # parser.add_argument("--switch_users", default=True, action='store_true', help="Switch users between chats")
    # parser.add_argument("--dm_model", default="rl(full)", type=str, help="Type of selector to use",
    #                     choices=["rl(full)", "rl(abstract)", "rl(specific)", "rl(random)", "random"])

    # # Parameters for experiment 3 (baseline: random specific)
    # parser.add_argument("--experiment_id", default="e3", type=str, help="ID for an experiment")
    # parser.add_argument("--switch_users", default=False, action='store_true', help="Switch users between chats")
    # parser.add_argument("--dm_model", default="rl(abstract)", type=str, help="Type of selector to use",
    #                     choices=["rl(full)", "rl(abstract)", "rl(specific)", "rl(random)", "random"])

    # Parameters for experiment 4 (baseline: random abstract)
    parser.add_argument("--experiment_id", default="e4", type=str, help="ID for an experiment")
    parser.add_argument("--switch_users", default=False, action='store_true', help="Switch users between chats")
    parser.add_argument("--dm_model", default="rl(specific)", type=str, help="Type of selector to use",
                        choices=["rl(full)", "rl(abstract)", "rl(specific)", "rl(random)", "random"])

    # # Parameters for experiment 5 (baseline: random)
    # parser.add_argument("--experiment_id", default="e5", type=str, help="ID for an experiment")
    # parser.add_argument("--switch_users", default=False, action='store_true', help="Switch users between chats")
    # parser.add_argument("--dm_model", default="rl(random)", type=str, help="Type of selector to use",
    #                     choices=["rl(full)", "rl(abstract)", "rl(specific)", "rl(random)", "random"])

    args = parser.parse_args()

    main(args)
