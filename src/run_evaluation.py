import argparse
import logging
from argparse import Namespace
from datetime import datetime
from pathlib import Path
import glob

from cltl.brain import logger as brain_logger
from cltl.reply_generation import logger as replier_logger
from cltl.thoughts.thought_selection import logger as thoughts_logger
from dialogue_system.rl_utils.hp_rdf_dataset import HarryPotterRDF
from dialogue_system.rl_utils.memory import ReplayMemory
from dialogue_system.rl_utils.rl_parameters import METRICS_TOINCLUDE, EXPERIENCE_POOL_SIZE, EXPERIENCE_BATCH_SIZE
from dialogue_system.rl_utils.state_encoder import StateEncoder
from dialogue_system.utils.global_variables import RESOURCES_PATH, RAW_VANILLA_USER_PATH, LOCATION
from dialogue_system.utils.helpers import replace_user_name, populate_replay_memory
from simulated_interaction import main as simulate_interaction_main

brain_logger.setLevel(logging.ERROR)
thoughts_logger.setLevel(logging.ERROR)
replier_logger.setLevel(logging.ERROR)


def get_trained_models(experiment_id, run_id, reward):
    # Find folder
    testing_path = glob.glob(f"{RESOURCES_PATH}experiments/{experiment_id}*/{reward.replace(' ', '-')}/run{run_id}/")

    if testing_path:
        testing_path = Path(testing_path[0]).resolve()
        all_chats = sorted(f for f in testing_path.iterdir())
        all_chats = [chat / "thoughts.pt" for chat in all_chats]
    else:
        all_chats = []

    return all_chats


def main(args):
    # Read dataset once to avoid loading several times
    hp_dataset = HarryPotterRDF('.')

    # Create share state encoder
    shared_encoder = StateEncoder(hp_dataset)

    # Get user
    user_model = RAW_VANILLA_USER_PATH
    printable_user_model = replace_user_name(user_model)

    # Build full experiment id
    args.experiment_id = f"{args.experiment_id} ({args.num_turns}turns_{len(trained_models)}chats_{args.num_runs}runs)"

    for run_id in range(1, args.num_runs + 1):
        # Create and pre-populate experience_memory from prev experiments
        run_memory = populate_replay_memory()  # Independent per run but shared across chats
        metric_memories = {reward: ReplayMemory(capacity=EXPERIENCE_POOL_SIZE, batch_size=EXPERIENCE_BATCH_SIZE)
                           for reward in METRICS_TOINCLUDE.keys()}

        # Run process
        for idx, (reward, setting_id) in enumerate(METRICS_TOINCLUDE.items()):
            # Get trained models
            trained_models = get_trained_models(args.testing_id, run_id, reward)

            for checkpoint_id, trained_model in enumerate(trained_models):
                # Determine id
                chat_id = checkpoint_id + 1
                r = run_id * 1000
                c = chat_id * 100
                context_id = r + c + setting_id

                print(f"REWARD: {reward}, \t\tRUN: {run_id}, \t\tCHAT: {chat_id}, "
                      f"\t\tMODEL: {trained_model},\t\tUSER: {printable_user_model}")

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
                    init_brain="None",
                    reward=reward,
                    dm_model=args.dm_model,
                    test_model=trained_model
                )

                simulate_interaction_main(chat_args, replay_memory=run_memory, shared_memory=metric_memories[reward],
                                          encoder=shared_encoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # # test
    # parser.add_argument("--num_turns", default=2, type=int, help="Number of turns for this experiment")
    # parser.add_argument("--num_runs", default=1, type=int, help="Number of runs for this experiment")

    # real
    parser.add_argument("--num_turns", default=10, type=int, help="Number of turns for this experiment")
    parser.add_argument("--num_runs", default=3, type=int, help="Number of runs for this experiment")

    parser.add_argument("--experiment_id", default="t1", type=str, help="ID for a test")
    parser.add_argument("--testing_id", default="e1", type=str, help="ID for an experiment")
    parser.add_argument("--dm_model", default="rl(full)", type=str, help="Type of selector to use",
                        choices=["rl(full)", "rl(abstract)", "rl(specific)", "rl(random)", "random"])

    # parser.add_argument("--experiment_id", default="t2 (10turns_3runs_8checkpoints)", type=str, help="ID for a test")
    # parser.add_argument("--testing_id", default="e2 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    # parser.add_argument("--dm_model", default="rl(full)", type=str, help="Type of selector to use",
    #                     choices=["rl(full)", "rl(abstract)", "rl(specific)", "rl(random)", "random"])

    # parser.add_argument("--experiment_id", default="t3 (10turns_3runs_8checkpoints)", type=str, help="ID for a test")
    # parser.add_argument("--testing_id", default="e3 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    # parser.add_argument("--dm_model", default="rl(abstract)", type=str, help="Type of selector to use",
    #                     choices=["rl(full)", "rl(abstract)", "rl(specific)", "rl(random)", "random"])

    # parser.add_argument("--experiment_id", default="t4 (10turns_3runs_8checkpoints)", type=str, help="ID for a test")
    # parser.add_argument("--testing_id", default="e4 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    # parser.add_argument("--dm_model", default="rl(specific)", type=str, help="Type of selector to use",
    #                     choices=["rl(full)", "rl(abstract)", "rl(specific)", "rl(random)", "random"])
    #
    # parser.add_argument("--experiment_id", default="t5 (10turns_3runs_8checkpoints)", type=str, help="ID for a test")
    # parser.add_argument("--testing_id", default="e5 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    # parser.add_argument("--dm_model", default="rl(random)", type=str, help="Type of selector to use",
    #                     choices=["rl(full)", "rl(abstract)", "rl(specific)", "rl(random)", "random"])

    args = parser.parse_args()
    main(args)
