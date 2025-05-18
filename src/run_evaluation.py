import argparse
import logging
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from cltl.brain import logger as brain_logger
from cltl.reply_generation import logger as replier_logger
from cltl.thoughts.thought_selection import logger as thoughts_logger
from dialogue_system.rl_utils.hp_rdf_dataset import HarryPotterRDF
from dialogue_system.rl_utils.memory import ReplayMemory
from dialogue_system.rl_utils.rl_parameters import METRICS
from dialogue_system.rl_utils.state_encoder import StateEncoder
from dialogue_system.utils.global_variables import RESOURCES_PATH, RAW_VANILLA_USER_PATH, LOCATION
from dialogue_system.utils.helpers import replace_user_name
from simulated_interaction import main as simulate_interaction_main

brain_logger.setLevel(logging.ERROR)
thoughts_logger.setLevel(logging.ERROR)
replier_logger.setLevel(logging.ERROR)


def get_trained_models(experiment_id, run_id, reward):
    testing_path = Path(
        f"{RESOURCES_PATH}experiments/{experiment_id}/{reward.replace(' ', '-')}/run{run_id}/").resolve()
    all_chats = sorted(f for f in testing_path.iterdir())

    all_chats = [chat / "thoughts.pt" for chat in all_chats]

    return all_chats


def main(args):
    # Read dataset once to avoid loading several times
    hp_dataset = HarryPotterRDF('.')

    # Create share state encoder
    shared_encoder = StateEncoder(hp_dataset)

    # Create and pre-populate memory from prev experiments
    shared_memory = ReplayMemory()
    if (Path(RESOURCES_PATH) / "2025_experiments_small").exists():
        shared_memory.pre_populate(Path(RESOURCES_PATH) / "2025_experiments_small", add_reward_type=True)
    if (Path(RESOURCES_PATH) / "2025_experiments_medium").exists():
        shared_memory.pre_populate(Path(RESOURCES_PATH) / "2025_experiments_medium", add_reward_type=True)
    if (Path(RESOURCES_PATH) / "2025_experiments_fail").exists():
        shared_memory.pre_populate(Path(RESOURCES_PATH) / "2025_experiments_fail", add_reward_type=True)
    if (Path(RESOURCES_PATH) / "2025_experiments_fail_taggedTransitions").exists():
        shared_memory.pre_populate(Path(RESOURCES_PATH) / "2025_experiments_fail_taggedTransitions",
                                   add_reward_type=False)

    shared_memory._log.info(f"Memory size: {len(shared_memory)}")

    printable_user_model = replace_user_name(args.user_model)

    for run_id in range(1, args.num_runs + 1):
        r = run_id * 1000

        # Run process
        for idx, (reward, setting_id) in enumerate(METRICS.items()):
            trained_models = get_trained_models(args.testing_id, run_id, reward)

            for checkpoint_id, trained_model in enumerate(trained_models):
                chat_id = checkpoint_id + 1
                c = chat_id * 100
                context_id = r + c + setting_id

                print(f"REWARD: {reward}, \t\tRUN: {run_id}, \t\tCHAT: {chat_id}, "
                      f"\t\tMODEL: {trained_model},\t\tUSER: {args.user_model}")

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
                    user_model=args.user_model,
                    # RL variables
                    init_brain=None,
                    reward=reward,
                    dm_model=args.dm_model,
                    test_model=trained_model
                )

                simulate_interaction_main(chat_args, memory=shared_memory, encoder=shared_encoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_turns", default=10, type=int, help="Number of turns for this experiment")
    parser.add_argument("--num_runs", default=3, type=int, help="Number of runs for this experiment")

    parser.add_argument("--experiment_id", default="t1 (10turns_3runs_8checkpoints)", type=str, help="ID for a test")
    parser.add_argument("--testing_id", default="e1 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    parser.add_argument("--user_model", default=RAW_VANILLA_USER_PATH, type=str, help="File or folder of user model")
    parser.add_argument("--dm_model", default="rl(full)", type=str, help="Type of selector to use",
                        choices=["rl(full)", "rl(abstract)", "rl(specific)", "random"])

    # parser.add_argument("--experiment_id", default="t2 (10turns_3runs_8checkpoints)", type=str, help="ID for a test")
    # parser.add_argument("--testing_id", default="e2 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    # parser.add_argument("--user_model", default=RAW_VANILLA_USER_PATH, type=str, help="File or folder of user model")
    # parser.add_argument("--dm_model", default="rl(full)", type=str, help="Type of selector to use",
    #                     choices=["rl(full)", "rl(abstract)", "rl(specific)", "random"])

    # parser.add_argument("--experiment_id", default="t3 (10turns_3runs_8checkpoints)", type=str, help="ID for a test")
    # parser.add_argument("--testing_id", default="e3 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    # parser.add_argument("--user_model", default=RAW_VANILLA_USER_PATH, type=str, help="File or folder of user model")
    # parser.add_argument("--dm_model", default="rl(abstract)", type=str, help="Type of selector to use",
    #                     choices=["rl(full)", "rl(abstract)", "rl(specific)", "random"])

    # parser.add_argument("--experiment_id", default="t4 (10turns_3runs_8checkpoints)", type=str, help="ID for a test")
    # parser.add_argument("--testing_id", default="e4 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    # parser.add_argument("--user_model", default=RAW_VANILLA_USER_PATH, type=str, help="File or folder of user model")
    # parser.add_argument("--dm_model", default="rl(specific)", type=str, help="Type of selector to use",
    #                     choices=["rl(full)", "rl(abstract)", "rl(specific)", "random"])
    #
    # parser.add_argument("--experiment_id", default="t5 (10turns_3runs_8checkpoints)", type=str, help="ID for a test")
    # parser.add_argument("--testing_id", default="e5 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    # parser.add_argument("--user_model", default=RAW_VANILLA_USER_PATH, type=str, help="File or folder of user model")
    # parser.add_argument("--dm_model", default="random", type=str, help="Type of selector to use",
    #                     choices=["rl(full)", "rl(abstract)", "rl(specific)", "random"])

    args = parser.parse_args()
    main(args)
