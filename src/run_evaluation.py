import argparse
import subprocess
import sys
from pathlib import Path

from dialogue_system.utils.global_variables import RAW_VANILLA_USER_PATH, RESOURCES_PATH
from dialogue_system.rl_utils.rl_parameters import METRICS

print(f"\n\n{sys.path}\n\n")


def get_trained_models(experiment_id, run_id, reward):
    testing_path = Path(
        f"{RESOURCES_PATH}experiments/{experiment_id}/{reward.replace(' ', '-')}/run{run_id}/").resolve()
    all_chats = sorted(f for f in testing_path.iterdir())

    all_chats = [chat / "thoughts.pt" for chat in all_chats]

    return all_chats


def main(args):
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

                subprocess.run([
                    "python", "-u", "simulated_interaction.py",
                    "--experiment_id", f"{args.experiment_id}",
                    "--turn_limit", f"{args.num_turns}", "--chat_id", f"{chat_id}", "--run_id", f"run{run_id}",
                    "--reward", reward,
                    "--init_brain", "None",
                    "--test_model", trained_model,
                    "--user_model", f"{args.user_model}", "--speaker", "vanilla",
                    "--context_id", f"{context_id}", "--place_id", "44", "--place_label", "bookstore"
                ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", default="t1 (10turns_3runs_8checkpoints)", type=str, help="ID for a test")
    parser.add_argument("--testing_id", default="e1 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    parser.add_argument("--num_turns", default=10, type=int, help="Number of turns for this experiment")
    parser.add_argument("--num_runs", default=3, type=int, help="Number of runs for this experiment")
    parser.add_argument("--user_model", default=RAW_VANILLA_USER_PATH, type=str,
                        help="Filepath of the user model (e.g. 'vanilla.trig')")

    args = parser.parse_args()
    main(args)
