import argparse
import subprocess
import sys
from pathlib import Path
from random import shuffle

from dialogue_system.utils.global_variables import RAW_VANILLA_USER_PATH
from dialogue_system.utils.helpers import create_session_folder
from dialogue_system.utils.rl_parameters import SHUFFLE_FREQUENCY, RESET_FREQUENCY

print(f"\n\n{sys.path}\n\n")

METRICS = {'Sparseness': 11, 'Average degree': 12, 'Shortest path': 13, 'Total triples': 14,
           'Average population': 21,
           'Ratio claims to triples': 31, 'Ratio perspectives to claims': 32, 'Ratio conflicts to claims': 33}


def collect_cumulative_graphs(run_id, chat_id):
    brains = []
    for reward, setting_id in METRICS.items():
        context_id = (run_id * 1000) + (chat_id * 100) + setting_id

        prev_sess = create_session_folder(args.experiment_id, f"run{run_id}", context_id - 100, reward, chat_id - 1,
                                          "vanilla")
        prev_sess = Path(f"{prev_sess}/cumulative_states").resolve()

        states = sorted(f for f in prev_sess.iterdir())

        brains.append(states[-1])

    return brains


def main(args):
    for run_id in range(1, args.num_runs + 1):
        r = run_id * 1000

        for chat_id in range(1, args.num_chats + 1):
            c = chat_id * 100
            print("\n")

            # Collect and shuffle (cumulative graphs)
            shuffle_brain = (chat_id % SHUFFLE_FREQUENCY == 0) and (chat_id != 1)
            if shuffle_brain:
                brains = collect_cumulative_graphs(run_id, chat_id)
                shuffle(brains)

            # Just check if we are resetting
            resetting_brain = (chat_id - 1) % RESET_FREQUENCY == 0

            if shuffle_brain:
                print(f"\n################ SHUFFLING BRAINS ################")
            elif resetting_brain:
                print(f"\n################ RESETTING BRAINS ################")

            # Run process
            for idx, (reward, setting_id) in enumerate(METRICS.items()):
                brain = brains[idx] if shuffle_brain else "None"
                printable_brain = brain.parents[1].name if shuffle_brain else "None"
                print(f"REWARD: {reward}, \t\tRUN: {run_id}, \t\tCHAT: {chat_id}, \t\tBRAIN: {printable_brain}")

                context_id = r + c + setting_id
                subprocess.run([
                    "python", "-u", "simulated_interaction.py",
                    "--experiment_id", f"{args.experiment_id}",
                    "--turn_limit", f"{args.num_turns}", "--chat_id", f"{chat_id}", "--run_id", f"run{run_id}",
                    "--reward", reward,
                    "--init_brain", brain,
                    "--user_model", "./../resources/users/raw/vanilla.trig", "--speaker", "vanilla",
                    "--context_id", f"{context_id}", "--place_id", "44", "--place_label", "bookstore"
                ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", default="e1 (25turns_24chats_3runs)", type=str, help="ID for an experiment")
    parser.add_argument("--user_model", default=RAW_VANILLA_USER_PATH, type=str,
                        help="Filepath of the user model (e.g. 'vanilla.trig')")
    parser.add_argument("--num_turns", default=25, type=int, help="Number of turns for this experiment")
    parser.add_argument("--num_chats", default=24, type=int, help="Number of chats for this experiment")
    parser.add_argument("--num_runs", default=3, type=int, help="Number of runs for this experiment")

    args = parser.parse_args()
    main(args)
