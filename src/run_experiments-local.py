import argparse
import subprocess
from pathlib import Path
from random import shuffle, choice

from dialogue_system.utils.global_variables import RAW_VANILLA_USER_PATH, RAW_USER_PATH
from dialogue_system.utils.helpers import create_session_folder
from dialogue_system.utils.rl_parameters import SHUFFLE_FREQUENCY, RESET_FREQUENCY, METRICS


def collect_and_shuffle_cumulative_graphs(run_id, chat_id):
    brains = []
    for reward, setting_id in METRICS.items():
        context_id = (run_id * 1000) + (chat_id * 100) + setting_id

        prev_sess = create_session_folder(args.experiment_id, f"run{run_id}", context_id - 100, reward, chat_id - 1,
                                          "vanilla")
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
    # Get list of users
    if args.switch_users:
        users_pool = get_user_models(args.users_path)

    for run_id in range(1, args.num_runs + 1):
        r = run_id * 1000

        for chat_id in range(1, args.num_chats + 1):
            c = chat_id * 100
            print("\n")

            # Collect and shuffle (cumulative graphs)
            shuffle_brain = (chat_id % SHUFFLE_FREQUENCY == 0) and (chat_id != 1)
            if shuffle_brain:
                brains = collect_and_shuffle_cumulative_graphs(run_id, chat_id)

            # Check if we are resetting
            resetting_brain = (chat_id - 1) % RESET_FREQUENCY == 0

            # Select user model if needed
            if args.switch_users:
                user_model = str(choice(users_pool))
            else:
                user_model = args.user_model

            if shuffle_brain:
                print(f"\n################ SHUFFLING BRAINS ################")
            elif resetting_brain:
                print(f"\n################ RESETTING BRAINS ################")

            # Run process
            for idx, (reward, setting_id) in enumerate(METRICS.items()):
                brain = brains[idx] if shuffle_brain else "None"
                printable_brain = brain.parents[1].name if shuffle_brain else "None"
                printable_user_model = user_model.split("/")[-1]
                print(f"REWARD: {reward}, \t\tRUN: {run_id}, \t\tCHAT: {chat_id}, "
                      f"\t\tBRAIN: {printable_brain},\t\tUSER: {printable_user_model}")

                context_id = r + c + setting_id
                subprocess.run([
                    "python", "-u", "simulated_interaction.py",
                    "--experiment_id", f"{args.experiment_id}",
                    "--turn_limit", f"{args.num_turns}", "--chat_id", f"{chat_id}", "--run_id", f"run{run_id}",
                    "--reward", reward,
                    "--init_brain", brain,
                    "--user_model", f"{user_model}", "--speaker", "vanilla",
                    "--context_id", f"{context_id}", "--place_id", "44", "--place_label", "bookstore"
                ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", default="e2 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    parser.add_argument("--num_turns", default=10, type=int, help="Number of turns for this experiment")
    parser.add_argument("--num_chats", default=8, type=int, help="Number of chats for this experiment")
    parser.add_argument("--num_runs", default=3, type=int, help="Number of runs for this experiment")
    parser.add_argument("--user_model", default=RAW_VANILLA_USER_PATH, type=str,
                        help="Filepath of the user model (e.g. 'vanilla.trig')")
    parser.add_argument("--switch_users", default=True, action='store_true',
                        help="Whether to switch users between chats")
    parser.add_argument("--users_path", default=RAW_USER_PATH, type=str, help="Directory where user models are")

    args = parser.parse_args()
    main(args)
