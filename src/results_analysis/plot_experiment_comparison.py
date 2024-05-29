import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.dialogue_system.utils.global_variables import RESOURCES_PATH, USER_PATH
from src.dialogue_system.utils.rl_parameters import ACTION_THOUGHTS

RESOURCES_PATH = Path(RESOURCES_PATH).resolve()
USER_PATH = Path(USER_PATH).resolve()

import sys

print(sys.path)


def plot(episode_data, plots_folder):
    episode_data["thought_types"] = episode_data["actions"].apply(lambda a: ACTION_THOUGHTS.get(a, None))

    # Histogram: count per thought type and per entity type
    fig = plt.figure(figsize=(10, 5), tight_layout=True)
    fig = sns.histplot(data=episode_data, x="thought_types", hue="condition")
    plt.ylabel("Count")
    plt.xlabel("Thought types")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(f"{plots_folder}/action_type_count.png", dpi=300)
    plt.show()

    # Point plot: Cumulative reward
    fig = plt.figure(figsize=(10, 5), tight_layout=True)
    # fig = sns.histplot(x=range(len(episode_data)), y=episode_data['rewards'], cumulative=True)
    # fig = sns.kdeplot(data=episode_data, x="turn", y="rewards", hue="condition")
    fig = sns.ecdfplot(data=episode_data, y="rewards", stat="count", hue="condition")
    plt.ylabel("Cumulative reward")
    plt.xlabel("Turn")
    plt.savefig(f"{plots_folder}/cumulative_reward.png", dpi=300)
    plt.show()

    # State fluctuation
    fig = plt.figure(figsize=(10, 5), tight_layout=True)
    fig = sns.relplot(data=episode_data, x="turn", y="states", hue="condition", kind="line")
    plt.ylim(0)
    plt.ylabel("Metric value")
    plt.xlabel("Turn")
    # plt.xticks(rotation=45, ha="right")

    plt.savefig(f"{plots_folder}/metric_over_turns.png", dpi=300)
    plt.show()

    print("DONE")


experiments = [f for f in RESOURCES_PATH.iterdir() if f.is_dir() and f != USER_PATH]

experiments_data = []
for ex in experiments:
    episode_file = ex / "history.json"

    with open(episode_file) as f:
        episode_dict = json.load(f)

    episode_data = pd.DataFrame.from_dict(episode_dict)
    episode_data["turn"] = range(len(episode_data))
    episode_data["condition"] = ex.name.split("_")[0]

    experiments_data.append(episode_data)

data = pd.concat(experiments_data, ignore_index=True)

plot(data, "./../resources/")
