from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from dialogue_system.rl_utils.rl_parameters import ACTION_THOUGHTS, ACTION_TYPES


def separate_thought_elements(selected_thoughts):
    entity_types = []
    thought_types = []
    for s in selected_thoughts.values:
        if s:
            [tt, ets] = s.split(":")
            et = ets.split("-")

            thought_types.append(tt)
            entity_types.extend(et)

    entity_types_counter = Counter(entity_types)
    thought_types_counter = Counter(thought_types)

    # Add missing element with count 0
    for k, v in ACTION_THOUGHTS.items():
        if v not in thought_types_counter.keys():
            thought_types_counter.setdefault(v, 0)

    # Add missing element with count 0
    for k, v in ACTION_TYPES.items():
        if v not in entity_types_counter.keys():
            entity_types_counter.setdefault(v, 0)

    return entity_types_counter, thought_types_counter


def plot_action_counts(entity_types_counter, thought_types_counter, plots_folder):
    fig = plt.figure(figsize=(10, 5), tight_layout=True)

    plt.subplot(1, 2, 1)
    fig = plt.bar(thought_types_counter.keys(), thought_types_counter.values())
    plt.ylabel("Count")
    plt.xlabel("Thought types")
    plt.xticks(rotation=45, ha="right")

    plt.subplot(1, 2, 2)
    fig = plt.bar(entity_types_counter.keys(), entity_types_counter.values())
    plt.ylabel("Count")
    plt.xlabel("Entity types")
    plt.xticks(rotation=45, ha="right")

    plt.savefig(f"{plots_folder}/action_type_count.png", dpi=300)
    # plt.show()


def plot_cumulative_reward(reward_history, plots_folder):
    fig = plt.figure(figsize=(10, 5), tight_layout=True)

    # fig = sns.histplot(x=range(len(episode_data)), y=episode_data['rewards'], cumulative=True)
    fig = sns.ecdfplot(y=reward_history, stat="count")
    plt.ylabel("Cumulative reward")
    plt.xlabel("Turn")
    # plt.xticks(rotation=45, ha="right")

    plt.savefig(f"{plots_folder}/cumulative_reward.png", dpi=300)
    # plt.show()


def plot_metrics_over_time(episode_data, plots_folder):
    fig = plt.figure(figsize=(10, 5), tight_layout=True)

    fig = sns.relplot(x=range(len(episode_data)), y=episode_data["states"], kind="line")
    plt.ylim(0)
    plt.ylabel("Metric value")
    plt.xlabel("Turn")
    # plt.xticks(rotation=45, ha="right")

    plt.savefig(f"{plots_folder}/metric_over_turns.png", dpi=300)
    # plt.show()
