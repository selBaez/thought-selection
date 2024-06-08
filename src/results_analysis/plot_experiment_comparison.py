import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from dialogue_system.d2q_selector import DQN, StateEncoder
from dialogue_system.utils.encode_state import HarryPotterRDF
from dialogue_system.utils.global_variables import RESOURCES_PATH
from dialogue_system.utils.rl_parameters import DEVICE, STATE_EMBEDDING_SIZE, N_ACTIONS_THOUGHTS, N_ACTION_TYPES, \
    ACTION_THOUGHTS, ACTION_TYPES

EXPERIMENTS_PATH = Path(f"{RESOURCES_PATH}/experiments").resolve()
PLOTS_PATH = Path(f"{RESOURCES_PATH}/plots").resolve()


def load_trained_model(filename):
    policy_net = DQN(STATE_EMBEDDING_SIZE, N_ACTIONS_THOUGHTS, N_ACTION_TYPES).to(DEVICE)

    model_dict = policy_net.state_dict()
    modelCheckpoint = torch.load(filename)

    new_dict = {k: v for k, v in modelCheckpoint.items() if k in model_dict.keys()}
    model_dict.update(new_dict)

    policy_net.load_state_dict(model_dict)

    return policy_net


def get_qvalues(policy_net, state_encoder, state_file=None):
    if state_file:
        encoded_state = state_encoder.encode(state_file)
    else:
        encoded_state = torch.tensor(np.zeros([1, STATE_EMBEDDING_SIZE]), dtype=torch.float)

    full_tensor = policy_net(encoded_state)

    action_tensor = full_tensor[:, :N_ACTIONS_THOUGHTS]
    subaction_tensor = full_tensor[:, N_ACTIONS_THOUGHTS:]

    return action_tensor[0].tolist(), subaction_tensor[0].tolist()


def plot_qvalues_distribution(data, plots_folder):
    # Create a figure and axis object for the subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Abstract actions

    # Explode to separate columns and compute the average softmax distribution per condition
    new_data = pd.DataFrame(data["abs_qvalues"].to_list(), columns=list(ACTION_THOUGHTS.values()))
    new_data[["condition", "run"]] = data[["condition", "run"]]
    average_qvalues = new_data.groupby(['condition']).mean().reset_index()
    melted_data = average_qvalues.melt(id_vars='condition')

    # Plotting
    sns.barplot(data=melted_data, x='variable', y='value', hue='condition', palette='muted', ci=None, ax=axes[0])
    axes[0].set_title('Distributions by Intention')
    axes[0].set_xlabel(f'Abstract Action')
    axes[0].set_ylabel('Selection Probability')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, title='Intention / Reward')

    # Specific actions

    # Explode to separate columns and compute the average softmax distribution per condition
    new_data = pd.DataFrame(data["spe_qvalues"].to_list(), columns=list(ACTION_TYPES.values()))
    new_data[["condition", "run"]] = data[["condition", "run"]]
    average_qvalues = new_data.groupby(['condition']).mean().reset_index()
    melted_data = average_qvalues.melt(id_vars='condition')

    # Plotting
    sns.barplot(data=melted_data, x='variable', y='value', hue='condition', palette='muted', ci=None, ax=axes[1])
    axes[1].set_title('Distributions by Intention')
    axes[1].set_xlabel(f'Specific Action')
    axes[1].set_ylabel('Selection Probability')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, title='Intention / Reward')

    plt.tight_layout()
    plt.savefig(plots_folder / f"distribution_qvalues.png", dpi=300)
    plt.show()


def plot_qvalues_stacked(data, plots_folder):
    # Create a figure and axis object for the subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Abstract actions

    # Explode to separate columns and compute the average softmax distribution per condition
    column_names = [el[1:] for el in list(ACTION_THOUGHTS.values())]
    new_data = pd.DataFrame(data["abs_qvalues"].to_list(), columns=column_names)
    new_data = new_data - (1 / N_ACTIONS_THOUGHTS)
    new_data[["condition", "run"]] = data[["condition", "run"]]
    average_qvalues = new_data.groupby(['condition']).mean()

    # Plotting
    average_qvalues.plot(kind='bar', stacked=True, ax=axes[0])
    axes[0].set_title('Distributions by Intention')
    axes[0].set_xlabel(f'Intention / Reward')
    axes[0].set_ylabel('Selection Probability')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=55)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, title='Abstract Action', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="small")

    # Specific actions

    # Explode to separate columns and compute the average softmax distribution per condition
    new_data = pd.DataFrame(data["spe_qvalues"].to_list(), columns=list(ACTION_TYPES.values()))
    new_data = new_data - (1 / N_ACTION_TYPES)
    new_data[["condition", "run"]] = data[["condition", "run"]]
    average_qvalues = new_data.groupby(['condition']).mean()

    # Plotting
    merged_colors = plt.cm.tab20.colors + plt.cm.tab20b.colors
    average_qvalues.plot(kind='bar', stacked=True, ax=axes[1], color=merged_colors)
    axes[1].set_title('Distributions by Intention')
    axes[1].set_xlabel(f'Intention / Reward')
    axes[1].set_ylabel('Selection Probability')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=55)
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, title='Specific Action', bbox_to_anchor=(1.05, 1.25), loc='upper left', fontsize="small")

    plt.tight_layout()
    plt.savefig(plots_folder / f"stacked_qvalues.png", dpi=300)
    plt.show()


def get_final_knowledge_stats(data, plots_folder):
    # Calculate mean for each condition and timestep
    avg_over_runs = data[['condition', 'timestep', "Average degree", "Sparseness", "Shortest path", "Total triples",
                          "Average population", "Ratio claims to triples", "Ratio perspectives to claims",
                          "Ratio conflicts to claims"]].groupby(['condition', 'timestep']).mean().reset_index()

    # Extract the last timestep per condition
    final_k = avg_over_runs.groupby('condition').apply(lambda x: x.loc[x['timestep'].idxmax() - 1]) \
        .reset_index(drop=True)

    final_k.drop(columns=["timestep"], inplace=True)

    final_k.to_csv(plots_folder / "final_k.csv")


def plot_cum_reward(data, plots_folder, log=False):
    data['cumulative_reward'] = data.groupby(['condition', 'run'])['rewards'].cumsum()

    # Calculate mean and standard deviation for each condition and timestep
    avg_over_runs = data.groupby(['condition', 'timestep']).agg(mean_cumulative_reward=('cumulative_reward', 'mean'),
                                                                std_cumulative_reward=(
                                                                    'cumulative_reward', 'std')).reset_index()

    # Plotting
    plt.figure(figsize=(12, 8), tight_layout=True)
    sns.lineplot(data=avg_over_runs, x='timestep', y='mean_cumulative_reward', hue='condition', linewidth=.5)

    # Add shaded area for standard deviation
    for condition in avg_over_runs['condition'].unique():
        condition_data = avg_over_runs[avg_over_runs['condition'] == condition]
        plt.fill_between(condition_data['timestep'],
                         condition_data['mean_cumulative_reward'] - condition_data['std_cumulative_reward'],
                         condition_data['mean_cumulative_reward'] + condition_data['std_cumulative_reward'],
                         alpha=0.3)

    plt.title('Cumulative Rewards Over Time Steps by Intention')
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Reward')
    plt.legend(title='Intention / Reward')
    if log:
        plt.yscale('log')
        plt.savefig(plots_folder / "cumulative_reward(log).png", dpi=300)
    else:
        plt.savefig(plots_folder / "cumulative_reward.png", dpi=300)
    plt.show()


def plot_action_count(data, plots_folder):
    data["thought_types"] = data["actions"].apply(lambda a: ACTION_THOUGHTS.get(a, None))

    # Count the occurrences of each action per condition
    action_counts = data.groupby(['condition', 'thought_types']).size().reset_index(name='counts')

    # Plotting
    plt.figure(figsize=(12, 8), tight_layout=True)
    sns.barplot(data=action_counts, x='thought_types', y='counts', hue='condition')

    plt.title('Action Counts by Intention')
    plt.xlabel('Action')
    plt.ylabel('Counts')
    plt.legend(title='Intention / Reward')
    plt.xticks(rotation=45, ha="right")
    plt.savefig(plots_folder / "action_type_count.png", dpi=300)
    plt.show()


def main(args):
    state_encoder = StateEncoder(HarryPotterRDF('.'))

    experiment_dir = EXPERIMENTS_PATH / args.experiment_id
    conditions = sorted(f for f in experiment_dir.iterdir() if f.is_dir())

    experiments_data = []
    qvalues_data = []

    for i, condition in enumerate(conditions):

        runs = sorted(f for f in condition.iterdir() if f.is_dir())

        for run in runs:
            chats = sorted(f for f in run.iterdir() if f.is_dir())

            # Get qvalues TODO input perfect graph
            last_chat = chats[-1]
            policy_net = load_trained_model(last_chat / "thoughts.pt")
            abs_qvalues, spe_qvalues = get_qvalues(policy_net, state_encoder, state_file=None)
            qvalues_data.append({"condition": condition.name, "run": run.name,
                                 "abs_qvalues": abs_qvalues, "spe_qvalues": spe_qvalues})

            # Get data for each chat
            for chat in chats:
                selections_file = chat / "selection_history.json"
                states_file = chat / "state_history.json"

                try:
                    # Data for cumulative rewards
                    with open(selections_file) as f:
                        selections_dict = json.load(f)

                    selections_data = pd.DataFrame.from_dict(selections_dict)
                    selections_data["condition"] = condition.name
                    selections_data["run"] = run.name
                    selections_data["chat"] = chat.name.split("_")[1]
                    selections_data["turn"] = range(len(selections_data))

                    experiments_data.append(selections_data)

                    # Data for states
                    with open(states_file) as f:
                        states_list = json.load(f)

                    states_data = pd.DataFrame(states_list)
                    selections_data["Average degree"] = states_data["Average degree"]
                    selections_data["Sparseness"] = states_data["Sparseness"]
                    selections_data["Shortest path"] = states_data["Shortest path"]
                    selections_data["Total triples"] = states_data["Total triples"]
                    selections_data["Average population"] = states_data["Average population"]
                    selections_data["Ratio claims to triples"] = states_data["Ratio claims to triples"]
                    selections_data["Ratio perspectives to claims"] = states_data["Ratio perspectives to claims"]
                    selections_data["Ratio conflicts to claims"] = states_data["Ratio conflicts to claims"]

                except:
                    continue

    # Create dataframe for softmax
    qvalues_data = pd.DataFrame(qvalues_data)
    plot_qvalues_stacked(qvalues_data, PLOTS_PATH)
    plot_qvalues_distribution(qvalues_data, PLOTS_PATH)

    # Create dataframe and calculate timesteps
    history_data = pd.concat(experiments_data, ignore_index=True)
    history_data["timestep"] = history_data.groupby(['condition', 'run']).cumcount() + 1

    # Plots and outputs
    plot_action_count(history_data, PLOTS_PATH)
    get_final_knowledge_stats(history_data, PLOTS_PATH)
    plot_cum_reward(history_data, PLOTS_PATH, log=True)
    plot_cum_reward(history_data, PLOTS_PATH, log=False)

    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", default="e1 (25turns_3chats_3runs)", type=str, help="ID for an experiment")

    args = parser.parse_args()

    PLOTS_PATH = PLOTS_PATH / args.experiment_id
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)

    main(args)
