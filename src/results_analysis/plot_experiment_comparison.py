import argparse
import json
from pathlib import Path
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import torch
from sklearn.preprocessing import MinMaxScaler

from dialogue_system.d2q_selector import DQN
from dialogue_system.rl_utils.hp_rdf_dataset import HarryPotterRDF
from dialogue_system.rl_utils.rl_parameters import DEVICE, STATE_EMBEDDING_SIZE, N_ACTIONS_THOUGHTS, N_ACTION_TYPES, \
    ACTION_THOUGHTS, ACTION_TYPES, METRICS, METRICS_TOINCLUDE
from dialogue_system.rl_utils.state_encoder import StateEncoder
from dialogue_system.utils.global_variables import RESOURCES_PATH

EXPERIMENTS_PATH = Path(f"{RESOURCES_PATH}/experiments").resolve()
EXPERIMENT_NAMES = {"1": "e1: perfect knowledge", "2": "e2: imperfect knowledge"}
BASELINE_NAMES = {"1": "action learning", "3": "b1: abstract action learning", "4": "b2: specific action learning",
                  "5": "b3: random action"}
PLOTS_PATH = Path(f"{RESOURCES_PATH}/plots").resolve()
STATE_ENCODER = StateEncoder(HarryPotterRDF('.'))
COLOR_PALETTE = sns.color_palette("tab10", len(METRICS))
COLOR_MAPPING = dict(zip(METRICS.keys(), COLOR_PALETTE))


def replace_experiment_name(experiments, e_id):
    if len(experiments) == 2:
        name = f"{EXPERIMENT_NAMES[e_id.split(' ')[0][-1]]}"
    else:
        name = f"{BASELINE_NAMES[e_id.split(' ')[0][-1]]}"

    return name


def collect_data(conditions, testing_experiment=False):
    experiments_data = []
    qvalues_data = []
    utterances_data = []

    for i, condition in enumerate(conditions):
        runs = sorted(f for f in condition.iterdir() if f.is_dir())

        for run in runs:
            chats = sorted(f for f in run.iterdir() if f.is_dir())

            if testing_experiment:
                # Get qvalues for last chat only
                policy_net = load_trained_model(chats)
                abs_qvalues, spe_qvalues = get_qvalues(policy_net, STATE_ENCODER, state_file=None)
                qvalues_data.append({"condition": condition.name.replace(" ", "-"), "run": run.name[-1],
                                     "abs_qvalues": abs_qvalues, "spe_qvalues": spe_qvalues})

            # Get data for each chat
            for chat in chats:
                selections_data = parse_selection_data(condition, run, chat)
                selections_data = add_state_data(chat, selections_data)

                if "rewards" not in selections_data.columns:
                    selections_data = patch_reward(condition, selections_data)

                # Cleanup
                selections_data['condition'] = selections_data['condition'].str.replace("-", " ", regex=False)
                selections_data['selections'] = selections_data['selections'].str.lstrip("_")
                experiments_data.append(selections_data)

            # get utterances for last chat (most trained model)
            with open(chats[-1] / "chat_history.json") as f:
                utterances_dict = json.load(f)
            user_u = [f'user: {el["utterance"]}' for el in utterances_dict["capsules_submitted"]]
            bot_u = [f'agent: {el["utterance"]}' for el in utterances_dict["capsules_suggested"]]
            transcript = []
            for u, b in zip(user_u, bot_u):
                transcript.append(u)
                transcript.append(b)
            utterances_data.append(transcript)

    return experiments_data, qvalues_data, utterances_data


def patch_reward(condition, selections_data):
    selections_data["tmp"] = selections_data[condition.name.replace('-', ' ')].shift(1)
    selections_data["rewards"] = selections_data[condition.name.replace('-', ' ')] / \
                                 selections_data["tmp"]
    selections_data["rewards"] = selections_data["rewards"] - 1
    selections_data.drop(columns=['tmp'], inplace=True)

    return selections_data


def add_state_data(chat, selections_data):
    with open(chat / "state_history.json") as f:
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

    return selections_data


def parse_selection_data(condition, run, chat):
    try:
        with open(chat / "selection_history.json") as f:
            selections_dict = json.load(f)

        selections_data = pd.DataFrame.from_dict(selections_dict)
        selections_data["condition"] = condition.name
        selections_data["run"] = run.name[-1]
        selections_data["chat"] = chat.name.split("_")[1]
        selections_data["turn"] = range(len(selections_data))
        num_turns = len(selections_data)

    except Exception as e:
        print(f"################################\n"
              f"Condition {condition.name}, Run: {run.name}, Chat: {chat.name.split('_')[1]}, error: {e}"
              f"################################\n")
        selections_data = pd.DataFrame(columns=['actions', 'rewards', 'states', 'selections'])
        selections_data["condition"] = condition.name
        selections_data["run"] = run.name
        selections_data["chat"] = chat.name.split("_")[1]
        selections_data["turn"] = range(num_turns)

    return selections_data


def load_trained_model(chats):
    last_chat = chats[-1]
    filename = last_chat / "thoughts.pt"

    policy_net = DQN(STATE_EMBEDDING_SIZE, N_ACTIONS_THOUGHTS, N_ACTION_TYPES).to(DEVICE)

    # Copy parameters from file
    model_dict = policy_net.state_dict()
    model_checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    new_dict = {k: v for k, v in model_checkpoint.items() if k in model_dict.keys()}
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


def get_final_knowledge_stats(data, plots_folder):
    # Calculate mean for each condition and timestep
    avg_over_runs = data[['condition', 'timestep',
                          "Average degree", "Sparseness", "Shortest path",
                          "Total triples", "Average population",
                          "Ratio claims to triples", "Ratio perspectives to claims"  # , "Ratio conflicts to claims"
                          ]] \
        .groupby(['condition', 'timestep']).mean().reset_index()

    # Extract the last timestep per condition
    final_k = avg_over_runs.groupby('condition').apply(lambda x: x.loc[x['timestep'].idxmax() - 1]) \
        .reset_index(drop=True)

    final_k.drop(columns=["timestep"], inplace=True)

    final_k["Average degree"] = final_k["Average degree"].round(3)
    final_k["Sparseness"] = final_k["Sparseness"] * 100
    final_k["Sparseness"] = final_k["Sparseness"].round(3)
    final_k["Shortest path"] = final_k["Shortest path"].round(3)
    final_k["Total triples"] = final_k["Total triples"].astype(int)
    final_k["Average population"] = final_k["Average population"].round(2)

    final_k.to_csv(plots_folder / "acc_knowledge.csv")

    latex_table = final_k.to_latex(index=False)
    with open(plots_folder / "acc_knowledge.tex", 'w') as f:
        f.write(latex_table)

    return final_k


def plot_action_count(data, plots_folder, action_type='abstract'):
    mapping_dict = ACTION_THOUGHTS if action_type == "abstract" else ACTION_TYPES
    data[f"{action_type}_action_names"] = data[f"{action_type}_actions"].apply(lambda a: mapping_dict.get(a, None))

    # TODO here use only the results from the last chat

    # Count the occurrences of each action per condition
    action_counts = data.groupby(['condition', f"{action_type}_action_names"]).size().reset_index(name='counts')

    # Plotting
    plt.figure(figsize=(12, 8), tight_layout=True)
    sns.barplot(data=action_counts, x=f"{action_type}_action_names", y='counts', hue='condition', palette=COLOR_MAPPING)

    plt.title(f'{action_type.title()} Action Counts by Intention')
    plt.xlabel(f'{action_type.title()} Action')
    plt.ylabel('Counts')
    plt.legend(title='Intention / Reward')
    plt.xticks(rotation=45, ha="right")
    plt.savefig(str(plots_folder / f"action_count({action_type}).png"), dpi=300)
    # plt.show()


def plot_qvalues_stacked(data, plots_folder):
    # Create a figure and axis object for the subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Abstract actions

    # Explode to separate columns and compute the average softmax distribution per condition
    column_names = [el[1:] for el in list(ACTION_THOUGHTS.values())]
    new_data = pd.DataFrame(data["abs_qvalues"].to_list(), columns=column_names)
    new_data = new_data - (1 / N_ACTIONS_THOUGHTS)
    new_data["condition"] = data["condition"].tolist()
    # new_data["run"] = data["run"].tolist()
    average_qvalues = new_data.groupby(['condition']).mean()

    # Plotting
    average_qvalues.plot(kind='bar', stacked=True, ax=axes[0])
    axes[0].set_title('Distributions by Intention')
    axes[0].set_xlabel(f'Intention / Reward')
    axes[0].set_ylabel('Normalized Selection Probability')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=55)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, title='Abstract Action', bbox_to_anchor=(1.05, 1), loc='upper left',
                   fontsize="small")

    # Specific actions

    # Explode to separate columns and compute the average softmax distribution per condition
    new_data = pd.DataFrame(data["spe_qvalues"].to_list(), columns=list(ACTION_TYPES.values()))
    new_data = new_data - (1 / N_ACTION_TYPES)
    new_data["condition"] = data["condition"].tolist()
    # new_data["run"] = data["run"].tolist()
    average_qvalues = new_data.groupby(['condition']).mean()

    # Plotting
    merged_colors = plt.cm.tab20.colors + plt.cm.tab20b.colors
    average_qvalues.plot(kind='bar', stacked=True, ax=axes[1], color=merged_colors)
    axes[1].set_title('Distributions by Intention')
    axes[1].set_xlabel(f'Intention / Reward')
    axes[1].set_ylabel('Normalized Selection Probability')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=55)
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, title='Specific Action', bbox_to_anchor=(1.05, 1.25), loc='upper left',
                   fontsize="small")

    plt.tight_layout()
    plt.savefig(str(plots_folder / f"stacked_qvalues.png"), dpi=300)
    # plt.show()


def plot_spider_scores(data, plots_folder):
    data = data.set_index("condition")

    scaler = MinMaxScaler()
    scaled_graph_scores = scaler.fit_transform(data)

    fig = go.Figure()

    for scores, id in zip(scaled_graph_scores, list(data.index)):
        fig.add_trace(go.Scatterpolar(r=scores,
                                      theta=list(data.columns),
                                      fill='toself',
                                      name=f'{id}'))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True), ), showlegend=True, title="Knowledge profiles")

    # fig.show()
    fig.write_image(plots_folder / f'starplot_acc_knowledge.png')


def plot_avg_reward(data, plots_folder):
    data.loc[:, 'cumulative_reward'] = data.groupby(['condition', 'run'])['rewards'].cumsum()
    data.loc[:, 'average_reward'] = data['cumulative_reward'] / data['timestep']

    # Calculate mean and standard deviation for each condition and timestep
    avg_over_runs = data.groupby(['condition', 'timestep']) \
        .agg(mean_average_reward=('average_reward', 'mean'),
             std_average_reward=('average_reward', 'std')).reset_index()

    # Calculate the moving average using a sliding window of 5 timesteps
    avg_over_runs['mean_average_reward_rolling'] = avg_over_runs.groupby(['condition'])['mean_average_reward'] \
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    avg_over_runs['std_average_reward_rolling'] = avg_over_runs.groupby(['condition'])['std_average_reward'] \
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())

    # only half the x-axis
    avg_over_runs = avg_over_runs[avg_over_runs["timestep"] < 30]

    # Plotting
    plt.figure(figsize=(12, 8), tight_layout=True)
    sns.lineplot(data=avg_over_runs, x='timestep', y='mean_average_reward_rolling', hue='condition', linewidth=1.5,
                 palette=COLOR_MAPPING)

    # Add shaded area for standard deviation
    for condition in avg_over_runs['condition'].unique():
        condition_data = avg_over_runs[avg_over_runs['condition'] == condition]
        plt.fill_between(condition_data['timestep'],
                         condition_data['mean_average_reward_rolling'] - condition_data['std_average_reward_rolling'],
                         condition_data['mean_average_reward_rolling'] + condition_data['std_average_reward_rolling'],
                         alpha=0.3)

    plt.xlim(left=0)
    plt.title('Average Rewards Over Time Steps by Intention')
    plt.xlabel('Training Timestep')
    plt.ylabel('Average Reward')
    plt.legend(title='Intention / Reward')
    plt.savefig(str(plots_folder / "average_reward.png"), dpi=300)
    # plt.show()


def plot_cum_reward_per_chat(data, plots_folder, log=False):
    data['cumulative_reward'] = data.groupby(['condition', 'run', 'chat'])['rewards'].cumsum()
    data["timestep"] = data.groupby(['condition', 'run', 'chat']).cumcount() - 1
    data = data[data["timestep"] == 9]

    # Calculate mean and standard deviation for each condition and timestep
    avg_over_runs = data.groupby(['condition', 'chat']).agg(mean_cumulative_reward=('cumulative_reward', 'mean'),
                                                            std_cumulative_reward=(
                                                                'cumulative_reward', 'std')).reset_index()

    # Calculate the moving average using a sliding window of 5 timesteps
    avg_over_runs['mean_cumulative_reward_rolling'] = avg_over_runs.groupby(['condition'])[
        'mean_cumulative_reward'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    avg_over_runs['std_cumulative_reward_rolling'] = avg_over_runs.groupby(['condition'])[
        'std_cumulative_reward'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

    # Plotting
    plt.figure(figsize=(12, 8), tight_layout=True)
    sns.lineplot(data=avg_over_runs, x='chat', y='mean_cumulative_reward', hue='condition', linewidth=.5,
                 palette=COLOR_MAPPING)

    # Add shaded area for standard deviation
    for condition in avg_over_runs['condition'].unique():
        condition_data = avg_over_runs[avg_over_runs['condition'] == condition]
        plt.fill_between(condition_data['chat'],
                         condition_data['mean_cumulative_reward'] - condition_data['std_cumulative_reward'],
                         condition_data['mean_cumulative_reward'] + condition_data['std_cumulative_reward'],
                         alpha=0.3)

    plt.xlim(left=0)
    plt.title('Cumulative Rewards Over Testing Checkpoint by Intention')
    plt.xlabel('Testing Checkpoint')
    plt.ylabel('Cumulative Reward')
    plt.legend(title='Intention / Reward')
    if log:
        plt.yscale('log')
        plt.savefig(str(plots_folder / "cumulative_reward_perChat(log).png"), dpi=300)
    else:
        plt.savefig(str(plots_folder / "cumulative_reward_perChat.png"), dpi=300)
    # plt.show()


def plot_avg_reward_compared_single_condition(data, plots_folder, range=False, condition="Ratio claims to triples",
                                              plot_id=""):
    data = data[data["condition"] == condition]

    data.loc[:, 'cumulative_reward'] = data.groupby(['experiment', 'run'])['rewards'].cumsum()
    data.loc[:, 'average_reward'] = data['cumulative_reward'] / data['timestep']

    # Calculate mean and standard deviation for each condition and timestep
    avg_over_runs = data.groupby(['experiment', 'timestep']) \
        .agg(mean_average_reward=('average_reward', 'mean'),
             std_average_reward=('average_reward', 'std')).reset_index()

    # Calculate the moving average using a sliding window of 5 timesteps
    avg_over_runs['mean_average_reward_rolling'] = avg_over_runs.groupby(['experiment'])['mean_average_reward'] \
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())

    # Fix visibility
    avg_over_runs = avg_over_runs[avg_over_runs["timestep"] < 30]
    if args.experiment_id.startswith("e"):
        avg_over_runs = avg_over_runs[avg_over_runs["timestep"] > 0]

    # Plotting
    plt.figure(figsize=(12, 8), tight_layout=True)
    sns.lineplot(data=avg_over_runs, x='timestep', y='mean_average_reward_rolling', style='experiment',
                 linewidth=1.5, palette=COLOR_MAPPING)

    # Add shaded area for standard deviation
    if range:
        for experiment in avg_over_runs['experiment'].unique():
            condition_data = avg_over_runs[avg_over_runs['experiment'] == experiment]
            plt.fill_between(condition_data['timestep'],
                             condition_data['mean_average_reward'] - condition_data['std_average_reward'],
                             condition_data['mean_average_reward'] + condition_data['std_average_reward'],
                             alpha=0.3)

    plt.xlim(left=0)
    plt.title(f'Average Rewards Over Time Steps for Intention: {condition}')
    plt.xlabel('Training Timestep')
    plt.ylabel('Average Reward')
    plt.legend()
    if range:
        plt.savefig(str(plots_folder / f"comparative_average_reward({plot_id}) - {condition}.png"), dpi=300)
    else:
        plt.savefig(str(plots_folder / f"comparative_average_reward-no_range({plot_id}) - {condition}.png"), dpi=300)
    # plt.show()


def plot_avg_reward_compared(data, plots_folder, range=False, condition="", plot_id=""):
    data.loc[:, 'cumulative_reward'] = data.groupby(['experiment', 'condition', 'run'])['rewards'].cumsum()
    data.loc[:, 'average_reward'] = data['cumulative_reward'] / data['timestep']

    # Calculate mean and standard deviation for each condition and timestep
    avg_over_runs = data.groupby(['experiment', 'condition', 'timestep']) \
        .agg(mean_average_reward=('average_reward', 'mean'),
             std_average_reward=('average_reward', 'std')).reset_index()

    # Calculate the moving average using a sliding window of 5 timesteps
    avg_over_runs['mean_average_reward_rolling'] = avg_over_runs.groupby(['experiment',
                                                                          'condition'])['mean_average_reward'] \
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())

    # Fix visibility
    avg_over_runs = avg_over_runs[avg_over_runs["timestep"] < 30]
    if args.experiment_id.startswith("e"):
        avg_over_runs = avg_over_runs[avg_over_runs["timestep"] > 0]
    if condition:
        avg_over_runs = avg_over_runs[avg_over_runs["condition"] == condition]

    # Plotting
    plt.figure(figsize=(12, 8), tight_layout=True)
    sns.lineplot(data=avg_over_runs, x='timestep', y='mean_average_reward_rolling', hue='condition', style='experiment',
                 linewidth=1.5, palette=COLOR_MAPPING)

    # Add shaded area for standard deviation
    if range:
        for experiment in avg_over_runs['experiment'].unique():
            for condition in avg_over_runs['condition'].unique():
                condition_data = avg_over_runs[(avg_over_runs['condition'] == condition) &
                                               (avg_over_runs['experiment'] == experiment)]
                color = COLOR_MAPPING[condition]

                plt.fill_between(condition_data['timestep'],
                                 condition_data['mean_average_reward'] - condition_data['std_average_reward'],
                                 condition_data['mean_average_reward'] + condition_data['std_average_reward'],
                                 alpha=0.3, color=color)

    plt.xlim(left=0)
    plt.title('Average Rewards Over Time Steps by Intention')
    plt.xlabel('Timestep')
    plt.ylabel('Average Reward')
    plt.legend(title='Intention / Reward')
    if range:
        plt.savefig(str(plots_folder / f"comparative_average_reward({plot_id}) - {condition}.png"), dpi=300)
    else:
        plt.savefig(str(plots_folder / f"comparative_average_reward-no_range({plot_id}) - {condition}.png"), dpi=300)
    # plt.show()


def plot_cum_reward_compared(data, plots_folder, range=False, plot_id=""):
    # Calculate cumulative sum
    data.loc[:, 'cumulative_reward'] = data.groupby(['experiment', 'condition', 'run'])['rewards'].cumsum()

    # Calculate mean and standard deviation for each condition and timestep
    avg_over_runs = data.groupby(['experiment', 'condition', 'timestep']).agg(
        mean_cumulative_reward=('cumulative_reward', 'mean'),
        std_cumulative_reward=('cumulative_reward', 'std')).reset_index()

    # Calculate the moving average using a sliding window of 5 timesteps
    avg_over_runs['mean_cumulative_reward_rolling'] = avg_over_runs.groupby(['experiment', 'condition'])[
        'mean_cumulative_reward'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    avg_over_runs['std_cumulative_reward_rolling'] = avg_over_runs.groupby(['experiment', 'condition'])[
        'std_cumulative_reward'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

    # # f x axis
    # avg_over_runs = avg_over_runs[avg_over_runs["timestep"] > 0]

    # Plotting
    plt.figure(figsize=(12, 8), tight_layout=True)
    sns.lineplot(data=avg_over_runs, x='timestep', y='mean_cumulative_reward_rolling', hue='condition',
                 style='experiment', linewidth=1.5, palette=COLOR_MAPPING)

    # Add shaded area for standard deviation
    if range:
        for experiment in avg_over_runs['experiment'].unique():
            for condition in avg_over_runs['condition'].unique():
                condition_data = avg_over_runs[(avg_over_runs['condition'] == condition) &
                                               (avg_over_runs['experiment'] == experiment)]
                color = COLOR_MAPPING[condition]

                plt.fill_between(condition_data['timestep'],
                                 condition_data['mean_cumulative_reward_rolling']
                                 - condition_data['std_cumulative_reward_rolling'],
                                 condition_data['mean_cumulative_reward_rolling']
                                 + condition_data['std_cumulative_reward_rolling'],
                                 alpha=0.3, color=color)

    plt.xlim(left=0)
    plt.title('Cumulative Rewards Over Turns by Intention')
    plt.xlabel('Turn')
    plt.ylabel('Cumulative Reward')
    plt.legend(title='Intention / Reward')
    if range:
        plt.savefig(str(plots_folder / f"comparative_cumulative_reward({plot_id}).png"), dpi=300)
    else:
        plt.savefig(str(plots_folder / f"comparative_cumulative_reward-no_range({plot_id}.png"), dpi=300)


def main(args):
    testing_experiment = args.experiment_id.startswith("t")

    # Take care of paths
    experiment_plots_path = PLOTS_PATH / args.experiment_id
    experiment_plots_path.mkdir(parents=True, exist_ok=True)

    # Collect data for this scenario
    experiment_dir = EXPERIMENTS_PATH / args.experiment_id
    conditions = sorted(f for f in experiment_dir.iterdir() if f.is_dir())
    experiments_data, qvalues_data, utterances_data = collect_data(conditions, testing_experiment=testing_experiment)

    # Plot qvalues
    if testing_experiment:
        qvalues_data = pd.DataFrame(qvalues_data)
        qvalues_data = qvalues_data[qvalues_data["condition"].isin(METRICS_TOINCLUDE)]
        plot_qvalues_stacked(qvalues_data, experiment_plots_path)  # for test

    # Calculate timesteps
    history_data = pd.concat(experiments_data, ignore_index=True)
    history_data["timestep"] = history_data.groupby(['condition', 'run']).cumcount() - 1

    if not testing_experiment:
        # Plot average reward
        plot_avg_reward(history_data[["condition", "run", "timestep", "rewards"]], experiment_plots_path)  # for train

        # Plot cum reward per chat (not super useful?)
        plot_cum_reward_per_chat(history_data, experiment_plots_path)  # for train

    if testing_experiment:
        # Filter only well-trained conditions
        history_data = history_data[history_data["condition"].isin(METRICS_TOINCLUDE)]

        # Plot action counts
        plot_action_count(history_data, experiment_plots_path, action_type="abstract")  # for test
        plot_action_count(history_data, experiment_plots_path, action_type="specific")  # for test

        # Plot final knowledge
        final_k = get_final_knowledge_stats(history_data, experiment_plots_path)  # for test
        plot_spider_scores(final_k, experiment_plots_path)

        # Export chat history, sample three
        chats_sample = {"sample1": sample(utterances_data, 1)[0],
                        "sample2": sample(utterances_data, 1)[0],
                        "sample3": sample(utterances_data, 1)[0]}
        with open(experiment_plots_path / 'sample_transcripts.json', 'w') as f:
            json.dump(chats_sample, f)  # for test

    return history_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--experiment_id", default="e1 (10turns_8chats_3runs)", type=str, help="ID for an experiment")
    # parser.add_argument("--compare_experiments", default=[], help="IDs of other experiments")

    # parser.add_argument("--experiment_id", default="t1 (10turns_3runs_8checkpoints)", type=str,
    #                     help="ID for an experiment")
    # parser.add_argument("--compare_experiments", default=["t2 (10turns_3runs_8checkpoints)"],
    #                     help="IDs of other experiments")

    # parser.add_argument("--experiment_id", default="e1 (10turns_8chats_3runs)", type=str,
    #                     help="ID for an experiment")
    # parser.add_argument("--compare_experiments", default=["e3 (10turns_8chats_3runs)",
    #                                                       "e4 (10turns_8chats_3runs)",
    #                                                       "e5 (10turns_8chats_3runs)"],
    #                     help="IDs of other experiments")

    parser.add_argument("--experiment_id", default="t1 (10turns_3runs_8checkpoints)", type=str,
                        help="ID for an experiment")
    parser.add_argument("--compare_experiments", default=["t3 (10turns_3runs_8checkpoints)",
                                                          "t4 (10turns_3runs_8checkpoints)",
                                                          "t5 (10turns_3runs_8checkpoints)"],
                        help="IDs of other experiments")

    args = parser.parse_args()

    base_experiment = main(args)
    if args.compare_experiments:
        # Save first experiment
        base_experiment["experiment"] = replace_experiment_name(args.compare_experiments, args.experiment_id)
        plot_name = f"{args.experiment_id.split(' ')[0]}"

        all_data = [base_experiment]
        for experiment_id in args.compare_experiments:
            # Make plots for second experiment
            args.experiment_id = experiment_id
            compare_experiment = main(args)

            # Fix names
            plot_name += f",{args.experiment_id.split(' ')[0]}"
            compare_experiment["experiment"] = replace_experiment_name(args.compare_experiments, args.experiment_id)
            all_data.append(compare_experiment)

        all_data = pd.concat(all_data, ignore_index=True)
        plot_avg_reward_compared(all_data[["experiment", "condition", "run", "timestep", "rewards"]], PLOTS_PATH,
                                 range=False, plot_id=plot_name)  # too crowded
        plot_avg_reward_compared(all_data[["experiment", "condition", "run", "timestep", "rewards"]], PLOTS_PATH,
                                 range=False, condition="Ratio claims to triples", plot_id=plot_name)
        plot_avg_reward_compared_single_condition(all_data[["experiment", "condition", "run", "timestep", "rewards"]],
                                                  PLOTS_PATH, range=False, condition="Ratio claims to triples",
                                                  plot_id=plot_name)
        plot_avg_reward_compared_single_condition(all_data[["experiment", "condition", "run", "timestep", "rewards"]],
                                                  PLOTS_PATH, range=True, condition="Ratio claims to triples",
                                                  plot_id=plot_name)  # std overlap too much
        plot_avg_reward_compared_single_condition(all_data[["experiment", "condition", "run", "timestep", "rewards"]],
                                                  PLOTS_PATH, range=True, condition="Ratio perspectives to claims",
                                                  plot_id=plot_name)  # our condition does not stand out

        # Cumulative rewards only if testing
        if args.experiment_id.startswith("t"):
            plot_cum_reward_compared(all_data[["experiment", "condition", "run", "timestep", "rewards"]], PLOTS_PATH,
                                     range=False, plot_id=plot_name)

    print("DONE")

# "e1 (10turns_8chats_3runs)" => avg reward for training process, qvalues for action distribution. cum reward per chat to see if we are improving
# "t1 (10turns_3runs_8checkpoints)" => knowledge spider and table for acquired knowledge and action counts for testing final.
# "t1 (10turns_3runs)" , True , "t2 (10turns_3runs)" => cum rewards for testing final (comparison)
