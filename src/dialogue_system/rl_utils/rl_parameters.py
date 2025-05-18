from collections import namedtuple

import torch

################## RL PARAMETERS ##################
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
REPLAY_PER_TURN = 10
DQN_HIDDEN_SIZE = 64
LR = 1e-5
EPSILON_INFO = {"start": 0.9, "end": 0.05, "decay": 1000}
GAMMA = 0.99
TAU = 0.005

################## STATE REPRESENTATION PARAMETERS ##################
STATE_HIDDEN_SIZE = 64  # original features per node is 87
STATE_EMBEDDING_SIZE = 16  # also n_observations

################## MEMORY PARAMETERS ##################
REPLAY_POOL_SIZE = 500000  # 5000 for DQN, 10000 for tutorial
Transition = namedtuple('Transition', ('state', 'abs_action', 'spe_action', 'next_state', 'reward'))
TaggedTransition = namedtuple('TaggedTransition', ('state', 'abs_action', 'spe_action', 'next_state', 'reward', 'reward_type'))

################## TRAINING PARAMETERS ##################
RESET_FREQUENCY = 2
SHUFFLE_FREQUENCY = 2

################## USER MODEL PARAMETERS ##################
USER_MODEL_CATEGORIES = ['amateur', 'doubtful', 'incoherent', 'confused']

################## REWARD PARAMETERS ##################
METRICS = {'Sparseness': 11, 'Average degree': 12, 'Shortest path': 13, 'Total triples': 14,
           'Average population': 21,
           'Ratio claims to triples': 31, 'Ratio perspectives to claims': 32, 'Ratio conflicts to claims': 33}
METRICS_TOINCLUDE = {'Sparseness': 11, 'Total triples': 14,
                     'Average population': 21,
                     'Ratio claims to triples': 31, 'Ratio perspectives to claims': 32}
METRICS_TOEXCLUDE = {'Average degree': 12, 'Shortest path': 13,
                     'Ratio conflicts to claims': 33}

################## DATASET SPECIFIC PARAMETERS ##################
ACTION_THOUGHTS = {0: '_complement_conflict', 1: '_negation_conflicts',
                   2: '_statement_novelty', 3: '_entity_novelty',
                   4: '_subject_gaps', 5: '_complement_gaps',
                   6: '_overlaps', 7: '_trust'}
N_ACTIONS_THOUGHTS = len(ACTION_THOUGHTS)
ACTION_THOUGHTS_REVERSED = {value: key for key, value in ACTION_THOUGHTS.items()}

ACTION_TYPES = {0: 'character', 1: "centaur", 2: "domestic-elf", 3: "ghost", 4: "giant", 5: "goblin", 6: "muggle",
                7: "spider", 8: "squib", 9: "werewolf", 10: "wizard",

                11: 'attribute', 12: "ability", 13: "activity", 14: "age", 15: "ancestry", 16: "designation",
                17: "enchantment", 18: "gender", 19: "institution", 20: "object", 21: "personality-trait",
                22: "physical-appearance", 23: "product",

                24: "Instance", 25: "Source", 26: "Actor"}
N_ACTION_TYPES = len(ACTION_TYPES)
ACTION_TYPES_REVERSED = {value: key for key, value in ACTION_TYPES.items()}


