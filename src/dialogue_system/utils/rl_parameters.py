from collections import namedtuple

import torch

################## STATE REPRESENTATION PARAMETERS ##################
STATE_HIDDEN_SIZE = 64  # original features per node is 87
STATE_EMBEDDING_SIZE = 16  # also n_observations

################## MEMORY PARAMETERS ##################
REPLAY_POOL_SIZE = 500  # 5000 for DQN, 10000 for tutorial
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

################## RL PARAMETERS ##################
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4  # 16 for D2Q, 128 tutorial
DQN_HIDDEN_SIZE = 64  # 80 for DQN
LR = 1e-4  # 1e-4 for D2Q
EPSILON_INFO = {"start": 0.9, "end": 0.05, "decay": 1000}
GAMMA = 0.99
TAU = 0.005

################## TRAINING PARAMETERS ##################
RESET_FREQUENCY = 6
SHUFFLE_FREQUENCY = 2

################## DATASET SPECIFIC PARAMETERS ##################
ACTION_THOUGHTS = {0: '_complement_conflict', 1: '_negation_conflicts',
                   2: '_statement_novelty', 3: '_entity_novelty',
                   4: '_subject_gaps', 5: '_complement_gaps',
                   6: '_overlaps', 7: '_trust'}
N_ACTIONS_THOUGHTS = len(ACTION_THOUGHTS)
ACTION_TYPES = {0: 'character', 1: "centaur", 2: "domestic-elf", 3: "ghost", 4: "giant", 5: "goblin", 6: "muggle",
                7: "spider", 8: "squib", 9: "werewolf", 10: "wizard",

                11: 'attribute', 12: "ability", 13: "activity", 14: "age", 15: "ancestry", 16: "designation",
                17: "enchantment", 18: "gender", 19: "institution", 20: "object", 21: "personality-trait",
                22: "physical-appearance", 23: "product",

                24: "Instance", 25: "Source", 26: "Actor"}
N_ACTION_TYPES = len(ACTION_TYPES)
ACTION_TYPES_REVERSED = {value: key for key, value in ACTION_TYPES.items()}
