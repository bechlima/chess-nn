import torch

# Board / move space
BOARD_PLANES = 18
MOVE_SPACE = 4672

# Model architecture
CHANNELS = 64
RES_BLOCKS = 4

# MCTS
MCTS_SIMS = 50
CPUCT = 1.5

# Replay buffer + training
REPLAY_BUFFER_SIZE = 100_000
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
SELF_PLAY_GAMES_PER_ITER = 20
TRAIN_STEPS_PER_ITER = 200
GRAD_CLIP = 1.0

# Exploration
DIRICHLET_EPS = 0.25
DIRICHLET_ALPHA = 0.3
TEMP_MOVES = 20       # moves before switching to low temperature
TEMP_EARLY = 1.0
TEMP_LATE = 0.1

# Game limits
MAX_GAME_MOVES = 300

# Arena
ARENA_GAMES = 20
ARENA_THRESHOLD = 0.55
ARENA_ALWAYS_ACCEPT_ITERS = 20  # always accept new model for first N iterations

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# MPS/GPU training settings
# MPS doesn't support float64, use float32 throughout
DTYPE = torch.float32
# Inference workers use CPU (MPS doesn't work well across processes)
INFERENCE_DEVICE = torch.device("cpu")

# Playout cap randomization
FULL_SEARCH_PROB = 0.25   # 25% of moves get full search
FAST_SIMS = 25            # cheap search
FULL_SIMS = 200           # deep search

# Paths
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SELF_PLAY_DIR = os.path.join(DATA_DIR, "self_play")
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
