# Copyright (c) 2024 Philipp Spiess
# All rights reserved.

LAYER_SIZE = 256
LEARNING_RATE = 0.00001
TRAINING_BATCH = 50000
MINI_BATCH_SIZE = 64
NUM_WORKERS = 64
ENTROPY_BONUS = 0.01         # 0.01 better for velocity control at 30Hz - 0.001 at 120 Hz

TASK = "standing"            # standing standing_up walking
ROBOT_TYPE = "half"               # legs half full
TESTING = False
MAX_EPISODES = 500000
ROBUST = False
MIMIC = False
LOW_ENERGY = False

REAL = False
REAL_TRAINING = False

MAX_EPISODE_LENGTH = 300     # This is equivalent to 10 seconds
PARALLEL = True
SHUFFLE = True
CURIOSITY_BONUS = 0.0
ENTROPY_DECAY = 1.0
LEARNING_DECAY = 1.0
LOSS_CLIPPING = 0.2
GAMMA = 0.99
EPOCHS = 10
LAMDA = 0.9
DEFAULT_TRAINING_BATCH = 50000

GOAL = 100
if "standing" in TASK:
    GOAL = 80
elif "jumping_mimic" in TASK:
    GOAL = 120
elif "walking" in TASK:
    GOAL = 300
