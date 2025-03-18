import os
from pathlib import Path

ROOT_PATH = (str(Path(__file__).resolve().parent.parent.parent))

CONFIGS_PATH = os.path.join(ROOT_PATH, 'configs')
NEPTUNE_CONFIG_PATH = os.path.join(CONFIGS_PATH, 'logging', 'neptune-config.json')
TRAIN_CONFIG_PATH = os.path.join(CONFIGS_PATH, 'train', 'default.json')
TEST_CONFIG_PATH = os.path.join(CONFIGS_PATH, 'test', 'default.json')

TENSORBOARD_LOGS_PATH = os.path.join(CONFIGS_PATH, 'logs', 'tensorboard')
