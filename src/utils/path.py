import os
from pathlib import Path

PATH = dict()

PATH['ROOT'] = (str(Path(__file__).resolve().parent.parent.parent))

# Assets
PATH['ASSETS'] = os.path.join(PATH['ROOT'], 'assets')

# Data
PATH['DATA'] = os.path.join(PATH['ASSETS'], 'data')

# Configs
PATH['CONFIGS'] = os.path.join(PATH['ASSETS'], 'configs')
PATH['NEPTUNE_CONFIG'] = os.path.join(PATH['CONFIGS'], 'logging', 'neptune-config.json')
PATH['TRAIN_CONFIGS'] = os.path.join(PATH['CONFIGS'], 'train')
PATH['TEST_CONFIGS'] = os.path.join(PATH['CONFIGS'], 'test')
PATH['DEFAULT_TRAIN_CONFIG'] = os.path.join(PATH['TRAIN_CONFIGS'], 'default.json')
PATH['DEFAULT_TEST_CONFIG'] = os.path.join(PATH['TEST_CONFIGS'], 'default.json')

# Outputs
PATH['OUTPUTS'] = os.path.join(PATH['ROOT'], 'outputs')

# Logs
PATH['TENSORBOARD_LOGS'] = os.path.join(PATH['OUTPUTS'], 'logs', 'tensorboard')


def make_project_dirs():
    """
    Makes all essential directories for the project.
    """
    for key, path in PATH.items():
        if os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
