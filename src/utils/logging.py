import os
import json
import logging
import neptune
from torch.utils.tensorboard import SummaryWriter

from src.utils.path import PATH


def suppress_logger_info():
    """
    Suppresses logger (tenrsorboard/neptune) info messages from showing up in the terminal
    """
    logging.getLogger("neptune").setLevel(logging.ERROR)
    logging.getLogger("tensorboard").setLevel(logging.ERROR)


def init_neptune_logger(run_name, fold, initial_epoch, args):
    with open(PATH['NEPTUNE_CONFIG']) as file:
        neptune_config = json.load(file)
    neptune_log = neptune.init_run(project=neptune_config['project'], api_token=neptune_config['api_token'],
                                   # Offline mode saves the files locally.
                                   # Load them online using terminal command "neptune sync -p <project-name>"
                                   # mode='offline',
                                   name=run_name + f'_fold{fold}', tags=[args['dataset'], args['model']])
    neptune_log['parameters'] = {'phase': 'train', 'dataset': args['dataset'], 'model': args['model'],
                                 'checkpoint_path': str(args['checkpoint_path']), 'fold': fold,
                                 'initial_epoch': initial_epoch, 'batch': args['batch'], 'epochs': args['epochs'],
                                 'learning_rate': args['learning_rate'], 'momentum': args['momentum'],
                                 'weight_decay': args['weight_decay'], 'patience': args['patience'],
                                 'optimizer': args['optimizer'], 'seed': args['seed']}
    return neptune_log


def init_tensorboard_logger(run_name, fold, args):
    # Potential Bug: If resuming from a previous run, sometimes the graph has double values at the interrupted epoch
    log_path = os.path.join(PATH['TENSORBOARD_LOGS'], args['dataset'], args['model'], run_name, f'fold_{fold}')
    os.makedirs(log_path, exist_ok=True)
    tensorboard_log = SummaryWriter(os.path.join(log_path))
    return tensorboard_log
