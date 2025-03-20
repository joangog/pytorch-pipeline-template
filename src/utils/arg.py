import os
import argparse
import json

from src.utils.path import PATH


def validate_args(args, parser):
    """
    Validates the arguments passed to the script.
    :param args: The args.
    :return: The args with some overridden values if needed.
    """

    # Check expected types for arguments/actions according to parser
    for action in parser._actions:
        if action.dest == 'help':  # If helper_action, do not check
            continue
        if action.type:  # If store_action, check if type is expected by parser
            if action.dest in ['weights', 'gpus', 'data', 'outputs']:  # These arguments can be None
                expected_types = [action.type, type(None)]
            else:
                expected_types = [action.type]
        else:  # If store_true_action, check if type is boolean
            expected_types = [bool]
        if type(args[action.dest]) not in expected_types:
            raise argparse.ArgumentTypeError(
                f'Argument "{action.dest}" must be of type {" or ".join(str(t) for t in expected_types)}.')

    # Check contraints
    if args['gpus'] is not None and args['device'] == "cpu":
        raise argparse.ArgumentTypeError('The argument "--gpus" can only be used when "--device" is set to "cuda"')
    if args['gpus'] is None and args['device'] == "cuda":
        raise argparse.ArgumentTypeError('The argument "--gpus" must be used when "--device" is set to "cuda"')
    if args['resume'] is True and args['weights'] is None:
        raise argparse.ArgumentTypeError('The argument "--resume" can only be used when "--weights" is defined')
    if args['learning_rate'] >= 1:
        raise ValueError('The argument "--learning_rate" must be < 1.')
    if args['momentum'] >= 1:
        raise ValueError('The argument "--momentum" must be < 1.')
    if args['weight_decay'] >= 1:
        raise ValueError('The argument "--weight_decay" must be < 1.')

    # Override values from the config file under conditions
    if not args['data']:
        args['data'] = os.path.join(PATH['DATA'], args['dataset'])
    if not args['outputs']:
        args['outputs'] = PATH['OUTPUTS']

    return args


def read_config(config_file):
    """
    Read configurations from file.
    :param config_file: Path to config file.
    :return: Configuration dictionary.
    """
    with open(config_file, 'r') as file:
        config = json.load(file)
        return config


def update_missing_args(args):
    """
    Use default argument values from configuration file for missing arguments.
    :param args: Argument dictionary.
    :return: Updated argument dictionary.
    """
    updated_args = {}
    config = read_config(args['config'])
    for key, value in args.items():
        if value is not None:
            updated_args[key] = value
        else:
            updated_args[key] = config[key]
    return updated_args
