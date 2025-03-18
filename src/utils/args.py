import argparse
import json


def validate_args(args, parser):
    """
    Validates the arguments passed to the script.
    :param args:
    :return:
    """
    # Check expected types for arguments/actions according to parser
    for action in parser._actions:
        if action.dest == 'help': # If helper_action, do not check
            continue
        if action.type:  # If store_action, check if type is expected by parser
            if action.dest in ['weights', 'gpus']:  # These arguments can be None
                expected_types = [action.type, type(None)]
            else:
                expected_types = [action.type]
        else:  # If store_true_action, check if type is boolean
            expected_types = [bool]
        if type(args[action.dest]) not in expected_types:
            raise argparse.ArgumentTypeError(f'Argument "{action.dest}" must be of type {" or ".join(str(t) for t in expected_types)}.')

    # Check contraints
    if args['gpus'] is not None and args['device'] == "cpu":
        raise argparse.ArgumentTypeError('The argument "--gpus" can only be used when "--device" is set to "cuda"')
    if args['gpus'] is None and args['device'] == "cuda":
        raise argparse.ArgumentTypeError('The argument "--gpus" must be used when "--device" is set to "cuda"')
    if args['resume'] is True and args['weights'] is None:
        raise argparse.ArgumentTypeError('The argument "--resume" can only be used when "--weights" is defined')


def read_config(config_file):
    """
    Read configurations from file.
    :param config_file: Path to config file
    :return: Configuration dictionary
    """
    with open(config_file, 'r') as file:
        config = json.load(file)
        return config


def update_missing_args(args):
    """
    Use default argument values from configuration file for missing arguments.
    :param args: Argument dictionary
    :return: Updated argument dictionary
    """
    updated_args = {}
    config = read_config(args['config'])
    for key, value in args.items():
        if value is not None:
            updated_args[key] = value
        else:
            updated_args[key] = config[key]
    return updated_args