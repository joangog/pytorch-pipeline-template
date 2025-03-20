import os
from datetime import datetime


def get_decimal_digits(num: float) -> str:
    """
    Returns the decimal digits of a number as a string.
    :param num: Float number.
    :return: Decimal digits string.
    """
    return str(num).split(".")[1] if "." in str(num) else ""


def gen_run_name(args):
    """
    Generates a unique string using the arguments passed to the script to be used for naming the run.
    :param args: The arguments passed to the string.
    :return: Run name string.
    """

    # If we are resuming a previous run
    if args['resume']:
        # Get run_name from the run folder of the weights we are using to resume from
        run_name = args['weights'].split(os.sep)[-2]
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = 'run_e{}_b{}_lr{}_m{}_w{}_p{}_s{}_time_{}'.format(args['epochs'], args['batch'],
                                                                     get_decimal_digits(args['learning_rate']),
                                                                     get_decimal_digits(args['momentum']),
                                                                     get_decimal_digits(args['weight_decay']),
                                                                     args['patience'],
                                                                     args['seed'], timestamp)
    return run_name
