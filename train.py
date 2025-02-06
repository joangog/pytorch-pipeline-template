import argparse
import os

from datasets.CustomDataset import CustomDataset, CIFAR10
from utils.args import validate_args, update_missing_args
from utils.models import load_checkpoint

# PARSE ARGUMENTS ------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Pytorch model training script')

# Config argument
default_configs = os.path.join(os.getcwd(),'configs','train','default.json')
parser.add_argument('--config', type=str, default=default_configs, help='Config file with default optional argument values')

# Regular arguments (default values set by config file)
parser.add_argument('--dataset', '-D', type=str, nargs='?', choices=['CIFAR10'], help='Dataset name')
parser.add_argument('--data', '-d', type=str, nargs='?', help='Path to data folder')
parser.add_argument('--model', '-M', type=str, nargs='?', choices=[''], help='Model name')
parser.add_argument('--weights', '-w', type=str, nargs='?', help='Path to model checkpoint')
parser.add_argument('--batch', '-b', type=int, nargs='?', help='Batch size')
parser.add_argument('--epochs', '-e', type=int, nargs='?', help='Number of epochs')
parser.add_argument('--learning-rate', '-lr', type=float, nargs='?', help='Learning rate')
parser.add_argument('--momentum', type=float, nargs='?', help='Momentum')
parser.add_argument('--weight_decay', type=float, nargs='?', help='Weight decay')
parser.add_argument('--patience', type=int, nargs='?', help='Patience for early stopping')
parser.add_argument('--device', type=str, nargs='?', default = "cpu", choices = ['cpu', 'cuda'], help = 'Device type to use (cpu, cuda)')
parser.add_argument('--gpus', type=int, nargs="*", help='GPU devices to use')
parser.add_argument('--seed', type=int, nargs='?', help='Random seed')

# Flag arguments
parser.add_argument('--resume', '-r', action='store_true', default=False, help='Resume from checkpoint')
parser.add_argument('--tensorboard', action='store_true', default=False, help='Enable tensorboard logging')

# Parse and validate the arguments
args = vars(parser.parse_args())
args = update_missing_args(args)  # Update args with default values from config file
validate_args(args, parser)


# MAIN SCRIPT ----------------------------------------------------------------------------------------------------------

Dataset = 0
dataset = CIFAR10(args['data'])


# if args['weights']:
#     load_checkpoint(args['weights'], args['resume'])