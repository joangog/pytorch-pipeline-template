import argparse
import os
from tqdm import tqdm
from torchmetrics import AUROC
from torch.utils.data import DataLoader
from torch import nn

from src.modules.Evaluator import Validator
from src.modules.Trainer import Trainer
from src.utils.logging import suppress_logger_info, init_neptune_logger, init_tensorboard_logger
from src.utils.path import PATH, make_project_dirs
from src.utils.arg import validate_args, update_missing_args, read_config
from src.utils.data import split_dataset, select_dataset, collate_fn
from src.utils.model import load_checkpoint, select_model
from src.utils.optimizer import select_optimizer
from src.utils.loss import select_loss
from src.utils.scheduler import select_scheduler
from src.utils.helper import gen_run_name, set_seed

# PARSE ARGUMENTS ------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Pytorch model training script')

# Config argument
parser.add_argument('--config_path', '-c', type=str, default=PATH['DEFAULT_TRAIN_CONFIG'],
                    help='Path to config file with default optional argument values')

# Value arguments (default values not set here are set by the default.json configs file)
parser.add_argument('--dataset', '-ds', type=str, nargs='?', choices=['CharacteristicsDataset'], help='Dataset name')
parser.add_argument('--data_path', '-d', type=str, nargs='?', help='Path to data folder')
parser.add_argument('--dataset_split_type', '-dst', type=str, nargs='?',
                    help='Method to split the dataset into train/val/test sets')
# parser.add_argument('--dataset_split_path', '-dsp', type=str, nargs='?', help='Path to file with dataset split indices')
parser.add_argument('--outputs_path', '-o', type=str, nargs='?', help='Path to outputs folder')
parser.add_argument('--model', '-m', type=str, nargs='?', choices=['RegressionModel'], help='Model name')
parser.add_argument('--checkpoint_path', '-w', type=str, nargs='?', help='Path to model checkpoint')
parser.add_argument('--batch', '-b', type=int, nargs='?', help='Batch size')
parser.add_argument('--loss', type=str, nargs='?', choices=['MSE', 'MAE', 'BCE', 'CE'], help='Loss function')
parser.add_argument('--epochs', '-e', type=int, nargs='?', help='Number of epochs')
parser.add_argument('--learning-rate', '-lr', type=float, nargs='?', help='Learning rate')
parser.add_argument('--optimizer', '-opt', type=str, nargs='?', choices=['SGD'], help='Optimizer')
parser.add_argument('--momentum', type=float, nargs='?', help='Momentum')
parser.add_argument('--weight_decay', type=float, nargs='?', help='Weight decay')
parser.add_argument('--scheduler', '-sch', type=str, nargs='?', choices=[None, 'None', 'StepLR'],
                    help='Learning rate scheduler')
parser.add_argument('--folds', '-f', type=int, nargs='?', help='Number of folds for cross validation')
parser.add_argument('--patience', type=int, nargs='?', help='Patience for early stopping')  # TODO Early stopping
parser.add_argument('--device', type=str, nargs='?', default="cpu", choices=['cpu', 'cuda'],
                    help='Device type to use (cpu, cuda)')
parser.add_argument('--gpus', type=int, nargs="*", help='GPU devices to use')
parser.add_argument('--seed', '-s', type=int, nargs='?', help='Random seed')

# Flag arguments
parser.add_argument('--resume', '-r', action='store_true', default=False, help='Resume from checkpoint')
parser.add_argument('--save', '-S', action='store_true', default=True, help='Save checkpoint')
parser.add_argument('--tensorboard', action='store_true', default=False, help='Enable tensorboard logging')
parser.add_argument('--neptune', action='store_true', default=False, help='Enable neptune logging')

# Parse and validate the arguments
args = vars(parser.parse_args())
args = update_missing_args(args)  # Update args with default values from config file
args = validate_args(args, parser)  # Validate args from config file

# INITIALIZATION -------------------------------------------------------------------------------------------------------

tqdm.write('')

# Initialize setup
if args['resume'] and args['checkpoint_path']:  # If resuming run from checkpoint
    args = read_config(os.path.join(args['checkpoint_path'], 'config.json'))  # Load args from checkpoint config file
    tqdm.write(f"Resuming run: {args['checkpoint_path'].split(os.sep)[-3]}. "
               f"Script arguments will be ignored and loaded from checkpoint configuration file.")
run_name = gen_run_name(args)  # Generate run name using args
tqdm.write(f'Run name: {run_name}\n')
tqdm.write('Run arguments:')
for arg, value in args.items():
    tqdm.write(f'  {arg}: {value}')
set_seed(args['seed'])  # Set seed
make_project_dirs()  # Make all essential directories

# Load dataset, split into subsets and generate loaders
dataset = select_dataset(args['dataset'], args)
train_set, val_set, test_set = split_dataset(dataset, val_ratio=0.2, test_ratio=0.2, split_type='random',
                                             seed=args['seed'])
train_loader = DataLoader(train_set, batch_size=args['batch'], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=args['batch'], shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=args['batch'], shuffle=False, collate_fn=collate_fn)

# Load model, optimizer and scheduler
model = select_model(args['model'], dataset)
optimizer = select_optimizer(args['optimizer'], model.parameters(), args)
scheduler = select_scheduler(args['scheduler'], optimizer, args)
initial_epoch = 0
if args['checkpoint_path']:  # Load state from checkpoint
    tqdm.write('Loading checkpoint from ' + args['checkpoint_path'])
    model, optimizer, scheduler, initial_epoch, start_fold = load_checkpoint(args['checkpoint_path'], model, optimizer,
                                                                             scheduler,
                                                                             args['resume'])

# Initialize logging
neptune_log = None
tensorboard_log = None
suppress_logger_info()  # Supress logger info messages
if args['neptune']:
    neptune_log = init_neptune_logger(run_name, initial_epoch, args)
if args['tensorboard']:
    tensorboard_log = init_tensorboard_logger(run_name, args)

# Define loss function
criterion = select_loss(args['loss'])

# Define evaluation metrics
metrics = {}  # {'auroc': AUROC}

# TRAINING/VALIDATION LOOP -------------------------------------------------------------------------------------------

tqdm.write('Training...')

# Initialize pipeline modules
progress_bar = tqdm(total=(args['epochs'] - initial_epoch) * (len(train_loader) + len(val_loader)),
                    initial=initial_epoch)
validator = Validator(metrics, progress_bar, args['epochs'])
trainer = Trainer(criterion, validator, args['epochs'], run_name,
                  os.path.join(args['outputs_path'], 'checkpoints', args['model'], args['dataset']), progress_bar,
                  args['neptune'], neptune_log, args['tensorboard'], tensorboard_log)

# Train + Validate
results = trainer.train(model, optimizer, train_loader, val_loader, scheduler, initial_epoch)

# Close loggers
if args['neptune']:
    neptune_log.stop()
if args['tensorboard']:
    tensorboard_log.close()
