import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from torchmetrics.classification import AUROC

from src.data.CIFAR10 import CIFAR10  # Seems unused but it actually isn't
from src.models.CIFARModel import CIFARModel  # Seems unused but it actually isn't
from src.modules.Evaluator import Validator
from src.modules.Trainer import Trainer
from src.utils.logging import suppress_logger_info, init_neptune_logger, init_tensorboard_logger
from src.utils.path import PATH, make_project_dirs
from src.utils.arg import validate_args, update_missing_args
from src.utils.data import split_dataset, collate_fn
from src.utils.model import load_checkpoint
from src.utils.optimizer import select_optimizer
from src.utils.scheduler import select_scheduler
from src.utils.helper import gen_run_name, set_seed

# PARSE ARGUMENTS ------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Pytorch model training script')

# Config argument
parser.add_argument('--config', type=str, default=PATH['DEFAULT_TRAIN_CONFIG'],
                    help='Path to config file with default optional argument values')

# Value arguments (default values not set here are set by the default.json configs file)
parser.add_argument('--dataset', '-D', type=str, nargs='?', choices=['CIFAR10'], help='Dataset name')
parser.add_argument('--data', '-d', type=str, nargs='?', help='Path to data folder')
parser.add_argument('--outputs', '-o', type=str, nargs='?', help='Path to outputs folder')
parser.add_argument('--model', '-M', type=str, nargs='?', choices=['CIFARModel'], help='Model name')
parser.add_argument('--weights', '-w', type=str, nargs='?', help='Path to model checkpoint')
parser.add_argument('--batch', '-b', type=int, nargs='?', help='Batch size')
parser.add_argument('--epochs', '-e', type=int, nargs='?', help='Number of epochs')
parser.add_argument('--learning-rate', '-lr', type=float, nargs='?', help='Learning rate')
parser.add_argument('--optimizer', '-opt', type=str, nargs='?', choices=['SGD'], help='Optimizer')
parser.add_argument('--momentum', type=float, nargs='?', help='Momentum')
parser.add_argument('--weight_decay', type=float, nargs='?', help='Weight decay')
# TODO: Make scheduler optional
parser.add_argument('--scheduler', '-sch', type=str, nargs='?', choices=[None, 'None', 'StepLR'],
                    help='Learning rate scheduler')
parser.add_argument('--patience', type=int, nargs='?', help='Patience for early stopping')
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

# Initialize setup
set_seed(args['seed'])  # Set seed
make_project_dirs()  # Make all essential directories
run_name = gen_run_name(args)  # Generate run name using args

print('Run name: ', run_name)

# Load dataset, split into subsets and generate loaders
dataset = globals()[args['dataset']](args['data'])
train_set, val_set, test_set = split_dataset(dataset, val_ratio=0.2, test_ratio=0.2, random=True)
train_loader = DataLoader(train_set, batch_size=args['batch'], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=args['batch'], shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=args['batch'], shuffle=False, collate_fn=collate_fn)

# Load model, optimizer and scheduler
model = globals()[args['model']]()
optimizer = select_optimizer(args['optimizer'], model.parameters(), args)
scheduler = select_scheduler(args['scheduler'], optimizer, args)
start_epoch = 0
if args['weights']:  # Load state from checkpoint
    print('Loading checkpoint from ', args['weights'])
    model, optimizer, scheduler, start_epoch = load_checkpoint(args['weights'], model, optimizer, scheduler,
                                                               args['resume'])

# Initialize logging
neptune_log = None
tensorboard_log = None
suppress_logger_info()  # Supress logger info messages
if args['neptune']:
    neptune_log = init_neptune_logger(run_name, start_epoch, args)
if args['tensorboard']:
    tensorboard_log = init_tensorboard_logger(run_name, args)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define evaluation metrics
metrics = {'auc': AUROC(task="multiclass", num_classes=dataset.num_classes)}

# TRAINING/VALIDATION LOOP -------------------------------------------------------------------------------------------

print('Training...')

progress_bar = tqdm(total=args['epochs'] * (len(train_loader) + len(val_loader)), initial=start_epoch)

# Initialize training modules
validator = Validator(metrics, progress_bar, args['epochs'])
trainer = Trainer(criterion, validator, args['epochs'], run_name,
                  os.path.join(args['outputs'], 'checkpoints', args['model'], args['dataset']), progress_bar,
                  args['neptune'], neptune_log, args['tensorboard'], tensorboard_log)

# Train + Validate
trainer.train(model, optimizer, train_loader, val_loader, scheduler)

# Close loggers
if args['neptune']:
    neptune_log.stop()
if args['tensorboard']:
    tensorboard_log.close()
