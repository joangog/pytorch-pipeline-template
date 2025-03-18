import argparse
import os
import json
from tqdm import tqdm
from tqdm.contrib import itertools
import neptune
import tensorboard as tb

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch import nn

from src.utils.paths import NEPTUNE_CONFIG_PATH, TRAIN_CONFIG_PATH, TENSORBOARD_LOGS_PATH
from src.utils.args import validate_args, update_missing_args
from src.utils.data import split_dataset, collate_fn
from src.utils.models import load_checkpoint
from src.data.CIFAR10 import CIFAR10
from src.models.CIFARModel import CIFARModel

# PARSE ARGUMENTS ------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Pytorch model training script')

# Config argument
parser.add_argument('--config', type=str, default=TRAIN_CONFIG_PATH,
                    help='Path to config file with default optional argument values')

# Value arguments (default values set by config file)
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
parser.add_argument('--device', type=str, nargs='?', default="cpu", choices=['cpu', 'cuda'],
                    help='Device type to use (cpu, cuda)')
parser.add_argument('--gpus', type=int, nargs="*", help='GPU devices to use')
parser.add_argument('--seed', type=int, nargs='?', help='Random seed')

# Flag arguments
parser.add_argument('--resume', '-r', action='store_true', default=False, help='Resume from checkpoint')
parser.add_argument('--tensorboard', action='store_true', default=False, help='Enable tensorboard logging')
parser.add_argument('--neptune', action='store_true', default=False, help='Enable neptune logging')

# Parse and validate the arguments
args = vars(parser.parse_args())
args = update_missing_args(args)  # Update args with default values from config file
validate_args(args, parser)

# INITIALIZATION -------------------------------------------------------------------------------------------------------

# Load dataset, split into subsets and generate loaders
dataset = CIFAR10(args['data'])
train_set, val_set, test_set = split_dataset(dataset, val_ratio=0.2, test_ratio=0.2, random=True)
train_loader = DataLoader(train_set, batch_size=args['batch'], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=args['batch'], shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=args['batch'], shuffle=False, collate_fn=collate_fn)

# Load model, optimizer and scheduler
model = CIFARModel()
optimizer = SGD(model.parameters(), lr=args['learning_rate'], momentum=args['momentum'],
                weight_decay=args['weight_decay'])
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
if args['weights']:  # Load model state
    load_checkpoint(model, optimizer, scheduler, args['weights'], args['resume'])

# Start logging
neptune_log = None
tensorboard_log = None
if args['neptune']:
    with open(NEPTUNE_CONFIG_PATH) as file:
        neptune_config = json.load(file)
    neptune_log = neptune.init_run(project=neptune_config['project'], api_token=neptune_config['api_token'])
    neptune_log['parameters'] = {'batch': args['batch'], 'epochs': args['epochs'],
                                 'learning_rate': args['learning_rate'],
                                 'momentum': args['momentum'], 'weight_decay': args['weight_decay'],
                                 'optimizer': type(optimizer).__name__}
if args['tensorboard']:
    run_name = 'run_e{}_b{}_lr{}_m{}_w{}_p{}_s{}'.format(args['epochs'], args['batch'], args['learning_rate'],
                                                         args['momentum'], args['weight_decay'], args['patience'],
                                                         args['seed'])
    run_path = os.path.join(TENSORBOARD_LOGS_PATH, 'train', args['dataset'], args['model'], run_name)
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    tensorboard_log = tb.SummaryWriter(os.path.join(run_path))

# Define loss function
criterion = nn.CrossEntropyLoss()

# TRAINING/VALIDATION LOOP -------------------------------------------------------------------------------------------

progress_bar = tqdm(total=args['epochs'] * (len(train_loader) + len(val_loader)))

for epoch in range(args['epochs']):

    # Train
    model.train()
    train_loss = 0.0
    for batch_idx, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar.desc = f"Epoch {epoch}/{args['epochs']}, Training, Batch {batch_idx}/{len(train_loader)}"
        progress_bar.update(1)
    avg_train_loss = train_loss / len(train_loader)

    # Log
    if args['neptune']:
        neptune_log['train/learning_rate'].log(optimizer.param_groups[0]['lr'])
        neptune_log['train/loss'].log(avg_train_loss)
    if args['tensorboard']:
        tensorboard_log.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        tensorboard_log.add_scalar('train/loss', avg_train_loss, epoch)

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():  # No gradients needed during validation
        for batch_idx, (image, label) in enumerate(val_loader):
            output = model(image)
            loss = criterion(output, label)
            val_loss += loss.item()
            progress_bar.desc = f"Epoch {epoch}/{args['epochs']}, Validation, Batch {batch_idx}/{len(val_loader)}"
            progress_bar.update(1)
    val_loss = val_loss / len(val_loader)

    # Log
    if args['neptune']:
        neptune_log['val/loss'].log(val_loss)
    if args['tensorboard']:
        tensorboard_log.add_scalar('val/loss', val_loss, epoch)

if args['neptune']:
    neptune_log.stop()
if args['tensorboard']:
    tensorboard_log.close()
