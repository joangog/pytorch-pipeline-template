import argparse
import os
import neptune
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch import nn

from utils.args import validate_args, update_missing_args
from utils.data import split_dataset, collate_fn
from utils.models import load_checkpoint
from data.CIFAR10 import CIFAR10
from models.CIFARModel import CIFARModel

# PARSE ARGUMENTS ------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Pytorch model training script')

# Config argument
default_configs = os.path.join(os.getcwd(), 'configs', 'train', 'default.json')
parser.add_argument('--config', type=str, default=default_configs,
                    help='Config file with default optional argument values')

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

# Start neptune logging
with open('neptune-config.json', 'r') as file:
    neptune_config = json.load(file)

# Log hyperparameters on neptune
log = None
if args['neptune']:
    log = neptune.init_run(project=neptune_config['project'], api_token=neptune_config['api_token'])
    log['parameters'] = {'batch': args['batch'], 'epochs': args['epochs'], 'learning_rate': args['learning_rate'],
                         'momentum': args['momentum'], 'weight_decay': args['weight_decay'],
                         'optimizer': type(optimizer).__name__}

# Define loss function
criterion = nn.CrossEntropyLoss()

# TRAINING/VALIDATION LOOP -------------------------------------------------------------------------------------------

for epoch in tqdm(range(args['epochs'])):

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
    avg_train_loss = train_loss / len(train_loader)
    if args['neptune']:
        log['train/learning_rate'].log(optimizer.param_groups[0]['lr'])
        log['train/loss'].log(avg_train_loss)

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():  # No gradients needed during validation
        for image, label in val_loader:
            output = model(image)
            loss = criterion(output, label)
            val_loss += loss.item()
    val_loss = val_loss / len(train_loader)
    if args['neptune']:
        log['val/loss'].log(val_loss)

log.stop()

print('Done')
