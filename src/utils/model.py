import os
import torch


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, resume=False):
    """
    Loads model, optimizer (optional) and scheduler (optional) state from given checkpoint path.
    :param checkpoint_path: The checkpoint path.
    :param model: The model instance.
    :param optimizer: The optimizer instance (to be used with resume=True).
    :param scheduler: The scheduler instance (to be used with resume=True).
    :param resume: Flag to resume training from last epoch, which requires an optimizer and scheduler as input.
    :return: The updated model, optimizer, and scheduler state, and the epoch training will start from.
    """
    # TODO: Check edge case where we resume from checkpoint but change the learning rate or another hyperparam
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
    start_epoch = 0
    if resume:  # If resuming training from last epoch, then load optimizer and scheduler states
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return model, optimizer, scheduler, start_epoch


def load_weights(checkpoint_path, model):
    """
    Loads model state from given checkpoint path. Useful for loading weights for inference.
    :param checkpoint_path: The checkpoint path.
    :param model: The model instance.
    :return: The updated model.
    """
    model, _, _, _ = load_checkpoint(checkpoint_path, model)
    return model


def save_checkpoint(outputs_path, run_name, model, optimizer, scheduler, epoch, train_loss, val_loss, args):
    """
    Saves training checkpoint in a corresponding folder inside the outputs path.
    :param outputs_path: The path to the folder that contains all script outputs.
    :param run_name: The run name.
    :param model: The model instance.
    :param optimizer: The optimizer instance.
    :param scheduler: The scheduler instance.
    :param epoch: The current epoch number.
    :param train_loss: The current train loss.
    :param val_loss: The current val loss.
    :param args: The arguments passed to the script.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    save_path = os.path.join(outputs_path, 'checkpoints', args['model'], args['dataset'], run_name)
    os.makedirs(save_path, exist_ok=True)
    torch.save(checkpoint, os.path.join(save_path, f'epoch_{epoch}.pth'))
