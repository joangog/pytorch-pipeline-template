import os
import torch

from src.models.CIFARModel import CIFARModel


def select_model(name, dataset):
    if name == 'CIFARModel':
        return CIFARModel(n_classes=dataset.n_labels)
    else:
        raise ValueError('Unknown model')


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, resume=False, cross_validation=False):
    """
    Loads model, optimizer (optional) and scheduler (optional) state from given checkpoint path.
    :param checkpoint_path: The checkpoint path.
    :param model: The model instance.
    :param optimizer: The optimizer instance (to be used with resume=True).
    :param scheduler: The scheduler instance (to be used with resume=True).
    :param resume: Flag to resume training from last epoch, which requires an optimizer and scheduler as input.
    :param cross_validation: Flag to determine whether cross validation is being performed or not.
    :return: The updated model, optimizer, and scheduler state, and the epoch training will start from.
    """
    # TODO: Check edge case where we resume from checkpoint but change the learning rate or another hyperparam
    # In that case not only we would need to load the model/opt/sched, but also update the args with the loaded hyperparameters otherwise the object hyps and the arg hyps are conflicting
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights from checkpoint
    initial_epoch = 0
    initial_fold = 0 if cross_validation else None
    if resume:  # If resuming training from last epoch, then load optimizer and scheduler states
        initial_epoch = checkpoint[
                            'epoch'] + 1  # If the checkpoint for that epoch is saved, it means that epoch is complete
        # TODO: fix fold resuming, because if last fold what do you do, if fold all what do you do
        initial_fold = checkpoint[
            'fold'] if cross_validation else None  # If the checkpoint for the last epoch of that fold is saved, then the next fold is iterated automatically because we will be at last_epoch1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return model, optimizer, scheduler, initial_epoch, initial_fold


def load_weights(checkpoint_path, model):
    """
    Loads model weights from given checkpoint path. Useful for loading weights for inference.
    :param checkpoint_path: The checkpoint path.
    :param model: The model instance.
    :return: The updated model.
    """
    model, _, _, _ = load_checkpoint(checkpoint_path, model)
    return model


def save_checkpoint(save_path, run_name, model, optimizer, scheduler, epoch, train_loss, val_loss=None, fold=None):
    """
    Saves training checkpoint in a corresponding folder inside the outputs path.
    :param save_path: The path to the directory with all saved checkpoints.
    :param run_name: The run name, it is used for the directory with the checkpoints at each epoch for the current run.
    :param model: The model instance.
    :param optimizer: The optimizer instance.
    :param scheduler: The scheduler instance.
    :param epoch: The current epoch number.
    :param train_loss: The current train loss.
    :param val_loss: The current val loss.
    :param fold: The current fold for cross validation.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'fold': fold,
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    run_save_path = os.path.join(save_path, run_name, f'fold_{fold}' if fold else '')
    os.makedirs(run_save_path, exist_ok=True)
    torch.save(checkpoint, os.path.join(run_save_path, f'epoch_{epoch}.pth'))
