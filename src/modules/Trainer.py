import os
import torch
from src.utils.model import save_checkpoint
from src.utils.arg import save_config


class Trainer(object):
    """
    A class for training the model.
    """

    def __init__(self, criterion, validator, epochs, run_name, run_args, save_checkpoint_path, device=None):
        """
        :param criterion: The loss criterion instance.
        :param validator: A Validator instance for validating during training

        :param epochs: The total number of epochs.
        :param run_name: The name of the run.
        :param run_args: The arguments of the run script.
        :param save_checkpoint_path: The path to the directory of saved checkpoints for the specific model and dataset.
        :param device: The device to use.
        """
        self.criterion = criterion
        self.validator = validator
        self.epochs = epochs
        self.run_name = run_name
        self.run_args = run_args
        self.save_checkpoint_path = save_checkpoint_path
        self.device = torch.device(device) if device else torch.device('cpu')

        # Add loss criterion as a validation metric by default
        self.validator.metrics['loss'] = self.criterion

    def train_step(self, model, optimizer, batch, tqdm_progress_bar=None, tqdm_desc=None):
        """
        Trains the model on one batch.
        :param model:  The model instance.
        :param optimizer: The optimizer instance.
        :param batch: A batch of images and labels.
        :param tqdm_desc: A description string for the tqdm progress bar.
        :return: The loss.
        """
        image, label = batch
        image, label = image.to(self.device), label.to(self.device)

        optimizer.zero_grad()

        # Get predictions
        output = model(image)

        # Get loss and update model
        loss = self.criterion(output, label)
        loss.backward()
        optimizer.step()

        # Update progress bar
        if tqdm_progress_bar:
            if tqdm_desc:
                tqdm_progress_bar.desc = tqdm_desc
            tqdm_progress_bar.update(1)

        return loss.item()

    def train_epoch(self, model, optimizer, train_loader, epoch, scheduler=None, tqdm_progress_bar=None):
        """
        Trains the model for one epoch on the dataset.
        :param model: The model instance.
        :param optimizer: The optimizer instance.
        :param train_loader: The training data loader.
        :param epoch: The current epoch.
        :param scheduler: The scheduler instance.
        :return: The average loss.
        """
        model.train()  # Set model to train mode
        loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            tqdm_desc = (f"Epoch {epoch} ({epoch + 1}/{self.epochs}): "
                         f"Training Batch {batch_idx} ({batch_idx + 1}/{len(train_loader)})")
            loss += self.train_step(model, optimizer, batch, tqdm_progress_bar, tqdm_desc)

        # Average loss
        loss = loss / len(train_loader)

        return loss

    def train(self, model, optimizer, train_loader, val_loader=None, scheduler=None, initial_epoch=0, fold=None,
              tqdm_progress_bar=None, neptune=False, neptune_log=None, tensorboard=False, tensorboard_log=None):
        """
        Performs full training + validation loop.
        :param model: The model instance.
        :param optimizer: The optimizer instance.
        :param train_loader: The training data loader or a list of training data loaders for cross validation.
        :param val_loader: The validation data loader, or a list of validation data loaders for cross validations.
        :param scheduler: The scheduler instance.
        :param initial_epoch: The initial epoch to start from (can be other than 0 when resuming).
        :param fold: The current fold for cross-validation.
        :param tqdm_progress_bar: A tqdm progress bar instance.
        :param neptune: If True, neptune logging is enabled.
        :param neptune_log: A dictionary of neptune logging parameters.
        :param tensorboard: If True, tensorboard logging is enabled.
        :param tensorboard_log: A dictionary of tensorboard logging parameters.
        :return:
        """

        val_metrics = dict()

        # Epoch Loop
        for epoch in range(min(initial_epoch, self.epochs), self.epochs):

            # Train
            train_loss = self.train_epoch(model, optimizer, train_loader, epoch, scheduler, tqdm_progress_bar)

            # Validate
            if val_loader:  # TODO: Implement and test how it works without validation
                val_metrics = self.validator.evaluate(model, val_loader, epoch, tqdm_progress_bar)
            val_loss = val_metrics.pop('loss', None)  # Extract val loss from val metrics

            # Save checkpoint
            save_checkpoint(self.save_checkpoint_path, self.run_name, model, optimizer, scheduler, epoch,
                            train_loss, val_loss, fold)

            # Save run configs
            save_config(self.run_args, os.path.join(self.save_checkpoint_path, self.run_name, 'config.json'))

            # Log
            tqdm_progress_bar.set_postfix(train_loss=train_loss, val_loss=val_loss,
                                          **{f'val_{metric}': metric_value for metric, metric_value in
                                             val_metrics.items()})
            if neptune:
                neptune_log['learning_rate'].log(optimizer.param_groups[0]['lr'])
                neptune_log['loss/train'].log(train_loss)
                if val_loss:
                    neptune_log['loss/val'].log(val_loss)
                for val_metric_name, val_metric_value in val_metrics.items():
                    neptune_log[f'{val_metric_name}/val'].log(val_metric_value)
            if tensorboard:
                tensorboard_log.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                if val_loss:
                    tensorboard_log.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
                else:
                    tensorboard_log.add_scalars('loss', {'train': train_loss}, epoch)
                for val_metric_name, val_metric_value in val_metrics.items():
                    tensorboard_log.add_scalar(f'{val_metric_name}/val', val_metric_value, epoch)

            # Update scheduler
            if scheduler:
                scheduler.step()  # Note that the scheduler is stepped every epoch, not every batch

        return model, val_metrics
