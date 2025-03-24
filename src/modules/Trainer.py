from tqdm import tqdm

from src.utils.model import save_checkpoint


class Trainer(object):
    """
    A class for training the model.
    """

    def __init__(self, criterion, validator, epochs, run_name, save_checkpoint_path, tqdm_progress_bar=None,
                 neptune=False, neptune_log=None, tensorboard=False, tensorboard_log=None, ):
        """
        :param criterion: The loss criterion instance.
        :param validator: A Validator instance for validating during training
        :param tqdm_progress_bar: A tqdm progress bar instance.
        :param epochs: The total number of epochs.
        :param run_name: The name of the run.
        :param save_checkpoint_path: The path to the directory of saved checkpoints for the specific model and dataset.
        ;param tqdm_progress_bar: A tqdm progress bar instance.
        :param neptune: If True, neptune logging is enabled.
        :param neptune_log: A dictionary of neptune logging parameters.
        :param tensorboard: If True, tensorboard logging is enabled.
        :param tensorboard_log: A dictionary of tensorboard logging parameters.
        """
        self.criterion = criterion
        self.validator = validator
        self.tqdm_progress_bar = tqdm_progress_bar
        self.epochs = epochs
        self.run_name = run_name
        self.save_checkpoint_path = save_checkpoint_path
        self.tqdm_progress_bar = tqdm_progress_bar
        self.neptune = neptune
        self.neptune_log = neptune_log
        self.tensorboard = tensorboard
        self.tensorboard_log = tensorboard_log

        # Add loss criterion as a validation metric by default
        self.validator.metrics['loss'] = self.criterion

    def train_step(self, model, optimizer, batch, scheduler=None, tqdm_desc=None):
        """
        Trains the model on one batch.
        :param model:  The model instance.
        :param optimizer: The optimizer instance.
        :param batch: A batch of images and labels.
        :param scheduler: The scheduler instance.
        :param tqdm_desc: A description string for the tqdm progress bar.
        :return: The loss.
        """
        image, label = batch

        optimizer.zero_grad()

        # Get predictions
        output = model(image)

        # Get loss and update model
        loss = self.criterion(output, label)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Update progress bar
        if self.tqdm_progress_bar:
            if tqdm_desc:
                self.tqdm_progress_bar.desc = tqdm_desc
            self.tqdm_progress_bar.update(1)

        return loss.item()

    def train_epoch(self, model, optimizer, train_loader, epoch, scheduler=None):
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
            loss += self.train_step(model, optimizer, batch, scheduler, tqdm_desc)

        # Average loss
        loss = loss / len(train_loader)

        return loss

    def train(self, model, optimizer, train_loader, val_loader=None, scheduler=None, initial_epoch=0, fold=None):
        """
        Performs full training + validation loop.
        :param model: The model instance.
        :param optimizer: The optimizer instance.
        :param train_loader: The training data loader or a list of training data loaders for cross validation.
        :param val_loader: The validation data loader, or a list of validation data loaders for cross validations.
        :param scheduler: The scheduler instance.
        :param fold: The current fold for cross-validation.
        :param initial_epoch: The initial epoch to start from (can be other than 0 when resuming).
        :return:
        """

        val_metrics = dict()

        # Epoch Loop
        for epoch in range(min(initial_epoch, self.epochs), self.epochs):

            # Train
            train_loss = self.train_epoch(model, optimizer, train_loader, epoch, scheduler)

            # Validate
            if val_loader:  # TODO: Implement and test how it works without validation
                val_metrics = self.validator.evaluate(model, val_loader, epoch)
            val_loss = val_metrics.pop('loss', None)  # Extract val loss from val metrics

            # Save weights
            save_checkpoint(self.save_checkpoint_path, self.run_name, model, optimizer, scheduler, epoch,
                            train_loss, val_loss, fold)
            # Log
            self.tqdm_progress_bar.set_postfix(train_loss=train_loss, val_loss=val_loss,
                                               **{f'val_{metric}': metric_value for metric, metric_value in
                                                  val_metrics.items()})
            if self.neptune:
                self.neptune_log['learning_rate'].log(optimizer.param_groups[0]['lr'])
                self.neptune_log['loss/train'].log(train_loss)
                self.neptune_log['loss/val'].log(val_loss)
                for val_metric_name, val_metric_value in val_metrics.items():
                    self.neptune_log[f'{val_metric_name}/val'].log(val_metric_value)
            if self.tensorboard:
                self.tensorboard_log.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                self.tensorboard_log.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
                for val_metric_name, val_metric_value in val_metrics.items():
                    self.tensorboard_log.add_scalar(f'{val_metric_name}/val', val_metric_value, epoch)

        return model, val_metrics

    def train_folds(self, model, optimizer, train_loaders, val_loaders, scheduler=None, initial_epoch=0,
                    initial_fold=0):
        """
        Performs full training + validation loop for each fold using cross validation.
        :param model: The model instance.
        :param optimizer: The optimizer instance.
        :param train_loaders: A list of training data loaders for each fold.
        :param val_loaders: A list of validation data loaders for each fold.
        :param scheduler: The scheduler instance.
        :return:
        """
        if len(train_loaders) != len(val_loaders):  # If both are lists but not the same length
            raise ValueError(
                'For cross-validation, the train_loader and val_loader must both have the same length (as big as the number of folds).')

        # Cross-validation Loop
        folds = len(train_loaders)
        for fold_i in range(min(initial_fold, folds), folds):
            tqdm.write(f'Fold {fold_i}')

            train_loader = train_loaders[fold_i]
            val_loader = val_loaders[fold_i]

            self.train(model, optimizer, train_loader, val_loader, scheduler, initial_epoch, fold_i)

        # After everything is done retrain using train+val as all data and save in fold_all folder
        # Also print average train loss val loss and metrics (because when we train on all we dont have an evaluation dataset, only perhaps  test dataset)
        # Maybe also summarize all fold metrics
