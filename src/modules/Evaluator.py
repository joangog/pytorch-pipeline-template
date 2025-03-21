import torch


class Evaluator(object):
    """
    A class for evaluating the model using the given metrics.
    """

    def __init__(self, metrics, tqdm_progress_bar=None):
        """
        :param metrics: A dictionary of torch metric instances.
        :param tqdm_progress_bar: A tqdm progress bar instance.
        """
        self.metrics = metrics
        self.tqdm_progress_bar = tqdm_progress_bar

    def evaluate_step(self, model, batch, tqdm_desc=None):
        """
        Evaluates the model on one batch.
        :param model:  The model instance.
        :param batch: A batch of images and labels.
        :param tqdm_desc: A description string for the tqdm progress bar.
        :return: A dictionary with the metric values.
        """
        image, label = batch

        # Get predictions
        output = model(image)

        # Evaluate using metrics
        metric_vals = dict()
        for metric_name, metric_func in self.metrics.items():
            metric_vals[metric_name] = metric_func(output, label).item()

        # Update progress bar
        if self.tqdm_progress_bar:
            if tqdm_desc:
                self.tqdm_progress_bar.desc = tqdm_desc
            self.tqdm_progress_bar.update(1)

        return metric_vals

    def evaluate(self, model, data_loader, tqdm_desc_func=None):
        """
        Evaluates the model on the dataset.
        :param model: The model instance.
        :param data_loader: The data loader.
        :param tqdm_desc_func: A function for generating a description string for the tqdm progress bar.
        :return: A dictionary for storing the metric values.
        """
        if not tqdm_desc_func:  # Initialize tqdm description function
            tqdm_desc_func = lambda batch_idx: self.tqdm_progress_bar.desc

        model.eval()  # Set model to evaluation mode
        metric_vals = {metric: 0.0 for metric in self.metrics.keys()}  # Initialize metric values

        # Evaluate
        with torch.no_grad():  # No gradients needed during evaluation
            for batch_idx, batch in enumerate(data_loader):
                batch_metric_vals = self.evaluate_step(model, batch, tqdm_desc_func(batch_idx))
                for metric in self.metrics.keys():
                    metric_vals[metric] += batch_metric_vals[metric]

        # Average metrics
        for metric_name, metric_fun in self.metrics.items():
            metric_vals[metric_name] = metric_vals[metric_name] / len(data_loader)

        return metric_vals


class Validator(Evaluator):
    """
    An Evaluator class specifically for validation during learning.

    """

    def __init__(self, metrics, tqdm_progress_bar=None, total_epochs=None):
        """
        :param metrics: The metric instances.
        :param tqdm_progress_bar: A tqdm progress bar instance.
        :param total_epochs: The total number of epochs.
        """
        super(Validator, self).__init__(metrics, tqdm_progress_bar)
        self.total_epochs = total_epochs

    def evaluate(self, model, val_loader, epoch):
        """
        Evaluates the model on the whole dataset.
        :param model: The model instance.
        :param val_loader: The validation data loader.
        :param epoch: The current epoch.
        :return: A dictionary with the average metric values.
        """
        # tqdm_desc_func is a function for generating a description string for the tqdm progress bar
        tqdm_desc_func = lambda epoch, batch_idx: (f"Epoch {epoch} ({epoch + 1}/{self.total_epochs}), Validation, "
                                                   f"Batch {batch_idx} ({batch_idx + 1}/{len(val_loader)})")
        metric_vals = super().evaluate(model, val_loader, lambda batch_idx: tqdm_desc_func(epoch, batch_idx))

        return metric_vals


class Tester(Evaluator):
    """
    An Evaluator class specifically for testing.

    """

    def __init__(self, metrics, tqdm_progress_bar=None):
        """
        :param metrics: The metric instances.
        :param tqdm_progress_bar: A tqdm progress bar instance.
        """
        super(Tester, self).__init__(metrics, tqdm_progress_bar)

    def evaluate(self, model, test_loader):
        """
        Evaluates the model on the whole dataset.
        :param model: The model instance.
        :param test_loader: The testing data loader.
        :return: A dictionary for storing the metric values.
        """
        # tqdm_desc_func is a function for generating a description string for the tqdm progress bar
        tqdm_desc_func = lambda batch_idx: f"Batch {batch_idx + 1}/{len(test_loader)}"
        metric_vals = super().evaluate(model, test_loader, tqdm_desc_func)

        return metric_vals
