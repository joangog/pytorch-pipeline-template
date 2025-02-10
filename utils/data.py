import numpy as np
from torch.utils.data.dataset import Subset


def split_dataset(dataset, val_ratio, test_ratio, random=True):
    """
    Function for splitting the dataset into training and test sets.
    :param dataset: Dataset object to split
    :param ratio: Ratio of dataset to use for training
    :param random: Whether to choose random or consecutive samples for the subsets
    :return: The Subset objects for the training and test sets.
    """
    n_test = int(test_ratio * len(dataset))
    n_val = int(val_ratio * len(dataset))
    n_train = len(dataset) - n_test - n_val

    if random:
        train_idx = np.random.choice(range(len(dataset)), n_train, replace=False)
        val_idx = np.random.choice(list(set(range(len(dataset))) - set(train_idx)), n_val, replace=False)
    else:
        train_idx = list(range(n_train))
        val_idx = list(range(n_train, n_train + n_val))

    test_idx = list(set(range(len(dataset))) - set(train_idx) - set(val_idx))

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    return train_set, val_set, test_set


def collate_fn(batch):
    pass
