import numpy as np
from torch.utils.data.dataset import Subset

def split_dataset(dataset, ratio, random=True):
    """
    Function for splitting the dataset into training and test sets.
    :param dataset: Dataset object to split
    :param ratio: Ratio of dataset to use for training
    :param random: Whether to choose random or consecutive samples for the training and test sets or
    :return: The Subset objects for the training and test sets.
    """
    n_train = int(ratio * len(dataset))

    if random:
        train_idx = np.random.choice(range(len(dataset)), n_train, replace=False)
    else:
        train_idx = list(range(n_train))

    test_idx = list(set(range(len(dataset))) - set(train_idx))

    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)

    return train_set, test_set



