import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

from src.data.CIFAR10 import CIFAR10


def select_dataset(name, args):
    if name == 'CIFAR10':
        return CIFAR10(args['data'])
    else:
        raise ValueError('Unknown dataset')


def split_dataset_in_two_sets(dataset, ratio, split_type, seed):
    """
    Splits the dataset into two sets.
    :param dataset: Dataset object to split.
    :param ratio: Ratio of dataset to use for the second set.
    :param split_type: Type of splitting strategy to use.
                        If 'random', we use random samples for each split.
                        If 'stratify', we take use class-balanced splits.
                        If 'consecutive', we take consecutive samples for each split.
    :param seed: Random seed for splitting.
    :return: The Subset objects for the two sets.
    """
    n_second = int(ratio * len(dataset))
    n_first = len(dataset) - n_second

    if split_type == 'random':
        first_idx = np.random.choice(range(len(dataset)), n_first, replace=False)
        second_idx = list(set(range(len(dataset))) - set(first_idx))
    elif split_type == 'stratify':
        labels = [dataset.labels[image] for image in dataset.images]
        first_idx, second_idx = train_test_split(range(len(dataset)), test_size=ratio,
                                                 stratify=labels,
                                                 random_state=seed)
    elif split_type == 'consecutive':
        first_idx = list(range(n_first))
        second_idx = list(range(n_first, n_first + n_second))
    else:
        raise ValueError('Invalid split type. Please choose from "random", "stratify", or "consecutive"')

    first_set = Subset(dataset, first_idx)
    second_set = Subset(dataset, second_idx)

    return first_set, second_set


def split_dataset(dataset, val_ratio, test_ratio, split_type='consecutive', seed=42, train_val_folds=1, train_idx=None,
                  val_idx=None, test_idx=None):
    """
    Splits the dataset into training and test sets.
    :param dataset: Dataset object to split.
    :param val_ratio: Ratio of dataset to use for validaton test.
    :param test_ratio: Ratio of dataset to use for test set.
    :param split_type: Type of splitting strategy to use.
                        If 'random', we use random samples for each split.
                        If 'stratify', we take use class-balanced splits.
                        If 'consecutive', we take consecutive samples for each split.
                        If 'idx', we take samples based on given ids.
    :param seed: Random seed for splitting.
    :param train_val_folds: Number of folds for cross-validation on the train+val set. If 1, no cross-validation is performed.
    :param train_idx: List or (Dictionary of Lists, when multiple folds) of indices for training set (to be used with split_type='ids').
    :param val_idx: List or (Dictionary of Lists, when multiple folds) of indices for validation set (to be used with split_type='ids').
    :param test_idx: List of indices for test set (to be used with split_type='ids').
    :return: The Subset objects for the training and test sets.
    """

    # Initialize
    train_set = {}  # Dictionary (for every fold)
    val_set = {}

    train_val_set, test_set = split_dataset_in_two_sets(dataset, test_ratio, split_type, seed)

    if split_type != 'idx':  # If not split from ids, then split using the given strategy
        if train_val_folds > 1:
            for fold in range(train_val_folds):
                train_set[fold], val_set[fold] = split_dataset_in_two_sets(train_val_set, val_ratio / (1 - test_ratio),
                                                                           split_type, seed)
        else:
            train_set, val_set = split_dataset_in_two_sets(train_val_set, val_ratio / (1 - test_ratio), split_type,
                                                           seed)
    else:  # If split from ids
        if train_val_folds > 1:
            for fold in range(train_val_folds):
                if train_idx is None or val_idx is None or test_idx is None:
                    raise ValueError('When using split_type="ids", train_ids, val_ids and test_ids must be provided.')
                train_set[fold] = Subset(dataset, train_idx[fold])
                val_set[fold] = Subset(dataset, val_idx[fold])
            test_set = Subset(dataset, test_idx)

    # Assert whole dataset is used
    if train_val_folds == 1:
        assert len(train_set) + len(val_set) + len(test_set) == len(dataset)
    else:
        assert len(train_set[0]) + len(val_set[0]) + len(test_set) == len(dataset)

    return train_set, val_set, test_set


def collate_fn(batch):
    """
    Collates a list of (image, label) pairs into a batch consisting of two tensors: images, labels.
    :param batch: The list of (image, label) pairs.
    :return: The two tensors: images and labels.
    """
    # Unzip
    images, labels = zip(*batch)
    images = list(images)
    labels = list(labels)

    # Transform each image to tensor
    transform = transforms.ToTensor()
    images = [transform(image) for image in images]

    # Stack tensors/elements into a batch tensor
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels)
    return images, labels
