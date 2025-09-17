import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from src.data.CIFAR10 import CIFAR10
import nibabel as nib
from torch.utils.data import ConcatDataset


def sample_random_slice_window_from_nifti(nifti_path, slice_window_size):
    """
    Samples a random window of slices from a 3D NIfTI image efficiently using lazy loading.
    :param image_path: Path to the NIfTI image file.
    :param window_size: The number of slices in the window.
    :return: The sampled window of slices as a numpy array.
    :returns: spacing of the image
    """
    img = nib.load(nifti_path)
    n_slices = img.shape[2]
    from_idx = np.random.randint(0, n_slices - slice_window_size + 1)
    to_idx = from_idx + slice_window_size
    data = img.dataobj[:, :, from_idx:to_idx]  # Allows for loading only a specific range of slices efficiently
    spacing = img.header.get_zooms()[:3]  # Get the first three dimensions for spacing
    return data, spacing


def get_dataset(name, args):
    if name == 'CIFAR10':
        return CIFAR10(args['data_path'])
    else:
        raise ValueError('Unknown dataset')


def split_dataset_in_two_sets(dataset, ratio, split_type, seed):
    """
    Splits the dataset into two sets.
    :param dataset: Dataset object to split.
    :param ratio: Ratio of dataset to use for the second set.
    :param split_type: Type of splitting strategy to use.
                        If 'random', we use random samples for each split.
                        If 'stratified', we take use class-balanced splits.
                        If 'consecutive', we take consecutive samples for each split.
    :param seed: Random seed for splitting.
    :return: The Subset objects for the two sets.
    """
    n_second = int(ratio * len(dataset))
    n_first = len(dataset) - n_second

    if split_type == 'random':
        first_idx = np.random.choice(range(len(dataset)), n_first, replace=False)
        second_idx = list(set(range(len(dataset))) - set(first_idx))
    elif split_type == 'stratified':
        labels = [dataset.labels[image] for image in dataset.images]
        first_idx, second_idx = train_test_split(range(len(dataset)), test_size=ratio,
                                                 stratify=labels,
                                                 random_state=seed)
    elif split_type == 'consecutive':
        first_idx = list(range(n_first))
        second_idx = list(range(n_first, n_first + n_second))
    else:
        raise ValueError('Invalid split type. Please choose * (or *_with_idx) from "random", "stratified", '
                         'or "consecutive".')

    first_set = Subset(dataset, first_idx)
    second_set = Subset(dataset, second_idx)

    return first_set, second_set


def split_dataset(dataset, split_type='consecutive', train_val_folds=1, val_ratio=None, test_ratio=None, train_idx=None,
                  val_idx=None, test_idx=None, seed=42):
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
    train_set = []  # list of Subsets for every fold)
    val_set = []

    train_val_set, test_set = split_dataset_in_two_sets(dataset, test_ratio, split_type, seed)

    # If split from ids
    if split_type == 'idx':
        if train_val_folds > 1:
            for fold in range(train_val_folds):
                if train_idx is None or val_idx is None or test_idx is None:
                    raise ValueError('When using split_type="ids", train_ids, val_ids and test_ids must be provided.')
                train_set.append(Subset(dataset, train_idx[fold]))
                val_set.append(Subset(dataset, val_idx[fold]))
            test_set = Subset(dataset, test_idx)
    # If split only test from ids
    elif '_with_test_idx' in split_type:
        if test_idx is None:
            raise ValueError('When using split_type="*_with_test_idx", test_ids must be provided.')
        test_set = Subset(dataset, test_idx)
        if train_val_folds > 1:
            for fold in range(train_val_folds):
                train_set[fold], val_set[fold] = split_dataset_in_two_sets(train_val_set, val_ratio / (1 - test_ratio),
                                                                           split_type.replace('_with_test_idx', ''),
                                                                           seed)
        else:
            train_set, val_set = split_dataset_in_two_sets(train_val_set, val_ratio / (1 - test_ratio),
                                                           split_type.replace('_with_test_idx', ''), seed)
    # If not split from ids, then split using the given strategy
    else:
        if train_val_folds > 1:
            fold_sets = []
            remaining_set = train_val_set
            fold_ratio = 1 / train_val_folds
            for fold in range(train_val_folds):
                remaining_set, fold_set = split_dataset_in_two_sets(remaining_set,
                                                                    fold_ratio * len(train_val_set) / len(
                                                                        remaining_set),
                                                                    split_type, seed)
                fold_sets.append(fold_set)
            for fold in range(train_val_folds):
                train_set.append(ConcatDataset([fold_sets[fold2] for fold2 in range(train_val_folds) if fold2 != fold]))
                val_set.append(fold_sets[fold])
        else:
            train_set, val_set = split_dataset_in_two_sets(train_val_set, val_ratio / (1 - test_ratio), split_type,
                                                           seed)

    # Assert whole dataset is used
    if train_val_folds == 1:
        assert len(train_set) + len(val_set) + len(test_set) == len(dataset)
    else:
        assert len(train_set[0]) + len(val_set[0]) + len(test_set) == len(dataset)

    return train_set, val_set, test_set


def get_dataloaders(train_set, val_set, test_set, batch_size, collate_fn):
    # TODO: Perhaps we will need to create dataloaders with stratified sampling so that the batches are balanced
    if isinstance(train_set, list):  # If cross-validation, then each set is a list of Subset for each fold
        train_loader = [torch.utils.data.DataLoader(train_set[fold], batch_size=batch_size, shuffle=True,
                                                    collate_fn=collate_fn) for fold in range(len(train_set))]
        val_loader = [torch.utils.data.DataLoader(val_set[fold], batch_size=batch_size, shuffle=False,
                                                  collate_fn=collate_fn) for fold in range(len(val_set))]
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                   collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                                 collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def collate_fn(batch):
    """
    Collates a list of (image, label) pairs into a batch consisting of two tensors: inputs, labels.
    :param batch: The list of (image, label) pairs.
    :return: The two tensors: inputs and labels.
    """
    # Unzip
    inputs, labels = zip(*batch)
    inputs = list(inputs)
    labels = list(labels)

    # Transform each input to tensor
    transform = transforms.ToTensor()
    inputs = [transform(input) for input in inputs]

    # Stack tensors/elements into a batch tensor
    inputs = torch.stack(inputs, dim=0)
    labels = torch.tensor(labels)
    return inputs, labels
