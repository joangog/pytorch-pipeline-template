import glob
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


def read_dataset_split(dataset_split_path, train_val_folds):
    """
    Reads indices for dataset splitting from directory containing a TXT file for each subset and fold.
    :param dataset_split_path: Path to directory containing the split files.
    :param train_val_folds: Number of folds for cross-validation on the train+val set.
    :return: Lists of indices for train, val (one for each fold) and test sets.
    """
    # Get file paths
    train_idx_files = glob.glob(os.path.join(dataset_split_path, 'train_fold_*.txt'))
    val_idx_files = glob.glob(os.path.join(dataset_split_path, 'val_fold_*.txt'))
    test_idx_file = os.path.join(dataset_split_path, 'test.txt')
    if len(train_idx_files) != 0 and len(val_idx_files) != 0:  # If train/val splits exist
        assert len(train_idx_files) == len(
            val_idx_files) == train_val_folds  # Check that there is one file per fold for train and val

    # Read indices
    train_idx, val_idx = [], []
    for fold in range(train_val_folds):
        with open(train_idx_files[fold], 'r') as f:
            idx = [int(id) for id in f.readlines()[1:]]
        train_idx.append(idx)
        with open(val_idx_files[fold], 'r') as f:
            idx = [int(id) for id in f.readlines()[1:]]
        val_idx.append(idx)
    with open(test_idx_file, 'r') as f:
        idx = [int(id) for id in f.readlines()[1:]]
    test_idx = idx

    return train_idx, val_idx, test_idx


def split_dataset(dataset, split_type='consecutive', val_ratio=None, test_ratio=None, train_val_folds=1,
                  split_path=None, seed=42):
    """
    Splits the dataset into training and test sets.
    :param dataset: Dataset object to split.
    :param split_type: Type of splitting strategy to use.
                        If 'random', we use random samples for each split.
                        If 'stratify', we take use class-balanced splits.
                        If 'consecutive', we take consecutive samples for each split.
                        If 'idx', we take samples based on given ids.
    :param val_ratio: Ratio of dataset to use for validation set (to be used when split_type is not 'idx').
    :param test_ratio: Ratio of dataset to use for test set (to be used when split_type is not 'idx' or '*_with_test_idx').
    :param train_val_folds: Number of folds for cross-validation on the train+val set. If 1, no cross-validation is performed.
    :param split_path: Path to directory containing the split files (to be used with split_type='ids' or '*_with_test_idx').
    :param seed: Random seed for splitting.
    :return: The Subset objects for the training and test sets.
    """

    # Initialize
    train_set = []  # list of Subsets for every fold
    val_set = []
    train_val_folds = 1 if train_val_folds is None else train_val_folds

    # If split from ids
    if split_type == 'idx':
        # Read indices from files
        if split_path is None:
            raise ValueError('When using split_type="ids", split_path must be provided.')
        train_idx, val_idx, test_idx = read_dataset_split(split_path, train_val_folds)
        for fold in range(train_val_folds):
            train_set.append(Subset(dataset, train_idx[fold]))
            val_set.append(Subset(dataset, val_idx[fold]))
        test_set = Subset(dataset, test_idx)

    # If not split all from ids,
    else:

        # If split only test from ids
        if '_with_test_idx' in split_type:
            # Read test indices from file
            if split_path is None:
                raise ValueError('When using split_type="*_with_test_idx", split_path must be provided.')
            _, _, test_idx = read_dataset_split(split_path, train_val_folds)
            test_set = Subset(dataset, test_idx)
            train_val_set = Subset(dataset, list(set(range(len(dataset))) - set(test_idx)))

        # If not split from ids, then split test using specified method
        else:
            train_val_set, test_set = split_dataset_in_two_sets(dataset, test_ratio, split_type, seed)

        # Split train/val using specified method
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
            train_set, val_set = split_dataset_in_two_sets(train_val_set, val_ratio / (1 - test_ratio),
                                                           split_type.replace('_with_test_idx', ''),
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
