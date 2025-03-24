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


def split_dataset(dataset, val_ratio, test_ratio, split_type=None, seed=42):
    """
    Splits the dataset into training and test sets.
    :param dataset: Dataset object to split.
    :param val_ratio: Ratio of dataset to use for validaton test.
    :param test_ratio: Ratio of dataset to use for test set.
    :param split_type: Type of splitting strategy to use.
                        If 'random', we use random samples for each split.
                        If 'stratify', we take use class-balanced splits.
                        If None or '', we take consecutive samples for each split.
    :param seed: Random seed for splitting.
    :return: The Subset objects for the training and test sets.
    """
    n_test = int(test_ratio * len(dataset))
    n_val = int(val_ratio * len(dataset))
    n_train = len(dataset) - n_test - n_val

    if split_type == 'random':  # If we want random splits
        train_idx = np.random.choice(range(len(dataset)), n_train, replace=False)
        val_idx = np.random.choice(list(set(range(len(dataset))) - set(train_idx)), n_val, replace=False)
        test_idx = list(set(range(len(dataset))) - set(train_idx) - set(val_idx))
    elif split_type == 'stratify':  # If we want class-balanced splits
        labels = [dataset.labels[image] for image in dataset.images]
        train_idx, val_test_idx = train_test_split(range(len(dataset)), test_size=(n_val + n_test),
                                                   stratify=labels,
                                                   random_state=seed)  # Split dataset into train and val+test
        val_idx, test_idx = train_test_split(val_test_idx, test_size=test_ratio,
                                             stratify=[labels[i] for i in val_test_idx],
                                             random_state=seed)  # Split val+test into val and test
    elif split_type is None or split_type == '':
        train_idx = list(range(n_train))
        val_idx = list(range(n_train, n_train + n_val))
        test_idx = list(set(range(len(dataset))) - set(train_idx) - set(val_idx))
    else:
        raise ValueError('Invalid split type. Please choose from "random", "stratify", or "None"')

    # TODO: implement splitting for cross-validation (you might use scikit.kfold but think about how to make stratified ones)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

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
