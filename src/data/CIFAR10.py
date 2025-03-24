import os
from src.data.Dataset import Dataset


class CIFAR10(Dataset):
    def __init__(self, data_dir, transform=None):
        image_dir = os.path.join(data_dir, 'train')
        label_file = os.path.join(data_dir, 'trainLabels.csv')

        label_to_int_map = {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9
        }

        super().__init__(image_dir, label_file, transform=None, label_to_int_map=label_to_int_map)

        self.num_classes = len(set([label for label in self.labels.values()]))
