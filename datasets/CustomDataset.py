from torch.utils.data import Dataset
import os
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.label_file = label_file
        self.transform = transform

        self.images = self._load_images(image_dir)  # List of image names
        self.labels = self._load_labels(label_file)  # List of labels

    def _load_images(self, image_dir):
        images = []  # List of image names
        for image_name in os.listdir(image_dir):
            images_name = os.path.join(image_dir, image_name)
            self.images.append(images_name)
        return images

    def _load_labels(self, label_file):
        labels = {}  # List of labels
        with open(label_file, 'r') as file:
            next(file)  # Skip header line
            for line in file:
                image_name, label = line.strip().split(',')
                labels[image_name] = label
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        label = self.labels[image_name]

        # Load image data
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

class CIFAR10(CustomDataset):
    def __init__(self, data_dir, transform=None):
        image_dir = os.path.join(data_dir, 'train')
        label_file = os.path.join(data_dir, 'trainLabels.csv')
        super().__init__(image_dir, label_file, transform)

        # TODO: Add dictionary mapping labels to label idx

    def _load_labels(self, label_file):
        labels = {}  # List of labels
        with open(label_file, 'r') as file:
            next(file)  # Skip header line
            for line in file:
                image_name, label = line.strip().split(',')
                labels[image_name] = label  # TODO: Convert labels to label indices
        return labels