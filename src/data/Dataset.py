import torch
import os
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_file, transform=None, label_to_int_map=None):
        self.image_dir = image_dir
        self.label_file = label_file
        self.transform = transform

        self.label_to_int_map = label_to_int_map  # Mapping from label name to label integer index

        self.images, self.image_ext = self._load_images(image_dir)  # List of image names and the image file extension
        self.labels = self._load_labels(label_file)  # List of labels

        self.num_classes = len(set([label for label in self.labels.values()]))

    def _label_to_int(self, label):
        if self.label_to_int_map is None:
            return label
        else:
            return self.label_to_int_map[label]

    def _load_images(self, image_dir):
        images = []  # List of image names
        image_ext = None  # Image file extension
        for image in os.listdir(image_dir):
            image_name, image_ext = os.path.splitext(image)
            images.append(image_name)
        return images, image_ext

    def _load_labels(self, label_file):
        labels = {}  # List of labels
        with open(label_file, 'r') as file:
            next(file)  # Skip header line
            for line in file:
                image_name, label = line.strip().split(',')
                image_name = os.path.splitext(image_name.replace(
                    self.image_dir, ''))[0]  # Clear path and file extension, keep only file name
                labels[image_name] = self._label_to_int(label)  # Convert labels to integers
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        label = self.labels[image_name]

        # Load image data
        image_path = os.path.join(self.image_dir, image_name) + self.image_ext
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
