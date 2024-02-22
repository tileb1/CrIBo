from torch.utils.data import Dataset
import os
import time
import tarfile
from PIL import Image
import torch


class ADE20K(Dataset):
    split_to_dir = {
        'train': 'training',
        'val': 'validation'
    }

    def __init__(self, root, transforms, split='train'):
        super().__init__()
        self.transforms = transforms
        self.split = split
        self.root = root

        # Collect the data
        self.data = self.collect_data()

    def collect_data(self):
        # Get the image and annotation dirs
        image_dir = os.path.join(self.root, f'images/{self.split_to_dir[self.split]}')
        annotation_dir = os.path.join(self.root, f'annotations/{self.split_to_dir[self.split]}')

        # Collect the filepaths
        image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
        annotation_paths = [os.path.join(annotation_dir, f) for f in sorted(os.listdir(annotation_dir))]
        data = list(zip(image_paths, annotation_paths))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the  paths
        image_path, annotation_path = self.data[index]

        # Load
        image = Image.open(image_path).convert("RGB")
        target = Image.open(annotation_path)

        # Augment
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    dataset = ADE20K(
        untar_path='/home/thomas/Downloads/temp',
        root='/media/thomas/Elements/cv_datasets/ade.tar',
        transforms=None,
        split='train'
    )

    im, ann = dataset.__getitem__(0)

    ann_arr = np.array(ann)
    im_arr = np.array(im)
    print(ann_arr.shape)
    print(im_arr.shape)
