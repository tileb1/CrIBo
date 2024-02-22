from torch.utils.data import Dataset
import os
import time
import tarfile
from PIL import Image
import torch


class PascalVOCAug(Dataset):
    def __init__(self, root, transforms, split='trainaug'):
        super().__init__()
        self.transforms = transforms
        self.split = split
        self.root = root

        # Collect the data
        self.data = self.collect_data()

    def collect_data(self):
        # Get the list of image names
        annotation_file = os.path.join(self.root, f'ImageSets/Segmentation/{self.split}.txt')
        with open(annotation_file, 'r') as f:
            names = [f'{line.strip()}' for line in f.readlines()]
            image_paths, annotation_paths = list(zip(*[(os.path.join(self.root, f'JPEGImages/{n}.jpg'),
                                                        os.path.join(self.root, f'SegmentationClassAug/{n}.png')) for n
                                                       in names]))
            assert all([os.path.exists(p) for p in image_paths])
            # annotation_paths = [os.path.join(self.root, f'SegmentationClassAug/{n}.png') for n in names]
            assert all([os.path.exists(p) for p in annotation_paths])
        data = list(zip(image_paths, annotation_paths))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            # Get the  paths
            image_path, annotation_path = self.data[index]

            # Load
            image = Image.open(image_path).convert("RGB")
            target = Image.open(annotation_path)

            # Augment
            if self.transforms is not None:
                image, target = self.transforms(image, target)
            return image, target
        except FileNotFoundError as e:
            print(len(self.data), len(os.listdir(os.path.join(self.root, 'JPEGImages'))))
            raise e


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    dataset = PascalVOCAug(
        untar_path='/home',
        root='/media/thomas/Elements/cv_datasets/VOC12',
        transforms=None,
        split='val')

    dataset.__getitem__(1)
    print(len(dataset))
