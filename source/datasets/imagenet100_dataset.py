from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import json


class Imagenet100Dataset(Dataset):
    def __init__(self, image_queries, labels_path, transform=None):
        self.image_queries = image_queries
        self.labels_path = labels_path
        self.transform = transform
        self.filepaths = self.get_filepaths()
        self.dirs_to_labels, dirs_to_names = self.load_labels_dict()

    def load_labels_dict(self):
        # Load the labels .json file
        with open(self.labels_path, 'r') as f:
            dirs_to_names = json.load(f)

        # Match each directory
        dirs_to_labels = dict(sorted(dirs_to_names.items(), key=lambda x: x[0]))
        dirs_to_labels = {k: i for i, (k, v) in enumerate(dirs_to_labels.items())}
        return dirs_to_labels, dirs_to_names

    def get_filepaths(self):
        filepaths = [filepath for query in self.image_queries for filepath in glob(query)]
        return filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Get the queried filepath
        filepath = self.filepaths[idx]

        # Infer the label
        key = filepath.split('/')[-2]
        label = self.dirs_to_labels[key]

        # Read the image
        image = Image.open(filepath).convert('RGB')
        return self.transform(image), label
