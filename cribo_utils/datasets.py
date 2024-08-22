import time
import tarfile
import os
import torch
from torchvision.datasets.folder import default_loader, DatasetFolder, IMG_EXTENSIONS
from typing import Any, Callable, Optional
from collections import defaultdict


def get_tensor_binary_instance_labels(label, nb_categories=91):
    tmp = torch.zeros(nb_categories)
    for obj in label:
        tmp[obj["category_id"]] = 1
    return tmp


def untar_to_dst(args, src):
    assert (args.untar_path != "")

    if args.untar_path[0] == '$':
        args.untar_path = os.environ[args.untar_path[1:]]
    start_copy_time = time.time()

    if int(args.gpu) == 0:
        with tarfile.open(src, 'r') as f:
            f.extractall(args.untar_path)

        print('Time taken for untar:', time.time() - start_copy_time)

    try:
        torch.distributed.barrier()
    except:
        pass
    time.sleep(5)


def get_dataset(args, transform, target_transform=lambda x: x, val_or_train='train', wrapper=None):
    if len(args.untar_path) > 0 and args.untar_path[0] == '$':
        args.untar_path = os.environ[args.untar_path[1:]]

    if args.dataset_type == 'imagenet1k':
        if args.imagenet1k_path.split('.')[-1] == 'tar':
            untar_to_dst(args, args.imagenet1k_path)
            root_dir = os.path.join(args.untar_path, args.imagenet1k_path.split('/')[-1].split('.')[0])
        else:
            root_dir = args.imagenet1k_path

        assert ('ILSVRC2012_img_train' in os.listdir(root_dir))
        assert ('ILSVRC2012_img_val' in os.listdir(root_dir))
        return CrIBoDataset(os.path.join(root_dir, 'ILSVRC2012_img_{}'.format(val_or_train)), args,
                             transform=transform,
                             return_index_instead_of_target=True)
    else:
        raise NotImplemented


def get_dataloader(args, dataset, **kwargs):
    keyword_args = {'batch_size': args.batch_size_per_gpu,
                    'num_workers': args.num_workers,
                    'pin_memory': True}
    if 'drop_last' not in kwargs:
        keyword_args['drop_last'] = True

    try:
        keyword_args['sampler'] = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        out = torch.utils.data.DataLoader(dataset, **keyword_args, **kwargs)
    except:
        out = torch.utils.data.DataLoader(dataset, **keyword_args, **kwargs)
    return out


class CrIBoDataset(DatasetFolder):
    def __init__(
            self,
            root: str,
            args,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            return_index_instead_of_target: Optional[bool] = False):
        self.args = args
        self.return_index_instead_of_target = return_index_instead_of_target
        self.classidx_to_pathlist = defaultdict(list)
        self.root = root
        classes, _ = self.find_classes(self.root)
        super().__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                         transform=transform,
                         target_transform=target_transform,
                         is_valid_file=is_valid_file)

        self.imgs = self.samples

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        sample_pos = None
        if self.transform is not None:
            sample, sample_pos = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_index_instead_of_target:
            target = index

        return sample, target, sample_pos
