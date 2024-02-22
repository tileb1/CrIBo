import torchvision.transforms.functional as F_viz
from torchvision.transforms.transforms import RandomResizedCrop, Compose, RandomHorizontalFlip
import torch
from torchvision import transforms


class CenterCropWithPos(RandomResizedCrop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_size = 10000
        x = torch.arange(max_size).repeat(max_size, 1)[None, :]
        CenterCropWithPos._pos = torch.cat((x, x.permute(0, 2, 1)), dim=0).float()
        self.transform = transforms.CenterCrop(224)

    def forward(self, img):
        # PIL convention
        w_pil, h_pil = img.size
        pos = RandomResizedCropWithPos._pos[:, :h_pil, :w_pil]
        out = self.transform(img)
        out_pos = self.transform(pos)
        return out, out_pos


class RandomResizedCropWithPos(RandomResizedCrop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_size = 10000
        x = torch.arange(max_size).repeat(max_size, 1)[None, :]
        RandomResizedCropWithPos._pos = torch.cat((x, x.permute(0, 2, 1)), dim=0).float()

    def forward(self, img):
        mask = None
        if isinstance(img, tuple):
            img, mask = img
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        # PIL convention
        w_pil, h_pil = img.size
        pos = RandomResizedCropWithPos._pos[:, :h_pil, :w_pil]
        out = F_viz.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        out_pos = F_viz.resized_crop(pos, i, j, h, w, self.size, self.interpolation)
        if mask is not None:
            mask = F_viz.resized_crop(mask, i, j, h, w, self.size, transforms.InterpolationMode.NEAREST)
        return out, out_pos, mask


class MyCompose(Compose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        flip_bool = 0
        pos = None
        mask = None
        for t in self.transforms:
            if type(t) == RandomResizedCropWithPos or type(t) == CenterCropWithPos:
                if isinstance(img, tuple):
                    img, pos, mask = t(img)
                else:
                    img, pos = t(img)
            elif type(t) == MyComposeInner:
                img, flip_bool = t(img)
            else:
                img = t(img)
        if flip_bool == 1:
            return img, F_viz.hflip(pos), F_viz.hflip(mask) if mask is not None else None
        return img, pos, mask


class MyComposeInner(Compose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        flip_bool = 0
        for t in self.transforms:
            if type(t) == RandomHorizontalFlipWithFlipBool:
                img, flip_bool = t(img)
            else:
                img = t(img)
        return img, flip_bool


class RandomHorizontalFlipWithFlipBool(RandomHorizontalFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img):
        if torch.rand(1) < self.p:
            return F_viz.hflip(img), 1
        return img, 0
