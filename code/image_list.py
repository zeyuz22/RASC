from torchvision.datasets import VisionDataset
import warnings
import torch
from PIL import Image
import os
import os.path
import numpy as np


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageList(object):
    def __init__(self, root=None, transform=None, strong_transform=None, target_transform=None, empty=False,
                 test=False):
        self.transform = transform
        self.strong_transform = strong_transform
        self.target_transform = target_transform
        self.empty = empty
        self.test = test
        if self.empty:
            self.samples = np.empty((1, 2), dtype=np.dtype((np.unicode_, 1000)))
        else:
            self.samples = np.loadtxt(root, dtype=np.dtype((np.unicode_, 1000)), delimiter=' ')
        self.loader = pil_loader

    def __getitem__(self, index):
        path, target = self.samples[index]
        target = int(target)

        sample = self.loader(path)

        if self.strong_transform is not None:
            sample_s = self.strong_transform(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.test:
            return sample, target, path, sample_s
        else:
            return sample, target, path

    def __len__(self):
        return len(self.samples)

    def add_item(self, addition):
        if self.empty:
            self.samples = addition
            self.empty = False
        else:
            self.samples = np.concatenate((self.samples, addition), axis=0)
        return self.samples

    def remove_item(self, reduced):
        self.samples = np.delete(self.samples, reduced, axis=0)
        return self.samples
