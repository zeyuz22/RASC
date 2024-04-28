from torchvision import datasets, transforms
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
from randaugment import RandAugmentMC


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


crop_size = 224
load_size = int(crop_size * 1.15)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    ResizeImage(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

strong_transform = transforms.Compose([
    ResizeImage(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(crop_size),
    RandAugmentMC(n=2, m=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    ResizeImage(256),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
