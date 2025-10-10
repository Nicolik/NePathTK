import os
import torch
import shutil
import numpy as np
from PIL import Image
import cv2
import albumentations as albu
from torch.utils.data import DataLoader


class SemSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, binary=True, resize=None):

        self.root = root
        self.transform = transform
        self.binary = binary
        self.resize = resize

        self.images_directory = os.path.join(self.root, "Image")
        self.masks_directory = os.path.join(self.root, "Mask")

        self.filenames = self._set_filenames()

    def _set_filenames(self):
        image_filenames = os.listdir(self.images_directory)
        mask_filenames = os.listdir(self.masks_directory)

        image_filenames_no_ext_jpeg = [f.split(".jpeg")[0] for f in image_filenames if ".jpeg" in f]
        image_filenames_no_ext_jpg = [f.split(".jpg")[0] for f in image_filenames if ".jpg" in f]
        image_filenames_no_ext = image_filenames_no_ext_jpg + image_filenames_no_ext_jpeg
        mask_filenames_no_ext = [f.split(".png")[0] for f in mask_filenames]

        filenames_no_ext = set(image_filenames_no_ext).intersection(set(mask_filenames_no_ext))
        return list(filenames_no_ext)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        jpeg_filename = os.path.join(self.images_directory, f"{filename}.jpeg")
        jpg_filename = os.path.join(self.images_directory, f"{filename}.jpg")
        image_path = jpeg_filename if os.path.exists(jpeg_filename) else jpg_filename
        mask_path = os.path.join(self.masks_directory, f"{filename}.png")

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = self._preprocess_mask(np.array(Image.open(mask_path)))

        if self.resize:
            image = cv2.resize(image, (self.resize, self.resize), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.resize, self.resize), interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            sample = self.transform(image=image, mask=mask)
        else:
            sample = dict(image=image, mask=mask)

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(sample["image"], -1, 0)
        sample["mask"] = (np.expand_dims(sample["mask"], 0)).astype(np.float32)
        sample["filename"] = filename

        return sample

    def _preprocess_mask(self, mask):
        return mask


class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_mapping = {
            -1: 0
        }
        self.filenames = self._set_filenames()

    def _set_filenames(self):
        filenames = []
        for dataset in self.datasets:
            filenames.extend(dataset.filenames)
        return filenames

    def __len__(self):
        len_all = 0
        for i, dataset in enumerate(self.datasets):
            len_all += len(dataset)
            self.dataset_mapping[i] = len_all
        return len_all

    def __getitem__(self, idx):
        for i in range(0, len(self.dataset_mapping)-1):
            if idx < self.dataset_mapping[i]:
                prev_len = self.dataset_mapping[i-1]
                return self.datasets[i][idx-prev_len]


def get_example_augmentation():
    return albu.Compose([
        albu.HorizontalFlip(p=1.0),
        albu.IAAAdditiveGaussianNoise(p=1.0),
    ])


def get_inference_augmentation():
    return albu.PadIfNeeded(min_height=1024, min_width=1024, position=albu.PadIfNeeded.PositionType.TOP_LEFT,
                            border_mode=cv2.BORDER_CONSTANT, value=0, p=1)


def get_inference_augmentation_uni():
    return albu.Compose([
        albu.Resize(height=224, width=224),
        albu.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
    ])


def get_trainining_augmentation_uni():
    return albu.Compose([
        get_inference_augmentation_uni(),
        get_training_augmentation()
    ])


def get_training_augmentation():
    return albu.Compose([
        albu.ColorJitter(hue=0.2, p=0.1),
        albu.HorizontalFlip(p=0.2),
        albu.VerticalFlip(p=0.2),

        albu.OneOf(
            [
                albu.GaussNoise(p=1),
                albu.ISONoise(p=1),
            ],
            p=0.1,
        ),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.1,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.1,
        ),

    ])


def hwc2chw(image):
    return np.moveaxis(image, -1, 0)
