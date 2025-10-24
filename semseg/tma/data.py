import os
import torch
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import DataLoader

from semseg.data import SemSegDataset, get_training_augmentation, hwc2chw


class TMADataset(SemSegDataset):
    def __init__(self, root, transform=None, binary=True, resize=None, rescale=True):
        super().__init__(root, transform=transform, binary=binary, resize=resize)
        self.rescale = rescale

    def _preprocess_mask(self, mask):
        if self.binary:
            mask[mask > 0] = 1
        elif self.rescale:
            mask[mask == 50] = 1
            mask[mask == 150] = 2
            mask[mask == 200] = 3
        return mask


class TMADatasetInference(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, binary=True, resize=None):
        self.root = root
        self.transform = transform
        self.binary = binary
        self.resize = resize
        self.filenames = self._set_filenames()

    def _set_filenames(self):
        image_filenames = os.listdir(self.root)
        image_filenames_no_ext = [os.path.splitext(f)[0] for f in image_filenames]
        return image_filenames_no_ext

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.root, f"{filename}.jpeg")
        image = np.array(Image.open(image_path).convert("RGB"))

        if self.resize:
            image = cv2.resize(image, (self.resize, self.resize), interpolation=cv2.INTER_LINEAR)

        if self.transform is not None:
            sample = self.transform(image=image)
        else:
            sample = dict(image=image)

        # convert to other format HWC -> CHW
        # sample["original_image"] = hwc2chw(image)
        sample["image"] = hwc2chw(sample["image"])
        sample["filename"] = filename
        return sample


def get_dataset_dir(hostname):
    if hostname == 'WS-BRIEF_WP3':
        root_dir = r'E:\AnnotationsCortexMedulla'
    else:
        root_dir = r'D:\AnnotationsCortexMedulla'
    return root_dir



def get_tma_datasets_v(downsample, binary, v, resize=None, train_augm=None, get_train=True, rescale=True, hostname=None):
    datasets = []
    dataset_dir = get_dataset_dir(hostname)

    if get_train:
        train_path = rf'{dataset_dir}\v{v}\Train\Downsample-{downsample}'
        if os.path.exists(train_path):
            train_tma = TMADataset(
                train_path,
                binary=binary,
                resize=resize,
                transform=train_augm,
                rescale=rescale
            )
        else:
            print(f"WARNING: {train_path} not exists!")
            train_tma = []
        datasets.append(train_tma)
    valid_path = rf'{dataset_dir}\v{v}\Validation\Downsample-{downsample}'
    if os.path.exists(valid_path):
        valid_tma = TMADataset(
            valid_path,
            binary=binary,
            resize=resize,
            rescale=rescale
        )
    else:
        print(f"WARNING: {valid_path} not exists!")
        valid_tma = []
    datasets.append(valid_tma)
    test_path = rf'{dataset_dir}\v{v}\Test\Downsample-{downsample}'
    if os.path.exists(test_path):
        test_tma = TMADataset(
            test_path,
            binary=binary,
            resize=resize,
            rescale=rescale
        )
    else:
        print(f"WARNING: {test_path} not exists!")
        test_tma = None
    datasets.append(test_tma)
    return datasets


def get_tma_datasets_v8(downsample, binary, resize=None, train_augm=None, get_train=True, hostname=None):
    return get_tma_datasets_v(downsample, binary, 8,
                              resize=resize, train_augm=train_augm,
                              get_train=get_train, rescale=False, hostname=hostname)


def get_tma_datasets_v6(downsample, binary, resize=None, train_augm=None, get_train=True, hostname=None):
    return get_tma_datasets_v(downsample, binary, 6,
                              resize=resize, train_augm=train_augm,
                              get_train=get_train, rescale=False, hostname=hostname)


def get_tma_datasets_v4(downsample, binary, resize=None, train_augm=None, get_train=True, hostname=None):
    return get_tma_datasets_v(downsample, binary, 4,
                              resize=resize, train_augm=train_augm,
                              get_train=get_train, rescale=False, hostname=hostname)


def get_tma_datasets_v3(downsample, binary, resize=None, train_augm=None, get_train=True):
    return get_tma_datasets_v(downsample, binary, 3,
                              resize=resize, train_augm=train_augm, get_train=get_train)


def get_tma_datasets(downsample, binary, resize=None, train_augm=None, get_train=True):
    valid_path = r'E:\AnnotationsCortexMedulla\v1\Validation\Downsample-' + str(downsample)
    test_path = r'E:\AnnotationsCortexMedulla\v1\Test\Downsample-' + str(downsample)
    valid_tma = TMADataset(
        valid_path,
        binary=binary,
        resize=resize
    )
    test_tma = TMADataset(
        test_path,
        binary=binary,
        resize=resize
    )
    print(f"Validation Set Size: {len(valid_tma)}")
    print(f"Test       Set Size: {len(test_tma)}")
    if get_train:
        train_path = r'E:\AnnotationsCortexMedulla\v1\Train\Downsample-' + str(downsample)
        train_tma = TMADataset(
            train_path,
            binary=binary,
            resize=resize,
            transform=train_augm
        )
        print(f"Training   Set Size: {len(train_tma)}")
        return train_tma, valid_tma, test_tma
    else:
        return valid_tma, test_tma


def get_tma_dataloaders(datasets, n_cpu=0, batch_size=4, inference_config=False):
    dataloaders = []
    for i, dataset in enumerate(datasets):
        shuffle = False if inference_config else i == 0
        dataloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_cpu))
    return dataloaders


if __name__ == '__main__':
    transform = get_training_augmentation()
    image_path = r'E:\AnnotationsCortexMedulla\v2\Train\Downsample-16\Image'
    mask_path = r'E:\AnnotationsCortexMedulla\v2\Train\Downsample-16\Mask'
    example_path = r'E:\AnnotationsCortexMedulla\v2\Train\Downsample-16\Example'
    os.makedirs(example_path, exist_ok=True)

    images = os.listdir(image_path)
    masks = os.listdir(mask_path)

    for image, mask in zip(images, masks):
        path_to_image_0 = os.path.join(image_path, image)
        path_to_mask_0 = os.path.join(mask_path, mask)

        image_0 = cv2.imread(path_to_image_0)
        mask_0 = cv2.imread(path_to_mask_0, cv2.IMREAD_GRAYSCALE)

        print(f"image --> {image_0.dtype}, {image_0.shape}")
        print(f"mask  --> {mask_0.dtype}, {mask_0.shape}")

        sample = transform(image=image_0, mask=mask_0)

        image_aug, mask_aug = sample['image'], sample['mask']

        print(f"image_aug --> {image_aug.dtype}, {image_aug.shape}")
        print(f"mask_aug  --> {mask_aug.dtype}, {mask_aug.shape}")

        image_aug_out = os.path.join(example_path, image)
        mask_aug_out = os.path.join(example_path, mask)

        cv2.imwrite(image_aug_out, image_aug)
        cv2.imwrite(mask_aug_out, mask_aug)
