import os
import random
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision.datasets.folder import pil_loader
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
from torchvision.datasets import ImageFolder


def build_2d_dataloader(dataset, batch_size, is_train=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=0)
    return dataloader


def build_2d_subsampled_dataloader(dataset, batch_size, is_train=True, subsize=.1):
    idxs = random.sample(range(len(dataset)), int(len(dataset) * subsize))
    sampler = SubsetRandomSampler(idxs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, sampler=sampler)
    return dataloader


def build_2d_weighted_dataloader(dataset, batch_size, is_train=True):
    cnt_targets = Counter(dataset.targets)
    unique_targets = set(dataset.targets)
    print(f"Counter Targets: {cnt_targets}")

    num_elements = len(dataset)
    class_weights = []
    for t in unique_targets:
        val = 1 - cnt_targets[t] / len(dataset)
        class_weights.append(val)

    print(f"Weights: {class_weights}")
    element_weights = []

    for label in dataset.targets:
        element_weights.append(class_weights[label])

    element_weights = torch.Tensor(element_weights)
    sampler = WeightedRandomSampler(element_weights, num_elements, replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=sampler)
    return dataloader


def build_2d_dataset(path, transform=None):
    dataset = ImageDataset(path, transform=transform)
    return dataset


def build_2d_dataset_simple(path, transform=None):
    dataset = ImageFolder(path, transform=transform)
    return dataset



class ImageDataset(ImageFolder):

    def __init__(self, path, transform=None):
        super().__init__(root=path, transform=transform)
        self.filename = path

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        ext_ids = [self.samples[idx][0], ]
        return {
            "image": x,
            "label": y,
            "external_id": ext_ids
        }


class ImageDatasetFromDF(data.Dataset):
    def __init__(self, df, split_col, split_type,
                 transform=None, label_mapping=None, exclude_labels=None, compartment=None,
                 root_dir=None, path_key='path', label_key='label',
                 filter_key=None, keep_value=None, discard_value=None):
        """
        df: DataFrame with 'filename', 'label', and 'split-*' columns
        split_col: str, column name for split, e.g., 'split-0'
        split_type: str, either 'train' or 'validation'
        transform: torchvision transforms to apply
        """
        df_filtered = df.copy()

        # Exclude specified labels
        if exclude_labels is not None:
            df_filtered = df_filtered[~df_filtered['label'].isin(exclude_labels)]

        if filter_key is not None:
            if keep_value is not None:
                df_filtered = df_filtered[df_filtered[filter_key].isin(keep_value)]
            if discard_value is not None:
                df_filtered = df_filtered[~df_filtered[filter_key].isin(discard_value)]

        # Filter by Compartment (if requested)
        if compartment is not None:
            if 'Compartment' not in df_filtered.columns:
                raise KeyError("Column 'Compartment' not found in DataFrame.")
            if isinstance(compartment, (list, tuple, set)):
                valid_compartments = set(compartment)
            else:
                valid_compartments = {compartment}
            df_filtered = df_filtered[df_filtered['Compartment'].isin(valid_compartments)]

        # Filter by split
        self.df = df_filtered[df_filtered[split_col] == split_type].reset_index(drop=True)
        self.transform = transform

        if label_mapping is not None:
            self.label_to_id = {
                label: class_id
                for class_id, label_list in label_mapping.items()
                for label in label_list
            }
            self.targets = [self.label_to_id[label] for label in self.df['label']]
        else:
            self.label_to_id = None
            self.targets = list(self.df['label'])
        self.classes = sorted(list(set(self.targets)))
        self.root_dir = root_dir
        self.path_key = path_key
        self.label_key = label_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.root_dir is not None:
            image_path = os.path.join(self.root_dir, row['filename'])
        else:
            image_path = row[self.path_key]
        label_str = row[self.label_key]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.label_to_id[label_str] if self.label_to_id else label_str

        return {
            "image": image,
            "label": label,
            "external_id": image_path
        }


class ImageFolderNoLabels(data.Dataset):
    def __init__(self, dataroot, transform, ext='.jpg'):
        self.dataroot = dataroot
        self.transform = transform
        self.images = [os.path.join(self.dataroot, i) for i in os.listdir(self.dataroot) if i.endswith(ext)]

    def __getitem__(self, index: int):
        path = self.images[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.images)
