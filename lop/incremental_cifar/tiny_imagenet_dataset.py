"""
Data loader for the Tiny ImageNet dataset (200 classes, 64x64), mimicking the
CifarDataSet interface from mlproj_manager so it can be dropped into the
incremental_cifar experiment with minimal changes.

Folder layout expected under root_dir (the extracted tiny-imagenet-200):
    root_dir/
        train/<wnid>/images/*.JPEG          (500 images per class, 200 classes)
        val/images/*.JPEG                   (50 images per class, 200 classes)
        val/val_annotations.txt             (maps each val image to a wnid)

The class index (0..199) is defined by the alphabetical order of the train
folder names, and val labels are mapped through the same order so train/val
stay consistent.
"""
import os

import numpy as np
import torch
from PIL import Image

from mlproj_manager.util.data_preprocessing_and_transformations import normalize, preprocess_labels


class TinyImageNetDataSet:
    """
    Tiny ImageNet dataset. Mirrors CifarDataSet's public interface:
        - .data["data"]   : (N, 64, 64, 3) float32, normalized
        - .data["labels"] : (N, num_classes) one-hot float32
        - .integer_labels : list of int class indices
        - .classes        : np.array of active class indices
        - .current_data   : dict partition of the active classes
    """

    def __init__(self, root_dir, train=True, transform=None, classes=None,
                 device=None, image_normalization=None, label_preprocessing=None,
                 use_torch=False, flatten=False, num_images_per_class=500):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.num_total_classes = 200
        self.num_images_per_class = num_images_per_class
        self.classes = np.array(classes, np.int32) if classes is not None else np.arange(self.num_total_classes)
        assert self.classes.size <= self.num_total_classes
        self.device = torch.device("cpu") if device is None else device
        self.image_norm_type = image_normalization
        self.label_preprocessing = label_preprocessing
        self.use_torch = use_torch
        self.flatten = flatten

        self.data = self.load_data()
        self.integer_labels = self.data["labels"]  # list of int labels (before one-hot)
        self.preprocess_data()
        self.current_data = self.partition_data()

    # ------------------------------------------------------------------ #
    def _get_class_to_idx(self):
        """Map wnid -> class index (0..199), sorted alphabetically (same as ImageFolder)."""
        train_dir = os.path.join(self.root_dir, "train")
        wnids = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        return {wnid: i for i, wnid in enumerate(wnids)}

    def load_data(self):
        """
        Load the train or val split as (N, 64, 64, 3) uint8 arrays and integer labels.
        :return: dict with keys "data" (np uint8 array) and "labels" (list of ints)
        """
        if self.train:
            return self._load_train()
        return self._load_val()

    def _load_train(self):
        train_dir = os.path.join(self.root_dir, "train")
        class_to_idx = self._get_class_to_idx()
        wnids = sorted(class_to_idx.keys(), key=lambda w: class_to_idx[w])

        paths, labels = [], []
        for wnid in wnids:
            class_img_dir = os.path.join(train_dir, wnid, "images")
            # 只取前 num_images_per_class 张，减少加载量/内存
            for fname in sorted(os.listdir(class_img_dir))[:self.num_images_per_class]:
                paths.append(os.path.join(class_img_dir, fname))
                labels.append(class_to_idx[wnid])

        data = np.zeros((len(paths), 64, 64, 3), dtype=np.uint8)
        for i, p in enumerate(paths):
            data[i] = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        return {"data": data, "labels": labels}

    def _load_val(self):
        val_dir = os.path.join(self.root_dir, "val")
        img_dir = os.path.join(val_dir, "images")
        annot_path = os.path.join(val_dir, "val_annotations.txt")
        class_to_idx = self._get_class_to_idx()

        paths, labels = [], []
        with open(annot_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, wnid = parts[0], parts[1]
                paths.append(os.path.join(img_dir, fname))
                labels.append(class_to_idx[wnid])

        data = np.zeros((len(paths), 64, 64, 3), dtype=np.uint8)
        for i, p in enumerate(paths):
            data[i] = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        return {"data": data, "labels": labels}

    # ------------------------------------------------------------------ #
    def preprocess_data(self):
        """Convert data to float/normalized and labels to one-hot, mirroring CifarDataSet."""
        if self.flatten:
            self.data["data"] = self.data["data"].reshape(self.data["data"].shape[0], -1)

        self.data["data"] = torch.tensor(self.data["data"], dtype=torch.float32) if self.use_torch \
            else np.float32(self.data["data"])
        self.data["data"] = normalize(self.data["data"], norm_type=self.image_norm_type, max_val=255)

        new = preprocess_labels(self.data["labels"], preprocessing_type=self.label_preprocessing)
        self.data["labels"] = torch.tensor(new, dtype=torch.float32) if self.use_torch else np.float32(new)

        if self.use_torch:
            self.data["data"] = self.data["data"].to(device=self.device)
            self.data["labels"] = self.data["labels"].to(device=self.device)

    def partition_data(self):
        """Partition the data according to the classes in self.classes."""
        current_data = {}
        correct_rows = np.ones(self.data['data'].shape[0], dtype=bool)
        if self.classes is not None:
            correct_rows = np.in1d(self.integer_labels, self.classes)

        current_data['data'] = self.data['data'][correct_rows, :, :, :]
        if self.label_preprocessing == "one-hot":
            current_data['labels'] = self.data['labels'][correct_rows][:, self.classes]
        else:
            current_data['labels'] = self.data['labels'][correct_rows]
        return current_data

    def select_new_partition(self, new_classes):
        """Regenerate the partition based on the given class indices."""
        self.classes = np.array(new_classes, dtype=np.int32)
        self.current_data = self.partition_data()

    # ------------------------------------------------------------------ #
    def __len__(self):
        return self.current_data['data'].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.current_data['data'][idx]
        label = self.current_data['labels'][idx]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def set_transformation(self, new_transformation):
        self.transform = new_transformation
