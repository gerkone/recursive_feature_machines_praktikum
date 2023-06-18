from typing import Dict, List
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from urllib import request
from scipy.io.arff import loadarff

import torch


CELEBA_FEATURES = {
    "glasses": 15,
    "sideburns": 30,
    "attractive": 2,
    "young": 39,
}


def onehot_encode(y, targets):
    onehot = np.zeros(len(targets))
    onehot[targets.index(y)] = 1
    return onehot


def split(trainset, p=0.8):
    train, val = train_test_split(trainset, train_size=p)
    return train, val


def get_celeba_balanced_data(
    dataset, num_classes: int = 2, target: str = "glasses", num_samples=None
):
    if num_samples is None:
        num_samples = len(dataset)

    by_class = {}
    features = []
    feature_idx = CELEBA_FEATURES[target]
    for idx in range(num_samples):
        ex, label = dataset[idx]
        features.append(label[feature_idx])
        g = label[feature_idx].numpy().item()
        ex = ex.flatten()
        ex = ex / torch.linalg.norm(ex)

        onehot = np.zeros(num_classes)
        onehot[g] = 1
        if g in by_class:
            by_class[g].append((ex, onehot))
        else:
            by_class[g] = [(ex, onehot)]
        if idx > num_samples:
            break
    data = []
    max_len = min(25000, len(by_class[1]))

    data.extend(by_class[1][:max_len])
    data.extend(by_class[0][:max_len])
    return data


def get_stl10_data(dataset, targets: List = [0, 9], num_samples=None, class_ratio=0.5):
    assert len(targets) == 2
    if num_samples is None:
        num_samples = 0

    in_target_samples = sum(1 if label in targets else 0 for _, label in dataset)
    num_samples = max(in_target_samples, num_samples)

    adjusted = []
    class_counts = [0, 0]
    class_ratios = [class_ratio, 1 - class_ratio]
    for _, (ex, label) in enumerate(dataset):
        if len(adjusted) < num_samples and label in targets:
            r = class_ratios[targets.index(label)]
            if class_counts[targets.index(label)] < num_samples * r:
                class_counts[targets.index(label)] += 1
                adjusted.append((ex.flatten(), onehot_encode(label, targets)))

    return adjusted


TABULAR_DATA_URLS = {
    "electricity": "https://api.openml.org/data/download/22103245/dataset",
    "telescope": "https://www.openml.org/data/download/54003/dataset",
    "miniboone": "https://api.openml.org/data/download/22103253/dataset",
    "higgs": "https://api.openml.org/data/download/22103254/dataset",
    "jannis": "https://api.openml.org/data/download/22111907/dataset",
    "covertype": "https://api.openml.org/data/download/22103246/dataset",
}

TABULAR_DATA_TARGETS = {
    "electricity": "class",
    "miniboone": "signal",
    "telescope": "class:",
    "higgs": "target",
    "jannis": "class",
    "covertype": "Y",
}


def get_tabular_datasets(data_list: List, max_samples=None) -> Dict:
    dest_dir = os.path.join(os.getcwd(), "datasets/tabular")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    dataset_dict = {}
    for ds in data_list:
        if ds in TABULAR_DATA_URLS:
            arff_dest = os.path.join(dest_dir, f"{ds}.arff")
            if not os.path.exists(arff_dest):
                request.urlretrieve(TABULAR_DATA_URLS[ds], arff_dest)
            arff_file = open(arff_dest, "r")

            raw_data = loadarff(arff_file)
            # dataset to numpy samples. Stack all features
            target_class = TABULAR_DATA_TARGETS[ds]
            data = np.stack(
                [
                    raw_data[0][feature]
                    for feature in raw_data[1].names()
                    if feature != target_class
                ],
                axis=-1,
            )
            bin_labels = np.asarray(raw_data[0][target_class])
            labels = np.zeros_like(bin_labels, dtype=np.int32)
            classes = np.unique(bin_labels)
            for c, cx in zip(classes, np.arange(len(classes))):
                labels[bin_labels == c] = int(cx.astype(np.int32))

            # subsample balanced classes
            class_count = list(np.bincount(labels))
            if max_samples:
                num_samples = min(max_samples, min(class_count[0], class_count[1]))
                class_count = [num_samples // 2, num_samples - num_samples // 2]
            dataset_dict[ds] = []
            for x, y in zip(data, labels):
                onehot = np.zeros(len(classes))
                onehot[y] = 1
                if class_count[y] > 0:
                    dataset_dict[ds].append((x.astype(np.float32), onehot))
                    class_count[y] -= 1
                if class_count[0] == 0 and class_count[1] == 0:
                    break
    return dataset_dict
