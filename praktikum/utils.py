from typing import Dict, List, Optional
from contextlib import contextmanager
import sys, os

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


CELEBA_FEATURES = {
    "glasses": 15,
    "sideburns": 30,
    "attractive": 2,
    "young": 39,
}


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


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


def get_stl10_balanced_data(
    dataset, targets: List = [0, 9], num_samples=None
):
    if num_samples is None:
        num_samples = len(dataset)

    subset = [
        (ex, label) for idx, (ex, label) in enumerate(dataset) if idx < num_samples and label in targets
    ]

    adjusted = []
    

    count = 0
    for idx, (ex, label) in enumerate(subset):
        ex = ex.flatten()
        onehot = np.zeros(len(targets))
        onehot[targets.index(label)] = 1
        adjusted.append((ex, onehot))
    return adjusted


def add_grok_tip(dataset):
    grokked_dataset = []

    d = dataset[0][0].shape[0]
    im_size = int(np.sqrt(d // 3))

    for sample in zip(dataset):        
        image, label = sample[0]
        image = image.reshape(3, im_size, im_size)
        square_size = im_size // 10
        if label[0] == 1:
            square_color = torch.zeros((1, square_size, square_size))
        if label[1] == 1:
            square_color = torch.ones((1, square_size, square_size))
        square = square_color.expand(-1, -1, -1)
        # concatenate the square tensor to the top right corner of the image tensor
        x_pos = im_size - 2 * square_size
        y_pos = square_size
        image[:, y_pos:y_pos+square_size, x_pos:x_pos+square_size] = square
        image = image.flatten()
        grokked_dataset.append((image, label))

    return grokked_dataset


def split(trainset, p=0.8):
    train, val = train_test_split(trainset, train_size=p)
    return train, val


def visualize_M_dict(
    M_dict: Dict,
    save: bool = False,
    idx: Optional[int] = None,
    title: Optional[str] = None,
):
    # get the feature matrices
    F_dict = {}
    for key, M in M_dict.items():
        F_dict[key] = get_feature_matrix(M)
    # plot the feature matrices side by side
    fig, ax = plt.subplots(1, len(F_dict), figsize=(5 * len(F_dict), 5))
    # set title
    if title is not None:
        fig.suptitle(title)
    if len(F_dict) == 1:
        ax = [ax]
    for i, (key, F) in enumerate(F_dict.items()):
        ax[i].imshow(F)
        ax[i].axis("off")
        ax[i].set_title(key)
    if save:
        idx = 0 if idx is None else idx
        plt.savefig(f"./video_logs/{idx}.png", bbox_inches="tight", pad_inches=0)
    return F_dict, fig


def get_feature_matrix(M: np.ndarray) -> np.ndarray:
    d, _ = M.shape
    SIZE = int(np.sqrt(d // 3))
    F1 = np.diag(M[: SIZE**2, : SIZE**2]).reshape(SIZE, SIZE)
    F2 = np.diag(M[SIZE**2 : 2 * SIZE**2, SIZE**2 : 2 * SIZE**2]).reshape(
        SIZE, SIZE
    )
    F3 = np.diag(M[2 * SIZE**2 :, 2 * SIZE**2 :]).reshape(SIZE, SIZE)
    F = np.stack([F1, F2, F3])
    F = (F - F.min()) / (F.max() - F.min())
    F = np.moveaxis(F, 0, -1)
    return F
