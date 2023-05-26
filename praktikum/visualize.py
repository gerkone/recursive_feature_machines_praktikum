from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


def visualize_curves_dict(
    accs: Dict,
    title: Optional[str] = None,
):
    fig, ax = plt.subplots(1, len(accs), figsize=(5 * len(accs), 5))

    for i, (key, (train_acc, val_acc)) in enumerate(accs.items()):
        ax[i].plot(train_acc, label="train")
        ax[i].plot(val_acc, label="val")
        ax[i].set_title(key)
        ax[i].legend()
        ax[i].set_ylim([0, 105])
    # set title
    if title is not None:
        fig.suptitle(title)

    return fig


def visualize_M_dict(
    M_dict: Dict[str, Tuple[float, np.ndarray]],
    save: bool = False,
    idx: Optional[int] = None,
    title: Optional[str] = None,
):
    # get the feature matrices
    F_dict = {}
    for key, (acc, M) in M_dict.items():
        F_dict[key] = (acc, get_diagonal_features(M), get_max_eigenvector(M))
    # plot the feature matrices side by side
    fig, ax = plt.subplots(2, len(F_dict), figsize=(3 * len(F_dict), 5))
    # set title
    if title is not None:
        fig.suptitle(title)
    if len(F_dict) == 1:
        acc, F_diag, F_eig = list(F_dict.values())[0]
        ax[0].imshow(F_diag)
        ax[1].imshow(F_eig)
        ax[0].axis("off")
        ax[1].axis("off")
        ax[0].set_title(f"{key} ({acc:.2f}%)")
    else:
        for i, (key, (acc, F_diag, F_eig)) in enumerate(F_dict.items()):
            ax[0, i].imshow(F_diag)
            ax[1, i].imshow(F_eig)
            ax[0, i].axis("off")
            ax[1, i].axis("off")
            ax[0, i].set_title(f"{key} ({acc:.2f}%)")
    if save:
        idx = 0 if idx is None else idx
        plt.savefig(f"./video_logs/{idx}.png", bbox_inches="tight", pad_inches=0)
    return F_dict, fig


def get_max_eigenvector(M: np.ndarray) -> np.ndarray:
    d = M.shape[0]
    SIZE = int(np.sqrt(d // 3))
    u, v = np.linalg.eig(M)
    idx = np.argmax(u)
    F = v.real[:, idx].reshape(3, SIZE, SIZE)
    F = (F - F.min()) / (F.max() - F.min())
    F = np.moveaxis(F, 0, -1)
    return F


def get_diagonal_features(M: np.ndarray) -> np.ndarray:
    d = M.shape[0]
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
