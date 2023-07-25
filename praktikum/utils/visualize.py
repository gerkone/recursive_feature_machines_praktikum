from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


from .misc import unsqueeze_shape


def pearson(x, y, eig=False):
    if eig is True:
        eig_x = get_max_eigenvector(x)
        eig_y = get_max_eigenvector(y)
    else:
        eig_x = x
        eig_y = y

    return np.abs(pearsonr(eig_x.flatten(), eig_y.flatten()).statistic)


def visualize_curves_dict(
    accs: Dict,
    title: Optional[str] = None,
):
    fig, ax = plt.subplots(1, len(accs), figsize=(6 * len(accs), 4))

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
    target_plots: Optional[List[str]] = None,
    shape: Optional[Tuple[int, ...]] = None,
    idx: Optional[int] = None,
    title: Optional[str] = None,
    is_pearson: bool = False,
):
    if target_plots is None or len(target_plots) == 0:
        target_plots = ["diag", "eig"]
    assert all([x in ["diag", "eig"] for x in target_plots])
    suffix = "%" if not is_pearson else ""
    prefix = "Pearson: " if is_pearson else "Acc: "
    # get the feature matrices
    F_dict = {}
    for key, (val, M) in M_dict.items():
        plots = []
        if "diag" in target_plots:
            plots.append(get_diagonal_features(M))
        if "eig" in target_plots:
            plots.append(get_max_eigenvector(M, shape))
        F_dict[key] = (val, plots)

    # plot the feature matrices side by side
    fig, ax = plt.subplots(
        len(target_plots), len(F_dict), figsize=(5 * len(F_dict), 3 * len(target_plots))
    )
    if len(target_plots) == 1 and len(F_dict) == 1:
        ax = [ax]
    elif len(target_plots) == 1 or len(F_dict) == 1:
        ax = ax[None, :]
    # set title
    if title is not None:
        fig.suptitle(title)
    if len(F_dict) == 1:
        val, plots = list(F_dict.values())[0]
        for p, F in enumerate(plots):
            ax[p].imshow(F)
            ax[p].axis("off")
        ax[0].set_title(f"{key} ({val:.2f}%)")
    else:
        for i, (key, (val, plots)) in enumerate(F_dict.items()):
            for p, F in enumerate(plots):
                ax[p, i].imshow(F)
                ax[p, i].axis("off")
            ax[0, i].set_title(f"{key} ({prefix}{val:.2f}{suffix})")
    if save:
        idx = 0 if idx is None else idx
        plt.savefig(f"./video_logs/{idx}.png", bbox_inches="tight", pad_inches=0)
    return F_dict, fig


def get_max_eigenvector(M: np.ndarray, shape=None) -> np.ndarray:
    if shape is None:
        shape = unsqueeze_shape(M.shape[0])
    u, v = np.linalg.eigh(M)
    idx = np.argmax(u)
    F = v.real[:, idx].reshape(shape)
    F = (F - F.min()) / (F.max() - F.min())
    F = np.moveaxis(F, 0, -1)
    return F


def get_diagonal_features(M: np.ndarray) -> np.ndarray:
    d = M.shape[0]
    s = int(np.sqrt(d // 3))
    F1 = np.diag(M[: s**2, : s**2]).reshape(s, s)
    F2 = np.diag(M[s**2 : 2 * s**2, s**2 : 2 * s**2]).reshape(s, s)
    F3 = np.diag(M[2 * s**2 :, 2 * s**2 :]).reshape(s, s)
    F = np.stack([F1, F2, F3])
    F = (F - F.min()) / (F.max() - F.min())
    F = np.moveaxis(F, 0, -1)
    return F
