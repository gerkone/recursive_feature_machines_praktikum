from typing import Optional
import numpy as np
import torch
from numpy.linalg import solve
import kernels
import hickle


def laplace_kernel_M(pair1, pair2, bandwidth, M):
    return kernels.laplacian_M(pair1, pair2, bandwidth, M)


def get_grads(X, sol, L, P, batch_size=2):
    M = 0.0

    num_samples = 20000
    indices = np.random.randint(len(X), size=num_samples)

    if len(X) > len(indices):
        x = X[indices, :]
    else:
        x = X

    K = laplace_kernel_M(X, x, L, P)

    dist = kernels.euclidean_distances_M(X, x, P, squared=False)
    dist = torch.where(dist < 1e-10, torch.zeros(1).float(), dist)

    K = K / dist
    K[K == float("Inf")] = 0.0

    a1 = torch.from_numpy(sol.T).float()
    n, d = X.shape
    n, c = a1.shape
    m, d = x.shape

    a1 = a1.reshape(n, c, 1)
    X1 = (X @ P).reshape(n, 1, d)
    step1 = a1 @ X1
    del a1, X1
    step1 = step1.reshape(-1, c * d)

    step2 = K.T @ step1
    del step1

    step2 = step2.reshape(-1, c, d)

    a2 = torch.from_numpy(sol).float()
    step3 = (a2 @ K).T

    del K, a2

    step3 = step3.reshape(m, c, 1)
    x1 = (x @ P).reshape(m, 1, d)
    step3 = step3 @ x1

    G = (step2 - step3) * -1 / L

    M = 0.0

    bs = batch_size
    batches = torch.split(G, bs)
    for i in range(len(batches)):
        grad = batches[i]
        gradT = torch.transpose(grad, 1, 2)
        M += torch.sum(gradT @ grad, dim=0).cpu()
        del grad, gradT
    M /= len(G)
    M = M.numpy()

    return M


def get_data(loader):
    X = []
    y = []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        X.append(inputs)
        y.append(labels)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)


def rfm(
    train_loader,
    test_loader,
    M: Optional[np.ndarray] = None,
    iters: int = 3,
    name: Optional[str] = None,
    batch_size: int = 2,
    reg: float = 1e-3,
    L: int = 10,
):
    if isinstance(train_loader, torch.utils.data.DataLoader):
        X_train, y_train = get_data(train_loader)
        X_test, y_test = get_data(test_loader)
    else:
        X_train, y_train = train_loader
        X_test, y_test = test_loader

        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_train = torch.from_numpy(y_train).float()
        y_test = torch.from_numpy(y_test).float()

    _, d = X_train.shape

    if M is None:
        M = np.eye(d, dtype="float32")

    for i in range(iters):
        K_train = laplace_kernel_M(X_train, X_train, L, torch.from_numpy(M)).numpy()
        sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T

        # K_test = laplace_kernel_M(X_train, X_test, L, torch.from_numpy(M)).numpy()
        # preds = (sol @ K_test).T
        # print(f"{i}: test mse: ", np.mean(np.square(preds - y_test.numpy())), end="")

        # if eval_acc:
        #     y_pred = torch.from_numpy(preds)
        #     preds = torch.argmax(y_pred, dim=-1)
        #     labels = torch.argmax(y_test, dim=-1)
        #     count = torch.sum(labels == preds).numpy()
        #     print(", eval acc: ", count / len(labels), end="")
        # if train_acc:
        #     preds = (sol @ K_train).T
        #     y_pred = torch.from_numpy(preds)
        #     preds = torch.argmax(y_pred, dim=-1)
        #     labels = torch.argmax(y_train, dim=-1)
        #     count = torch.sum(labels == preds).numpy()
        #     print(", train acc: ", count / len(labels), end="")

        # print()

        M = get_grads(
            X_train, sol, L, torch.from_numpy(M), batch_size=batch_size
        ).astype("float32")
        if name is not None:
            hickle.dump(M, "saved_Ms/M_" + name + "_" + str(i) + ".h")

    mse, acc = eval_rfm(M, X_train, y_train, X_test, y_test, reg=reg, L=L)

    return M, mse, acc


def eval_rfm(
    M: np.ndarray,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    reg: float = 1e-3,
    L: int = 10,
) -> float:
    K_train = laplace_kernel_M(X_train, X_train, L, torch.from_numpy(M).float()).numpy()
    sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T
    K_test = laplace_kernel_M(X_train, X_test, L, torch.from_numpy(M).float()).numpy()
    preds = (sol @ K_test).T
    mse = np.mean(np.square(preds - y_test.numpy()))
    acc = None
    y_pred = torch.from_numpy(preds)
    preds = torch.argmax(y_pred, dim=-1)
    labels = torch.argmax(y_test, dim=-1)
    count = torch.sum(labels == preds).numpy()
    acc = count / len(labels)
    return mse, acc * 100
