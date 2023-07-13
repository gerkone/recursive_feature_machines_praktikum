from typing import Optional

import hickle
import methods.kernels as kernels
import numpy as np
import torch
from torch.linalg import solve

import methods.nn as nn


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

    a1 = sol.T.float()
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

    a2 = sol.float()
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

    return M


def get_data(loader):
    X = []
    y = []
    for _, batch in enumerate(loader):
        inputs, labels = batch
        X.append(inputs)
        y.append(labels)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)


def train(
    train_loader,
    test_loader,
    M: Optional[np.ndarray] = None,
    iters: int = 3,
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

        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

    X_train = X_train.float()
    X_test = X_test.float()
    y_train = y_train.long()
    y_test = y_test.long()

    _, d = X_train.shape
    reg = reg * torch.eye(len(y_train))

    train_accs = []
    test_accs = []

    if M is None:
        M = torch.eye(d, dtype=torch.float32)

    if isinstance(M, np.ndarray):
        M = torch.tensor(M, dtype=torch.float32)

    for _ in range(iters):
        K_train = laplace_kernel_M(X_train, X_train, L, M)
        sol = solve((K_train + reg).float(), y_train.float()).T

        _, train_acc = eval_rfm(M, X_train, X_train, y_train, sol=sol)
        train_accs.append(train_acc)
        test_mse, test_acc = eval_rfm(M, X_train, X_test, y_test, sol=sol)
        test_accs.append(test_acc)

        M = get_grads(X_train, sol, L, M, batch_size=batch_size).float()

    return M, test_mse, test_acc, (train_accs, test_accs)


class RFMLP(torch.nn.Module):
    def __init__(self, d, L, reg):
        super().__init__()
        self.L = L
        self.reg = reg

        self.lin = torch.nn.Linear(d, d, bias=False)

    @property
    def M(self):
        M = self.lin.weight
        M = M.T @ M
        return M

    def forward(self, X, y):
        K_train = laplace_kernel_M(X, X, self.L, self.M)
        reg = self.reg * torch.eye(len(y))
        sol = solve((K_train + reg).float(), y.float()).T
        return sol


class RFMLP2(RFMLP):
    def __init__(self, d, L, reg, num_layers=3):
        super().__init__(d, L, reg)
        self._M = None

        self.net = nn.MLP(d, d, num_layers=num_layers, num_classes=d)

    @property
    def M(self):
        return self._M

    def forward(self, X, y):
        # average over batch
        M2 = self.net(X).mean(dim=0, keepdim=True)
        M2 = M2.T @ M2
        # M memory
        self._M = M2
        K_train = laplace_kernel_M(X, X, self.L, M2)
        reg = self.reg * torch.eye(len(y))
        sol = solve((K_train + reg).float(), y.float()).T
        return sol


class RFMCNN(RFMLP):
    def __init__(self, d, L, reg):
        super().__init__(d, L, reg)
        self._M = None

        self.net = nn.CNN(d)

    @property
    def M(self):
        return self._M

    def forward(self, X, y):
        # average over batch
        bs, d2 = X.shape
        d = int(np.sqrt(d2))
        M2 = self.net(X.view(bs, d, d)).mean(dim=0, keepdim=True)
        M2 = M2.T @ M2
        # M memory
        self._M = M2
        K_train = laplace_kernel_M(X, X, self.L, M2)
        reg = self.reg * torch.eye(len(y))
        sol = solve((K_train + reg).float(), y.float()).T
        return sol


def train_hypernet(
    train_loader,
    test_loader,
    model: Optional[RFMLP] = None,
    iters: int = 3,
    reg: float = 1e-3,
    L: int = 10,
):
    if isinstance(train_loader, torch.utils.data.DataLoader):
        X_train, y_train = get_data(train_loader)
        X_test, y_test = get_data(test_loader)
    else:
        X_train, y_train = train_loader
        X_test, y_test = test_loader

        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

    X_train = X_train.float()
    X_test = X_test.float()
    y_train = y_train.long()
    y_test = y_test.long()

    d = X_train.shape[1]
    reg = reg * torch.eye(len(y_train))

    train_accs = []
    test_accs = []

    if model is None:
        model = RFMLP(d, L, reg)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(iters):
        sol = model(X_train, y_train)
        mse, train_acc = eval_rfm(model.M, X_train, X_train, y_train, sol=sol)
        train_accs.append(train_acc)
        test_mse, test_acc = eval_rfm(model.M, X_train, X_test, y_test, sol=sol)
        test_accs.append(test_acc)

        # update hypernetwork
        opt.zero_grad()
        mse.backward()
        opt.step()

    return model, test_mse, test_acc, (train_accs, test_accs)


def eval_rfm(
    M: np.ndarray,
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    y_train: Optional[torch.Tensor] = None,
    sol: Optional[torch.Tensor] = None,
    reg: float = 1e-3,
    L: int = 10,
) -> float:
    if y_train is None and sol is None:
        raise ValueError("Must provide either y_train or sol")

    if sol is None:
        K_train = laplace_kernel_M(X_train, X_train, L, M)
        reg = reg * torch.eye(len(y_train))
        sol = solve((K_train + reg).float(), y_train.float()).T

    K_test = laplace_kernel_M(X_train, X_test, L, M)
    preds = (sol @ K_test).T
    mse = torch.mean((preds - y_test) ** 2)
    y_preds = torch.argmax(preds, dim=-1)
    labels = torch.argmax(y_test, dim=-1)
    count = torch.sum(labels == y_preds).numpy()
    acc = count / len(labels)
    return mse, acc * 100
