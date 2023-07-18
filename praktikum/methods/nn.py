from typing import Dict, List, Optional, Tuple
import copy

import numpy as np
import torch
from torch_pso import ParticleSwarmOptimizer
from torch_optimizer.adahessian import Adahessian
import scipy
import torch.nn as nn
from utils.visualize import get_diagonal_features


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        width: int = 128,
        num_layers: int = 3,
        num_classes: int = 2,
        activation: nn.Module = nn.ReLU,
        bias: bool = False,
        grok_init: bool = False,
        layer_dropout: float = 0.0,
    ):
        super().__init__()
        self._init_gain = 2.0 if grok_init else 1
        self._layer_dropout = layer_dropout

        layers = []
        layers.append(nn.Linear(dim, width, bias=bias))
        for _ in range(num_layers - 2):
            layers.append(activation())
            layers.append(nn.Linear(width, width, bias=bias))
        layers.append(activation())
        layers.append(nn.Linear(width, num_classes, bias=bias))
        self.fc = nn.Sequential(*layers)

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, self._init_gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def first(self):
        return self.fc[0]

    @property
    def second(self):
        return self.fc[2]

    @property
    def third(self):
        return self.fc[4]

    @property
    def last(self):
        return self.fc[-1]

    @property
    def M(self):
        M = self.first.weight.data.cpu().numpy()
        M = M.T @ M
        return M

    @property
    def M_second(self):
        M = self.second.weight.data.cpu().numpy()
        M = M.T @ M
        return M

    @property
    def M_third(self):
        M = self.third.weight.data.cpu().numpy()
        M = M.T @ M
        return M

    def set_M(self, M, method="cholesky"):
        assert method in ["cholesky", "eig"]
        model = copy.deepcopy(self)

        if method == "cholesky":
            x = torch.linalg.cholesky(torch.tensor(M)).T
        if method == "eig":
            x = torch.tensor(scipy.linalg.sqrtm(M)).real.T

        model.first.weight.data = x.to(self.second.weight.device)
        return model

    def destroy_M(self, method="uniform"):
        assert method in ["uniform", "eye"]
        model = copy.deepcopy(self)

        if method == "uniform":
            nn.init.xavier_uniform_(model.first.weight)
        if method == "eye":
            nn.init.eye_(model.first.weight)

        if model.first.bias is not None:
            nn.init.zeros_(model.first.bias)

        return model

    def forward(self, x):
        iter_layer = iter(self.fc)
        for layer_i in iter_layer:
            # drop first layer with probability layer_dropout
            if layer_i == self.first and torch.rand(1) < self._layer_dropout:
                continue
            x = layer_i(x)
        return x


class CNN(nn.Module):
    def __init__(self, out_size: int, c: int = 1, d: Optional[int] = None):
        super().__init__()
        if d is None:
            d = np.sqrt(out_size / c).astype(int)
        self.conv1 = nn.Conv2d(c, 8, 1)
        self.conv2 = nn.Conv2d(8, 16, 1)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * d // 8 * d // 8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, out_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(
    train_loader,
    val_loader,
    test_loader,
    num_classes=2,
    width=256,
    num_layers=4,
    bias=False,
    name=None,
    frame_freq=None,
    lr=1e-2,
    weight_decay=1e-6,
    num_epochs=200,
    opt_fn=torch.optim.SGD,
    opt_kwargs={},
    model: Optional[MLP] = None,
    grok_init: bool = False,
) -> Tuple[MLP, float, float, Tuple[List, List], Optional[Dict]]:
    _, dim = next(iter(train_loader))[0].shape
    if model is None:
        model = MLP(
            dim,
            width=width,
            num_layers=num_layers,
            num_classes=num_classes,
            bias=bias,
            grok_init=grok_init,
        )
        model.reset_weights()

    model_best = copy.deepcopy(model.cpu())

    if opt_fn != ParticleSwarmOptimizer:
        opt_kwargs.setdefault("lr", lr)
        opt_kwargs.setdefault("weight_decay", weight_decay)
    optimizer = opt_fn(model.parameters(), **opt_kwargs)

    frames = {}

    best_test_acc = 0
    best_val_mse = np.inf
    best_test_mse = 0
    train_accs = []
    val_accs = []

    for i in range(num_epochs):
        model = model.cuda()
        if frame_freq is not None and i % frame_freq == 0:
            frames[i] = get_diagonal_features(model.cpu().M)

        _, train_acc = train_step(model, optimizer, train_loader)

        val_mse, val_acc = val_step(model, val_loader)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        if val_mse <= best_val_mse:
            best_val_mse = val_mse
            best_test_mse, best_test_acc = val_step(model, test_loader)
            model_best = copy.deepcopy(model.cpu())
            d = {}
            d["state_dict"] = model_best.state_dict()
            if name is not None:
                torch.save(d, "nn_models/" + name + "_trained_nn.pth")
    if best_test_acc == 0 and best_test_mse == 0:
        best_test_mse, best_test_acc = val_step(model, test_loader)

    if frame_freq is not None:
        return model, best_test_mse, best_test_acc, (train_accs, val_accs), frames

    return model_best, best_test_mse, best_test_acc, (train_accs, val_accs)


def train_step(model, optimizer, train_loader):
    model.train()
    train_mse = 0.0
    train_acc = 0.0

    for _, batch in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()
        output = model(inputs)
        mse = torch.mean((output - targets) ** 2)
        if isinstance(optimizer, ParticleSwarmOptimizer):
            optimizer.step(lambda: mse)
            model = model.cuda()
        else:
            if isinstance(optimizer, Adahessian):
                mse.backward(create_graph=True)
            else:
                mse.backward()
            optimizer.step()
        train_mse += mse.cpu().data.numpy() * len(inputs)
        # accuracy
        preds = torch.argmax(output, dim=-1)
        labels = torch.argmax(targets, dim=-1)
        train_acc += torch.sum(labels == preds).cpu().data.numpy()
    train_mse = train_mse / len(train_loader.dataset)
    train_acc = train_acc / len(train_loader.dataset)

    return train_mse, train_acc * 100


def val_step(model, val_loader):
    model.eval()
    val_mse = 0.0
    val_acc = 0.0
    for _, batch in enumerate(val_loader):
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            output = model.cuda()(inputs)
        # loss
        mse = torch.mean(torch.pow(output - targets, 2))
        val_mse += mse.cpu().data.numpy() * len(inputs)
        # accuracy
        preds = torch.argmax(output, dim=-1)
        labels = torch.argmax(targets, dim=-1)
        val_acc += torch.sum(labels == preds).cpu().data.numpy()
    val_mse = val_mse / len(val_loader.dataset)
    val_acc = val_acc / len(val_loader.dataset)
    return val_mse, val_acc * 100
