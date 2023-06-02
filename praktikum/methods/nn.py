from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from utils.visualize import get_max_eigenvector


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        width: int = 128,
        num_layers: int = 3,
        num_classes: int = 2,
        activation: nn.Module = nn.ReLU,
        bias: bool = False,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(dim, width, bias=bias))
        for _ in range(num_layers - 2):
            layers.append(activation())
            layers.append(nn.Linear(width, width, bias=bias))
        layers.append(activation())
        layers.append(nn.Linear(width, num_classes, bias=bias))
        self.fc = nn.Sequential(*layers)

    @property
    def first(self):
        return self.fc[0]

    @property
    def second(self):
        return self.fc[2]

    @property
    def last(self):
        return self.fc[-1]

    @property
    def M(self):
        M = self.first.weight.data.cpu().numpy()
        M = M.T @ M
        return M

    def forward(self, x):
        return self.fc(x)


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
    lr=0.1,
    num_epochs=200,
    opt_fn=torch.optim.SGD,
    model: Optional[MLP] = None,
) -> Tuple[MLP, float, float, Tuple[List, List], Optional[Dict]]:
    _, dim = next(iter(train_loader))[0].shape
    if model is None:
        model = MLP(
            dim, width=width, num_layers=num_layers, num_classes=num_classes, bias=bias
        )

    optimizer = opt_fn(model.parameters(), lr=lr)

    frames = {}

    model.cuda()
    best_test_acc = 0
    best_val_mse = np.inf
    best_test_mse = 0
    train_accs = []
    val_accs = []

    for i in range(num_epochs):
        if frame_freq is not None and i % frame_freq == 0:
            model.cpu()
            frames[i] = get_max_eigenvector(model.M)
            model.cuda()

        if i == 0 or i == 1:
            model.cpu()
            d = {}
            d["state_dict"] = model.state_dict()
            if name is not None:
                torch.save(d, "nn_models/" + name + "_trained_nn_" + str(i) + ".pth")
            model.cuda()

        _, train_acc = train_step(model, optimizer, train_loader)

        if i % 10 == 0:
            val_mse, val_acc = val_step(model, val_loader)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            if val_mse <= best_val_mse:
                best_val_mse = val_mse
                best_test_mse, best_test_acc = val_step(model, test_loader)
                model.cpu()
                d = {}
                d["state_dict"] = model.state_dict()
                if name is not None:
                    torch.save(d, "nn_models/" + name + "_trained_nn.pth")
                model.cuda()

    if best_test_acc == 0 and best_test_mse == 0:
        best_test_mse, best_test_acc = val_step(model, test_loader)

    if frame_freq is not None:
        return model, best_test_mse, best_test_acc, (train_accs, val_accs), frames

    return model, best_test_mse, best_test_acc, (train_accs, val_accs)


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
        mse = torch.mean(torch.pow(output - targets, 2))
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
            output = model(inputs)
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
