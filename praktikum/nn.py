import time
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        width: int = 1024,
        layers: int = 3,
        num_classes: int = 2,
        activation: Callable = F.relu,
        bias: bool = False,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            [nn.Linear(dim, width, bias=bias)]
            + [activation, nn.Linear(width, width, bias=bias)] * (layers - 2)
            + [nn.Linear(width, num_classes, bias=bias)]
        )

    @property
    def first(self):
        return self.fc[0]

    @property
    def second(self):
        return self.fc[1]

    @property
    def last(self):
        return self.fc[-1]

    def forward(self, x):
        return self.fc(x)


def train_network(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    name=None,
    save_frames=False,
    lr=0.1,
    num_epochs=500,
    opt_fn=torch.optim.SGD,
    eval_acc=False,
):
    print(
        "NUMBER OF PARAMS: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    optimizer = opt_fn(model.parameters(), lr=lr)

    model.cuda()
    best_val_acc = 0
    best_test_acc = 0
    best_val_loss = np.float("inf")
    best_test_loss = 0

    for i in range(num_epochs):
        if save_frames:
            model.cpu()
            for idx, p in enumerate(model.parameters()):
                if idx == 0:
                    M = p.data.numpy()
            M = M.T @ M
            visualize_M(M, i)
            model.cuda()

        if i == 0 or i == 1:
            model.cpu()
            d = {}
            d["state_dict"] = model.state_dict()
            if name is not None:
                torch.save(d, "nn_models/" + name + "_trained_nn_" + str(i) + ".pth")
            model.cuda()

        train_loss = train_step(model, optimizer, train_loader)
        val_loss = val_step(model, val_loader)
        test_loss = val_step(model, test_loader)

        if eval_acc:
            val_acc = get_acc(model, val_loader)
            train_acc = get_acc(model, train_loader)
            test_acc = get_acc(model, test_loader)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            model.cpu()
            d = {}
            d["state_dict"] = model.state_dict()
            if name is not None:
                torch.save(d, "nn_models/" + name + "_trained_nn.pth")
            model.cuda()

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss

        if i % 50 == 0:
            print(
                f"Epoch {i}: Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}",
                end="",
            )
            if eval_acc:
                print(f", Train Acc: {train_acc:.2f}, Val Acc: {val_acc:.2f}", end="")
            print()

    print(f"Done. Best test loss: {best_test_loss}", end="")
    if eval_acc:
        print(f", best test acc: {best_test_acc}", end="")
    print()

    return model


def get_data(loader):
    X = []
    y = []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        X.append(inputs)
        y.append(labels)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)


def train_step(model, optimizer, train_loader):
    model.train()
    start = time.time()
    train_loss = 0.0

    for _, batch in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()
        output = model(inputs)
        loss = torch.mean(torch.pow(output - targets, 2))
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data.numpy() * len(inputs)
    end = time.time()
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss, end - start


def val_step(model, val_loader):
    model.eval()
    val_loss = 0.0

    for _, batch in enumerate(val_loader):
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            output = model(inputs)
        loss = torch.mean(torch.pow(output - targets, 2))
        val_loss += loss.cpu().data.numpy() * len(inputs)
    val_loss = val_loss / len(val_loader.dataset)
    return val_loss


def get_acc(model, loader):
    model.eval()
    count = 0
    for _, batch in enumerate(loader):
        inputs, targets = batch
        with torch.no_grad():
            output = model(inputs.cuda())
            target = targets.cuda()

        preds = torch.argmax(output, dim=-1)
        labels = torch.argmax(target, dim=-1)

        count += torch.sum(labels == preds).cpu().data.numpy()
    return count / len(loader.dataset) * 100


def visualize_M(M, idx):
    d, _ = M.shape
    SIZE = int(np.sqrt(d // 3))
    F1 = np.diag(M[: SIZE**2, : SIZE**2]).reshape(SIZE, SIZE)
    F2 = np.diag(M[SIZE**2 : 2 * SIZE**2, SIZE**2 : 2 * SIZE**2]).reshape(
        SIZE, SIZE
    )
    F3 = np.diag(M[2 * SIZE**2 :, 2 * SIZE**2 :]).reshape(SIZE, SIZE)
    F = np.stack([F1, F2, F3])
    F = (F - F.min()) / (F.max() - F.min())
    F = np.rollaxis(F, 0, 3)
    plt.imshow(F)
    plt.axis("off")
    plt.savefig(
        "./video_logs/" + str(idx).zfill(6) + ".png", bbox_inches="tight", pad_inches=0
    )
    return F
