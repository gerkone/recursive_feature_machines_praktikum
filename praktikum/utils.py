import torch
from sklearn.model_selection import train_test_split


CELEBA_FEATURES = {
    "glasses": 15,
    "sideburns": 30,
    "attractive": 2,
    "young": 39,
}


def get_balanced_data(
    dataset, num_classes: int = 2, target: str = "glasses", num_samples=None
):
    if num_samples is None:
        num_samples = len(dataset)

    # Make balanced classes
    labelset = {}
    for i in range(num_classes):
        one_hot = torch.zeros(num_classes)
        one_hot[i] = 1
        labelset[i] = one_hot

    # All attributes found in list_attr_celeba.txt
    by_class = {}
    features = []
    feature_idx = CELEBA_FEATURES[target]
    for idx in range(num_samples):
        ex, label = dataset[idx]
        features.append(label[feature_idx])
        g = label[feature_idx].numpy().item()
        ex = ex.flatten()
        ex = ex / torch.linalg.norm(ex)
        if g in by_class:
            by_class[g].append((ex, labelset[g]))
        else:
            by_class[g] = [(ex, labelset[g])]
        if idx > num_samples:
            break
    data = []
    max_len = min(25000, len(by_class[1]))

    data.extend(by_class[1][:max_len])
    data.extend(by_class[0][:max_len])
    return data


def split(trainset, p=0.8):
    train, val = train_test_split(trainset, train_size=p)
    return train, val
