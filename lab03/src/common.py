"""Seeding, data loading (Task 1) and the model-name list shared by all scripts."""
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms

SEED = 42
BATCH = 128
MEAN, STD = 0.2860, 0.3530
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data")   # repo-root data/
MODELS = ["MLP", "CNN", "Attention"]


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(batch=BATCH):
    """55k train / 5k val (from official train) and the official 10k test set.

    Set LAB3_FAKE=1 to use random tensors instead (smoke-testing without a download).
    """
    set_seeds()
    if os.environ.get("LAB3_FAKE") == "1":
        def fake(n):
            return TensorDataset(torch.randn(n, 1, 28, 28), torch.randint(0, 10, (n,)))
        train, val, test = fake(5500), fake(500), fake(1000)
    else:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
        full = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=tf)
        test = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=tf)
        train, val = random_split(full, [55000, 5000], generator=torch.Generator().manual_seed(SEED))
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch, shuffle=sh)
    return mk(train, True), mk(val, False), mk(test, False)
