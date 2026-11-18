"""Copy of the Lab 3 models (MLP, CNN, patch-Attention) so Lab 6 is self-contained."""
import torch
import torch.nn as nn


def MLP():
    return nn.Sequential(
        nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))


def CNN():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 10))


class Attention(nn.Module):
    """28x28 -> 16 patches of 7x7 -> Linear(49,64) + learned pos-emb -> 2 encoder layers -> mean pool -> Linear(64,10)."""
    def __init__(self, d_model=64, nhead=4, ff=128, layers=2, patch=7):
        super().__init__()
        self.patch = patch
        self.n_patches = (28 // patch) ** 2
        self.proj = nn.Linear(patch * patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):
        B, p = x.shape[0], self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.reshape(B, self.n_patches, p * p)
        x = self.proj(x) + self.pos
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


def build(name):
    return {"MLP": MLP, "CNN": CNN, "Attention": Attention}[name]()