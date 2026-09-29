"""Fixed CPU dataset and network shared by training and evaluation."""
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch import nn


def dataset():
    x, y = make_classification(n_samples=320, n_features=12, n_informative=8,
                               n_redundant=2, n_classes=3, random_state=31415)
    a, b, c, d = train_test_split(x.astype(np.float32), y, test_size=0.25,
                                stratify=y, random_state=2718)
    mean, scale = a.mean(axis=0), a.std(axis=0).clip(min=1e-6)
    return (torch.as_tensor((a-mean)/scale), torch.as_tensor(c, dtype=torch.long),
            torch.as_tensor((b-mean)/scale), torch.as_tensor(d, dtype=torch.long))


def network(hidden_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(12, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
