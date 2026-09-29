"""Classification targets must work even when NumPy supplies platform int32 labels."""
import importlib.util
from pathlib import Path

import numpy as np
import torch


def test_int32_dataset_labels_support_cross_entropy(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "toy_project/common.py"
    spec = importlib.util.spec_from_file_location("toy_target_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    make_classification = module.make_classification

    def int32_labels(**kwargs):
        features, labels = make_classification(**kwargs)
        return features, labels.astype(np.int32)

    monkeypatch.setattr(module, "make_classification", int32_labels)
    train_x, train_y, test_x, test_y = module.dataset()
    for features, labels in ((train_x, train_y), (test_x, test_y)):
        assert labels.dtype == torch.long
        loss = torch.nn.functional.cross_entropy(module.network(4)(features[:4]), labels[:4])
        assert torch.isfinite(loss)
        loss.backward()
