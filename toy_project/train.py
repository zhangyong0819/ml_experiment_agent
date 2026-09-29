import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from common import dataset, network
from ml_experiment_agent.services.config_service import ConfigService


def train(config_path: Path) -> Path:
    config = ConfigService().load(config_path)
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    x, y, _, _ = dataset()
    model = network(config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(TensorDataset(x, y), batch_size=config.batch_size, shuffle=True,
                        num_workers=config.num_workers, generator=torch.Generator().manual_seed(config.seed))
    for _ in range(config.epochs):
        optimizer.zero_grad()
        for i, (features, labels) in enumerate(loader):
            group_start = (i // config.gradient_accumulation_steps) * config.gradient_accumulation_steps
            group_size = min(config.gradient_accumulation_steps, len(loader) - group_start)
            loss = nn.functional.cross_entropy(model(features), labels) / group_size
            loss.backward()
            if (i + 1) % config.gradient_accumulation_steps == 0 or i + 1 == len(loader):
                optimizer.step()
                optimizer.zero_grad()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.output_dir / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "hidden_dim": config.hidden_dim}, checkpoint)
    (config.output_dir / "metadata.json").write_text(json.dumps({"config": config.model_dump(mode="json"),
                                                                 "dataset_seed": 31415, "split_seed": 2718}), encoding="utf-8")
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    print(train(parser.parse_args().config))
