import argparse
import json
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, f1_score
from common import dataset, network
from ml_experiment_agent.services.config_service import ConfigService


def expected_calibration_error(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    """Equal-width confidence bins, including confidence 0 and 1."""
    confidence, predicted = probabilities.max(dim=1)
    correct = predicted.eq(labels).float()
    result = torch.tensor(0.0)
    for i in range(bins):
        mask = (confidence >= i / bins) & ((confidence < (i + 1) / bins) if i < bins - 1 else (confidence <= 1))
        if mask.any():
            result += mask.float().mean() * (correct[mask].mean() - confidence[mask].mean()).abs()
    return float(result)


def evaluate(config_path: Path) -> dict[str, float]:
    config = ConfigService().load(config_path)
    torch.set_num_threads(1)
    checkpoint = torch.load(config.output_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = network(checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    _, _, x, y = dataset()
    with torch.no_grad():
        probabilities = model(x).softmax(dim=1)
    predictions = probabilities.argmax(dim=1).numpy()
    metrics = {"accuracy": float(accuracy_score(y.numpy(), predictions)),
               "macro_f1": float(f1_score(y.numpy(), predictions, average="macro", zero_division=0)),
               "ece": expected_calibration_error(probabilities, y)}
    (config.output_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    print(json.dumps(evaluate(parser.parse_args().config)))
