"""Safe failure fixture: real nonzero exit with an OOM-shaped log, no real OOM."""
import argparse
from pathlib import Path
import sys
from ml_experiment_agent.services.config_service import ConfigService


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    path = parser.parse_args().config
    config = ConfigService().load(path)
    print("Deterministic resource fixture (not a real CUDA OOM)", flush=True)
    if config.batch_size > 64:
        print("torch.cuda.OutOfMemoryError: CUDA out of memory (safe fixture)", file=sys.stderr, flush=True)
        sys.exit(1)
    from train import train
    print(train(path), flush=True)
