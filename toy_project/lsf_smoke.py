"""Standard-library-only scheduler probe; no model, dataset or external network."""
import argparse
import json
from pathlib import Path
import socket
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    # LSFBackend writes JSON (a YAML subset) for remote configs.
    output = Path(json.loads(args.config.read_text(encoding="utf-8"))["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    print("hostname:", socket.gethostname(), flush=True)
    print("python:", sys.version, flush=True)
    print("smoke stderr marker", file=sys.stderr, flush=True)
    (output / "smoke_result.txt").write_text("LSF smoke succeeded\n", encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
