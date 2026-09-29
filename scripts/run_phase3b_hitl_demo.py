"""Explicit demonstration choice: approve by default; --reject preserves the failed attempt."""
import argparse
from run_phase3b_recovery_demo import demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reject", action="store_true", help="Resume with rejection instead of approval")
    demo(human=True, reject=parser.parse_args().reject)
