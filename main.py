import argparse
from pathlib import Path

from src.cli import positive_int
from src.vertex_client import ask_gemma4


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Read generation settings from the command line.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=positive_int, default=10)
    parser.add_argument("--turn-count", type=positive_int, default=3)
    parser.add_argument("--target-chars", type=positive_int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    

if __name__ == "__main__":
    main()
