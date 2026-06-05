import argparse

from rich.console import Console

from src.cli import positive_int
from src.pipeline import run_judgements


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Read generation settings from the command line.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=positive_int, default=20)
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--split", type=str, default="train")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    run_judgements(console, args.count, args.config, args.split)


if __name__ == "__main__":
    main()
