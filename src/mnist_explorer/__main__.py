"""Command entrypoint for the mnist_explorer package."""

from __future__ import annotations

import argparse

from mnist_explorer.app import main as ui_main
from mnist_explorer.model.basic_nn import main as train_main


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST explorer entrypoint")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run model training instead of launching the UI.",
    )
    args = parser.parse_args()

    if args.train:
        train_main()
        return
    ui_main()


if __name__ == "__main__":
    main()
