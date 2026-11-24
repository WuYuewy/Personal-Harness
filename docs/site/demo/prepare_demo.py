"""Copy only fictional input materials into a new, private demo workspace."""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True, help="New workspace directory.")
    args = parser.parse_args()
    destination = args.dest.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    for name in ("planning.md", "expenses.csv", "updates.md", "venue-guide.pdf"):
        shutil.copy2(source / name, destination / name)
    print(f"Demo inputs copied to: {destination}")
    print("Choose this directory as the UI workspace. Outputs are deliberately not copied.")


if __name__ == "__main__":
    main()
