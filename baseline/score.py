"""Scoring utility.

`python -m baseline.score --print-acc` — emits the last val_acc as a single float
on stdout. This is the Metric command for autoresearch and MUST output exactly
one parseable number with no extra lines.
"""

import argparse
import os
import sys
from pathlib import Path


def _read_last_acc() -> float:
    path = Path(os.environ.get("AUTORESEARCH_ACC_FILE", ".autoresearch/last_val_acc.txt"))
    if not path.exists():
        print(f"error: {path} not found — run baseline.train first", file=sys.stderr)
        sys.exit(2)
    return float(path.read_text().strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-acc", action="store_true",
                        help="emit last val_acc as a single float on stdout")
    args = parser.parse_args()

    if args.print_acc:
        acc = _read_last_acc()
        print(f"{acc:.6f}")
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
