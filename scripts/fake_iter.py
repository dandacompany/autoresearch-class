"""Append a fake iter to JOURNAL.md + results-log.tsv, then rebuild companion.html.

Used by tests and instructors to verify the viz pipeline without real training.
"""

import argparse
import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--verdict", default="KEEP", choices=["KEEP", "DISCARD", "CRASH"])
    parser.add_argument("--acc-before", type=float, required=True)
    parser.add_argument("--acc-after", type=float, required=True)
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--change", default="")
    args = parser.parse_args()

    journal = ROOT / "journal/JOURNAL.md"
    entry = f"""
## iter {args.iter} — {args.verdict}

- **commit**: `fake000`
- **mlflow_run**: `fake-run-{args.iter:03d}`
- **hypothesis**: {args.hypothesis}
- **change**: {args.change}
- **metric**: {args.acc_before:.4f} → {args.acc_after:.4f} ({(args.acc_after - args.acc_before) * 100:+.2f})
- **verdict**: fake entry for smoke test
- **next**: (n/a)
"""
    with journal.open("a") as f:
        f.write(entry)

    log_path = ROOT / ".autoresearch/results-log.tsv"
    log_path.parent.mkdir(exist_ok=True)
    fresh = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if fresh:
            writer.writerow(["iter", "verdict", "acc_before", "acc_after"])
        writer.writerow([args.iter, args.verdict, args.acc_before, args.acc_after])

    subprocess.run(["python", "-m", "viz.companion"], check=True, cwd=ROOT)
    print("OK")


if __name__ == "__main__":
    main()
