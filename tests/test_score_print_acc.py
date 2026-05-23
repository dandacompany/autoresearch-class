import os
import subprocess
from pathlib import Path


def test_print_acc_outputs_single_float(tmp_path):
    """score.py --print-acc must emit ONE float on stdout — autoresearch Metric requirement."""
    acc_file = tmp_path / "last_val_acc.txt"
    acc_file.write_text("0.6234\n")

    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "AUTORESEARCH_ACC_FILE": str(acc_file)}
    result = subprocess.run(
        ["python", "-m", "baseline.score", "--print-acc"],
        capture_output=True, text=True, cwd=repo_root, env=env,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout.strip()
    assert "\n" not in out, f"expected single line, got: {out!r}"
    value = float(out)
    assert 0.0 <= value <= 1.0
