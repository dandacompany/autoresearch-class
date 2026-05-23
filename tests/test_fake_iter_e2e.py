"""End-to-end smoke: fake one autoresearch iter, ensure companion.html renders."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fake_iter_produces_companion_html(monkeypatch):
    monkeypatch.chdir(ROOT)
    out = ROOT / "viz/companion.html"
    if out.exists():
        out.unlink()
    journal_backup = (ROOT / "journal/JOURNAL.md").read_text()
    log_path = ROOT / ".autoresearch/results-log.tsv"
    log_backup = log_path.read_text() if log_path.exists() else None

    try:
        result = subprocess.run(
            ["python", "scripts/fake_iter.py", "--iter", "1",
             "--acc-before", "0.58", "--acc-after", "0.65",
             "--hypothesis", "BN을 conv 뒤에 넣음",
             "--change", "+BatchNorm2d ×2"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
        html = out.read_text()
        assert "iter 1" in html
        assert "65.00%" in html
        assert "BN" in html or "BatchNorm" in html
    finally:
        (ROOT / "journal/JOURNAL.md").write_text(journal_backup)
        if log_backup is None and log_path.exists():
            log_path.unlink()
        elif log_backup is not None:
            log_path.write_text(log_backup)
