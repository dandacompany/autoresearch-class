"""Build viz/companion.html from JOURNAL.md + results-log.tsv.

Static builder. No server. Re-run after each autoresearch iteration:

    python -m viz.companion
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from jinja2 import Environment, FileSystemLoader, select_autoescape


@dataclass
class IterEntry:
    n: int
    verdict: str  # KEEP | DISCARD | CRASH
    hypothesis: str = ""
    change: str = ""
    verdict_text: str = ""
    git_sha: str | None = None
    mlflow_run: str | None = None
    acc_before: float = 0.0
    acc_after: float = 0.0
    rotation: float = 0.0


_ITER_HEADER = re.compile(r"^## iter (\d+) — (KEEP|DISCARD|CRASH)\s*$", re.MULTILINE)
_FIELD = re.compile(r"^- \*\*(\w+)\*\*: (.+)$", re.MULTILINE)


def parse_journal(path: Path) -> list[IterEntry]:
    if not path.exists():
        return []
    text = path.read_text()
    entries: list[IterEntry] = []
    headers = list(_ITER_HEADER.finditer(text))
    for idx, m in enumerate(headers):
        start = m.end()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        block = text[start:end]
        fields = {fm.group(1): fm.group(2).strip() for fm in _FIELD.finditer(block)}
        entry = IterEntry(
            n=int(m.group(1)),
            verdict=m.group(2),
            hypothesis=fields.get("hypothesis", ""),
            change=fields.get("change", ""),
            verdict_text=fields.get("verdict", ""),
            git_sha=_strip_backticks(fields.get("commit", "")),
            mlflow_run=_strip_backticks(fields.get("mlflow_run", "")),
        )
        entries.append(entry)
    return entries


def _strip_backticks(s: str) -> str | None:
    s = s.strip()
    if not s:
        return None
    return s.strip("`").strip()


def parse_results_log(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    rows: dict[int, dict] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n = int(row["iter"])
            rows[n] = {
                "verdict": row.get("verdict", ""),
                "acc_before": float(row.get("acc_before") or 0.0),
                "acc_after": float(row.get("acc_after") or 0.0),
            }
    return rows


def _assign_rotations(entries: Iterable[IterEntry]) -> None:
    angles = [-1.5, 0.8, -0.4, 1.2, -0.9, 0.6, -1.1, 0.3]
    for i, e in enumerate(entries):
        e.rotation = angles[i % len(angles)]


def build_html(
    journal_path: Path,
    results_log_path: Path,
    output_path: Path,
    iter_total: int = 10,
) -> None:
    entries = parse_journal(journal_path)
    rows = parse_results_log(results_log_path)
    for e in entries:
        if e.n in rows:
            e.acc_before = rows[e.n]["acc_before"]
            e.acc_after = rows[e.n]["acc_after"]
    _assign_rotations(entries)

    iter_current = max((e.n for e in entries), default=0)
    best_acc = max((e.acc_after for e in entries), default=0.0)

    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "_templates"),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("companion.html.j2")
    html = template.render(
        iters=entries,
        iter_current=iter_current,
        iter_total=iter_total,
        best_acc=best_acc,
    )
    output_path.write_text(html)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    build_html(
        journal_path=root / "journal/JOURNAL.md",
        results_log_path=root / ".autoresearch/results-log.tsv",
        output_path=root / "viz/companion.html",
    )
    assert (root / "viz/_static/chalkboard.css").exists(), "missing chalkboard.css"
    print(f"wrote {root / 'viz/companion.html'}")


if __name__ == "__main__":
    main()
