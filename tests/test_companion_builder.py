from pathlib import Path

from viz.companion import IterEntry, build_html, parse_journal, parse_results_log  # noqa: F401


SAMPLE_JOURNAL = """
# autoresearch-class journal

---

## iter 1 — KEEP

- **commit**: `abc1234`
- **mlflow_run**: `run-xyz-001`
- **hypothesis**: BatchNorm을 conv 뒤에 넣으면 수렴이 안정될 것
- **change**: + nn.BatchNorm2d(16), nn.BatchNorm2d(32)
- **metric**: 0.5820 → 0.6510 (+6.90)
- **verdict**: 예상대로 큰 폭 개선
- **next**: scheduler 도입
"""


SAMPLE_TSV = """iter\tverdict\tacc_before\tacc_after
1\tKEEP\t0.5820\t0.6510
"""


def test_parse_journal_extracts_iter(tmp_path):
    p = tmp_path / "JOURNAL.md"
    p.write_text(SAMPLE_JOURNAL)
    entries = parse_journal(p)
    assert len(entries) == 1
    e = entries[0]
    assert e.n == 1
    assert e.verdict == "KEEP"
    assert e.git_sha == "abc1234"
    assert e.mlflow_run == "run-xyz-001"
    assert "BatchNorm" in e.hypothesis


def test_parse_results_log_extracts_metrics(tmp_path):
    p = tmp_path / "results-log.tsv"
    p.write_text(SAMPLE_TSV)
    rows = parse_results_log(p)
    assert rows[1]["acc_after"] == 0.6510


def test_build_html_writes_file_with_required_markers(tmp_path):
    journal = tmp_path / "JOURNAL.md"
    journal.write_text(SAMPLE_JOURNAL)
    tsv = tmp_path / "results-log.tsv"
    tsv.write_text(SAMPLE_TSV)
    out = tmp_path / "companion.html"

    build_html(journal_path=journal, results_log_path=tsv, output_path=out, iter_total=10)

    html = out.read_text()
    assert "chalkboard.css" in html
    assert "Nanum+Pen+Script" in html
    assert "iter 1" in html
    assert "BatchNorm" in html
    assert "65.10%" in html
