"""Each STAGES/*.md must have the 4 canonical sections."""

from pathlib import Path

import pytest


STAGES_DIR = Path(__file__).resolve().parents[1] / "STAGES"

REQUIRED_SECTIONS = ["## 목표", "## 사전 점검", "## 소크라테스 질문", "## 통과 조건"]

EXPECTED_FILES = [
    "00-setup.md",
    "01-skill-tour.md",
    "02-dataset.md",
    "03-mlflow.md",
    "04-baseline.md",
    "05-journal.md",
    "06-loop.md",
    "07-debrief.md",
]


@pytest.mark.parametrize("name", EXPECTED_FILES)
def test_stage_file_exists(name):
    assert (STAGES_DIR / name).exists(), f"missing STAGES/{name}"


@pytest.mark.parametrize("name", EXPECTED_FILES)
def test_stage_has_required_sections(name):
    text = (STAGES_DIR / name).read_text()
    for sec in REQUIRED_SECTIONS:
        assert sec in text, f"STAGES/{name} missing section {sec!r}"
