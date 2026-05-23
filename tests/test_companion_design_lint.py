"""Mechanical guards for Chalkboard rules (spec §7.4)."""

from pathlib import Path
import re


CSS = (Path(__file__).resolve().parents[1] / "viz/_static/chalkboard.css").read_text()
TEMPLATE = (Path(__file__).resolve().parents[1] / "viz/_templates/companion.html.j2").read_text()


def test_no_gradient_in_css():
    assert "linear-gradient" not in CSS
    assert "radial-gradient" not in CSS


def test_no_backdrop_filter():
    assert "backdrop-filter" not in CSS


def test_no_rounded_square_corners():
    """Forbid rounded-square chrome (9–99px). Pills/circles (>=100px) are allowed."""
    for m in re.finditer(r"border-radius:\s*([0-9]+)px", CSS):
        v = int(m.group(1))
        # OK: small rounded (<=8) or pill/circle (>=100). Forbidden: rounded-square (9..99).
        assert v <= 8 or v >= 100, (
            f"border-radius {m.group(0)} violates Chalkboard rule "
            "(allow <=8px or pill >=100px; rounded-square 9..99px forbidden)"
        )


def test_chalkboard_bg_color_set():
    assert "#2f4f4f" in CSS.lower()


def test_accent_colors_are_tomato_and_yellow():
    assert "#ff6347" in CSS.lower()
    assert "#ffd700" in CSS.lower()


def test_template_references_chalkboard_css():
    assert "chalkboard.css" in TEMPLATE


def test_template_loads_required_fonts():
    for fam in ("Nanum+Pen+Script", "Gowun+Dodum", "JetBrains+Mono"):
        assert fam in TEMPLATE, f"missing font: {fam}"
