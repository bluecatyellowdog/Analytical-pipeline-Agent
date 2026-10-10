"""REPORT.html: the same content as REPORT.pdf, for a browser, with more to unfold.

The look lives in templates/report.html, which can be edited without touching
Python. The charts are embedded, so the file can be sent on its own.
"""

from __future__ import annotations

import base64
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from pipeline.reports.summary import MainSummary

TEMPLATES = Path(__file__).parent / "templates"
TEMPLATE = "report.html"


def embedded(path: Path) -> str:
    """A chart as a data URI, so the page needs no other file."""
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def render_html(summary: MainSummary) -> str:
    environment = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=select_autoescape(["html"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return environment.get_template(TEMPLATE).render(s=summary, img=embedded)


def write_html(path: Path, summary: MainSummary) -> Path:
    path.write_text(render_html(summary), encoding="utf-8")
    return path
