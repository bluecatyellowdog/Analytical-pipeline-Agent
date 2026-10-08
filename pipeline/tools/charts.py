"""Charts for analysis artifacts, drawn by code from aggregated numbers only.

One plain style for every chart: the numbers come from the tool's result, never
from an LLM, and nothing here receives rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - the backend must be chosen first

BAR = "#4c72b0"
REFERENCE = "#c44e52"
LOG_TICKS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)


def _figure(title: str, height: float = 3.4):
    figure, axes = plt.subplots(figsize=(6.4, height), dpi=150)
    axes.set_title(title, fontsize=10, loc="left")
    axes.spines[["top", "right"]].set_visible(False)
    axes.tick_params(labelsize=8)
    return figure, axes


def bars(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    ylabel: str,
    lower: Sequence[float] | None = None,
    upper: Sequence[float] | None = None,
    reference: float | None = None,
    percent: bool = False,
):
    figure, axes = _figure(title)
    positions = range(len(values))
    errors = None
    if lower is not None and upper is not None:
        errors = [
            [max(v - lo, 0) for v, lo in zip(values, lower, strict=True)],
            [max(hi - v, 0) for v, hi in zip(values, upper, strict=True)],
        ]
    axes.bar(positions, values, color=BAR, yerr=errors, capsize=3, ecolor="#444444")
    axes.set_xticks(list(positions), [str(label) for label in labels], rotation=30, ha="right")
    axes.set_ylabel(ylabel, fontsize=8)
    if reference is not None:
        axes.axhline(reference, color=REFERENCE, linewidth=1, linestyle="--", label="overall")
        axes.legend(fontsize=7, frameon=False)
    if percent:
        axes.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    figure.tight_layout()
    return figure


def grouped_bars(
    groups: Sequence[str],
    series: Sequence[str],
    values: dict[tuple[str, str], float],
    *,
    title: str,
    ylabel: str,
    legend: str,
    reference: float | None = None,
    percent: bool = False,
):
    """One cluster of bars per group, one bar per series; a missing combination is left out."""
    figure, axes = _figure(title)
    width = 0.8 / max(len(series), 1)
    for j, name in enumerate(series):
        offset = (j - (len(series) - 1) / 2) * width
        heights = [values.get((group, name), float("nan")) for group in groups]
        axes.bar([i + offset for i in range(len(groups))], heights, width=width, label=str(name))
    axes.set_xticks(range(len(groups)), [str(g) for g in groups], rotation=30, ha="right")
    axes.set_ylabel(ylabel, fontsize=8)
    if reference is not None:
        axes.axhline(reference, color=REFERENCE, linewidth=1, linestyle="--")
    axes.legend(fontsize=7, frameon=False, title=legend, title_fontsize=7)
    if percent:
        axes.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    figure.tight_layout()
    return figure


def histogram(edges: Sequence[float], counts: Sequence[int], *, title: str, xlabel: str):
    figure, axes = _figure(title)
    widths = [b - a for a, b in zip(edges[:-1], edges[1:], strict=True)]
    axes.bar(edges[:-1], counts, width=widths, align="edge", color=BAR, edgecolor="white")
    axes.set_xlabel(xlabel, fontsize=8)
    axes.set_ylabel("rows", fontsize=8)
    figure.tight_layout()
    return figure


def boxes(stats: list[dict], *, title: str, ylabel: str):
    """Box plots drawn from quantiles, so no individual value is ever plotted."""
    figure, axes = _figure(title)
    axes.bxp(
        stats,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#dbe3f0", "edgecolor": BAR},
    )
    axes.set_ylabel(ylabel, fontsize=8)
    figure.tight_layout()
    return figure


def forest(
    labels: Sequence[str],
    values: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    title: str,
    xlabel: str,
    reference: float,
    log: bool = False,
):
    """Effects as points with their 95% intervals, against the line of 'no effect'."""
    if log:
        # An interval that runs off to 0 or infinity is drawn to the edge of the chart.
        lower = [min(max(lo, 1e-3), v) for v, lo in zip(values, lower, strict=True)]
        upper = [max(min(hi, 1e3), v) for v, hi in zip(values, upper, strict=True)]
    height = max(2.2, 0.32 * len(values) + 1.0)
    figure, axes = _figure(title, height=height)
    positions = list(range(len(values)))
    errors = [
        [max(v - lo, 0) for v, lo in zip(values, lower, strict=True)],
        [max(hi - v, 0) for v, hi in zip(values, upper, strict=True)],
    ]
    axes.errorbar(
        values, positions, xerr=errors, fmt="o", color=BAR, ecolor="#444444", capsize=3, ms=4
    )
    axes.set_yticks(positions, [str(label) for label in labels])
    axes.invert_yaxis()
    axes.axvline(reference, color=REFERENCE, linewidth=0.8, linestyle="--")
    if log:
        axes.set_xscale("log")
        left = min(max(min(lower), 0.01), reference) / 1.2
        right = max(min(max(upper), 100), reference) * 1.2
        axes.set_xlim(left, right)
        ticks = [t for t in LOG_TICKS if left <= t <= right]
        axes.set_xticks(ticks, [f"{t:g}" for t in ticks])
        axes.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes.set_xlabel(xlabel, fontsize=8)
    figure.tight_layout()
    return figure


def heatmap(labels: Sequence[str], matrix: Sequence[Sequence[float]], *, title: str):
    size = max(3.4, 0.45 * len(labels) + 1.6)
    figure, axes = _figure(title, height=size)
    image = axes.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    axes.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axes.set_yticks(range(len(labels)), labels)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            axes.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6)
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path
