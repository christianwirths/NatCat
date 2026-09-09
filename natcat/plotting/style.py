"""House visual style for natcat figures.

Provides a colour-blind-safe, brand-neutral palette, Saffir-Simpson category
colours, matplotlib rcParams management (``apply_style`` / ``style_context``),
and small helpers used across ``natcat.plotting`` (currency-formatted axes,
source-note captions, figure saving).

All other ``natcat.plotting`` modules apply this style internally via
``style_context()`` so callers never need to invoke it themselves, but it is
also exposed for use in ad-hoc matplotlib code.

Examples
--------
>>> from natcat.plotting.style import style_context
>>> import matplotlib.pyplot as plt
>>> with style_context():
...     fig, ax = plt.subplots()
...     _ = ax.plot([0, 1], [0, 1])
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import matplotlib as mpl
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

__all__ = [
    "FIGSIZE_MAP",
    "FIGSIZE_SINGLE",
    "PALETTE",
    "SAFFIR_SIMPSON",
    "SAFFIR_SIMPSON_COLORS",
    "SAFFIR_SIMPSON_ORDER",
    "add_source_note",
    "apply_style",
    "category_colormap",
    "damage_colormap",
    "format_currency_axis",
    "save",
    "style_context",
]

#: Colour-blind-safe, brand-neutral palette shared by every natcat figure.
PALETTE: Final[dict[str, str]] = {
    "ink": "#1F2933",
    "muted": "#6B7280",
    "accent": "#2563EB",
    "accent_orange": "#F59E0B",
    "red": "#DC2626",
    "green": "#059669",
    "purple": "#7C3AED",
    "land": "#F3F4F6",
    "ocean": "#E8F1F8",
    "coast": "#9CA3AF",
}

#: Category label order, weakest to strongest.
SAFFIR_SIMPSON_ORDER: Final[tuple[str, ...]] = ("TD", "TS", "C1", "C2", "C3", "C4", "C5")

#: Saffir-Simpson category -> (lower bound, upper bound) in knots, lower inclusive.
SAFFIR_SIMPSON: Final[dict[str, tuple[float, float]]] = {
    "TD": (0.0, 34.0),
    "TS": (34.0, 64.0),
    "C1": (64.0, 83.0),
    "C2": (83.0, 96.0),
    "C3": (96.0, 113.0),
    "C4": (113.0, 137.0),
    "C5": (137.0, float("inf")),
}

#: Category -> colour (NHC-like), used to colour track segments and footprints by intensity.
SAFFIR_SIMPSON_COLORS: Final[dict[str, str]] = {
    "TD": "#5EBAFF",
    "TS": "#00FAF4",
    "C1": "#FFF3A0",
    "C2": "#FFE775",
    "C3": "#FFC140",
    "C4": "#FF8F20",
    "C5": "#FF6060",
}

#: Default figure size for single-axes figures (curves, histograms, EP curves).
FIGSIZE_SINGLE: Final[tuple[float, float]] = (7.5, 4.5)
#: Default figure size for map figures.
FIGSIZE_MAP: Final[tuple[float, float]] = (11.0, 6.0)

_FONT_STACK: Final[list[str]] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
_C5_PLOT_CEILING: Final[float] = 200.0


def category_colormap() -> tuple[ListedColormap, BoundaryNorm]:
    """Build a discrete colormap and norm for the Saffir-Simpson scale.

    Returns
    -------
    tuple[matplotlib.colors.ListedColormap, matplotlib.colors.BoundaryNorm]
        A colormap with one colour per category (TD..C5) and a norm whose
        boundaries are the category wind-speed thresholds in knots (the
        open-ended C5 bin is capped at 200 kt for normalisation purposes).

    Examples
    --------
    >>> cmap, norm = category_colormap()
    >>> cmap.N
    7
    """
    colors = [SAFFIR_SIMPSON_COLORS[label] for label in SAFFIR_SIMPSON_ORDER]
    boundaries = [SAFFIR_SIMPSON[label][0] for label in SAFFIR_SIMPSON_ORDER] + [_C5_PLOT_CEILING]
    cmap = ListedColormap(colors, name="saffir_simpson")
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def damage_colormap() -> ListedColormap:
    """Sequential colormap for damage ratios: pale rose at zero, deep red at one.

    Matplotlib's ``Reds`` with the near-white start trimmed off, so a location
    with a small but non-zero damage ratio (the tropical-storm-force fringe of
    a footprint) is still visible against the grey exposure layer while
    remaining clearly lighter than the damage core.

    Returns
    -------
    matplotlib.colors.ListedColormap

    Examples
    --------
    >>> damage_colormap().N
    256
    """
    base = mpl.colormaps["Reds"]
    return ListedColormap(base(np.linspace(0.12, 1.0, 256)), name="natcat_damage")


def _build_rc() -> dict[str, object]:
    """Return the rcParams dict defining the house style."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": list(_FONT_STACK),
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": PALETTE["coast"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.facecolor": "white",
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "grid.color": PALETTE["muted"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": PALETTE["ink"],
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.figsize": FIGSIZE_SINGLE,
        "figure.facecolor": "white",
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.15,
    }


def apply_style() -> None:
    """Apply the natcat house style to the global matplotlib rcParams.

    This mutates ``matplotlib.rcParams`` for the remainder of the process
    (or until reset). Prefer ``style_context()`` inside library code so the
    change does not leak into unrelated plots.

    Examples
    --------
    >>> apply_style()
    """
    mpl.rcParams.update(_build_rc())


@contextmanager
def style_context() -> Iterator[None]:
    """Context manager applying the house style only for its duration.

    Yields
    ------
    None

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> with style_context():
    ...     fig, ax = plt.subplots()
    """
    with mpl.rc_context(rc=_build_rc()):
        yield


def _format_currency(value: float, _pos: int | None = None) -> str:
    """Format a raw dollar amount as a compact tick label, e.g. 1e9 -> '$1B'."""
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1e12:
        text = f"{magnitude / 1e12:g}T"
    elif magnitude >= 1e9:
        text = f"{magnitude / 1e9:g}B"
    elif magnitude >= 1e6:
        text = f"{magnitude / 1e6:g}M"
    elif magnitude >= 1e3:
        text = f"{magnitude / 1e3:g}K"
    else:
        text = f"{magnitude:g}"
    return f"{sign}${text}"


def format_currency_axis(ax: mpl.axes.Axes, axis: str = "x") -> None:
    """Format an axis's tick labels as compact currency (e.g. $1B, $100M).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose ticks should be reformatted.
    axis : str, default "x"
        Which axis to format, ``"x"`` or ``"y"``.

    Raises
    ------
    ValueError
        If ``axis`` is not ``"x"`` or ``"y"``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> _ = ax.plot([1e8, 1e9, 1e10], [0, 1, 2])
    >>> format_currency_axis(ax, axis="x")
    """
    formatter = FuncFormatter(_format_currency)
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    elif axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def add_source_note(fig: Figure, text: str, *, gap_in: float = 0.12) -> None:
    """Add a small grey source-attribution note directly below the figure content.

    The note is placed just under the lowest drawn artist (axes, colorbar or
    legend), so it stays attached to the figure regardless of the map aspect
    ratio and of ``bbox_inches="tight"`` cropping.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to annotate.
    text : str
        Note text, e.g. ``"Source: NHC ATCF best track; LitPop exposure"``.
    gap_in : float, default 0.12
        Vertical gap between the content and the note, in inches.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> add_source_note(fig, "Source: NHC ATCF best track")
    """
    with style_context():
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = fig.get_tightbbox(renderer)
        width_in, height_in = fig.get_size_inches()
        x = max(bbox.x0 / width_in, 0.0)
        y = bbox.y0 / height_in - gap_in / height_in
        fig.text(
            x,
            y,
            text,
            fontsize=7.5,
            color=PALETTE["muted"],
            ha="left",
            va="top",
        )


def save(fig: Figure, path: str | Path, formats: tuple[str, ...] = ("png",)) -> Path | list[Path]:
    """Save a figure to one or more formats, creating parent directories.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or pathlib.Path
        Destination path. Its suffix is replaced by each entry in ``formats``.
    formats : tuple[str, ...], default ("png",)
        File extensions (without a leading dot) to save.

    Returns
    -------
    pathlib.Path or list[pathlib.Path]
        The saved path if a single format was requested, otherwise the list
        of saved paths.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> out = save(fig, "/tmp/example.png")  # doctest: +SKIP
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    with style_context():
        for fmt in formats:
            out_path = destination.with_suffix(f".{fmt.lstrip('.')}")
            fig.savefig(out_path)
            saved.append(out_path)
    return saved[0] if len(saved) == 1 else saved
