"""Calibration figures: modelled-versus-observed scatter and calibrated curves."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from natcat.plotting.style import PALETTE, format_currency_axis, style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from natcat.calibration.calibrator import CalibrationResult
    from natcat.calibration.hazard import HazardGridResult
    from natcat.vulnerability.value import ValueDependentVulnerability

__all__ = ["plot_calibration", "plot_hazard_grid", "plot_value_dependent_curves"]


def plot_calibration(
    result: CalibrationResult,
    *,
    ax: Axes | None = None,
    show_initial: bool = True,
    label_storms: bool = True,
    band_factor: float = 2.0,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Modelled versus observed loss per storm, before and after calibration.

    Parameters
    ----------
    result : CalibrationResult
        Output of :meth:`natcat.calibration.Calibrator.fit`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created when omitted.
    show_initial : bool, default True
        Also draw the starting model's losses in grey.
    label_storms : bool, default True
        Annotate calibrated points with the storm name.
    band_factor : float, default 2.0
        Shade the band within this factor of the 1:1 line.
    title : str, optional
        Left-aligned bold title.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    with style_context():
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(7.0, 6.0))
        else:
            fig = ax.figure

        table = result.table
        observed = table["observed"].to_numpy()
        lo = 10 ** np.floor(np.log10(min(observed.min(), table["modelled"].min()) / 3.0))
        hi = 10 ** np.ceil(np.log10(max(observed.max(), table["modelled"].max()) * 3.0))
        line = np.array([lo, hi])

        ax.fill_between(
            line, line / band_factor, line * band_factor, color=PALETTE["accent"], alpha=0.07, lw=0
        )
        ax.plot(line, line, color=PALETTE["ink"], lw=1.0, ls="--", label="1:1")
        if show_initial:
            ax.scatter(
                observed, table["modelled_initial"], s=34, color=PALETTE["muted"], alpha=0.6,
                linewidth=0, label="Before calibration", zorder=3,
            )  # fmt: skip
        ax.scatter(
            observed, table["modelled"], s=42, color=PALETTE["accent"], edgecolor="white",
            linewidth=0.6, label="After calibration", zorder=4,
        )  # fmt: skip
        if label_storms:
            for _, row in table.iterrows():
                ax.annotate(
                    row["name"], (row["observed"], row["modelled"]), xytext=(4, 3),
                    textcoords="offset points", fontsize=7, color=PALETTE["ink"],
                )  # fmt: skip

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        format_currency_axis(ax, axis="x")
        format_currency_axis(ax, axis="y")
        ax.set_xlabel("Observed loss (reference-year USD)")
        ax.set_ylabel("Modelled ground-up loss (USD)")
        ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9, edgecolor="none")
        ax.text(
            0.98, 0.03,
            f"typical factor error {10 ** np.sqrt(result.initial_objective):.2f}x "
            f"→ {10 ** result.rmse_log10:.2f}x",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=PALETTE["muted"],
        )  # fmt: skip
        if title:
            ax.set_title(title, loc="left", fontweight="bold")
    return fig, ax


def plot_value_dependent_curves(
    model: ValueDependentVulnerability,
    *,
    tivs: Sequence[float] = (1e5, 1e6, 1e7, 1e8, 1e9),
    reference: object | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Damage-ratio curves of a value-dependent model for several tile values.

    Parameters
    ----------
    model : ValueDependentVulnerability
        Model to draw.
    tivs : sequence of float, default (1e5, ..., 1e9)
        Tile values (USD) to draw one curve for each.
    reference : VulnerabilityModel, optional
        A model to draw as a dashed reference (e.g. the uncalibrated
        ``WindVulnerability("Frame")``).
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    with style_context():
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(7.0, 4.2))
        else:
            fig = ax.figure

        winds = np.linspace(20.0, 180.0, 321)
        cmap = plt.get_cmap("viridis")
        for i, tiv in enumerate(tivs):
            colour = cmap(0.15 + 0.7 * i / max(len(tivs) - 1, 1))
            ax.plot(
                winds, model.damage_ratio(winds, tiv=np.full(winds.shape, tiv)),
                color=colour, lw=1.8, label=f"tile value ${tiv:,.0f}",
            )  # fmt: skip
        if reference is not None:
            ax.plot(
                winds, reference.damage_ratio(winds), color=PALETTE["ink"], lw=1.2, ls="--",
                label="reference", zorder=2,
            )  # fmt: skip
        ax.axvline(model.threshold_kt, color=PALETTE["muted"], lw=0.8, ls=":")
        ax.set_xlim(20, 180)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Maximum sustained wind (kt)")
        ax.set_ylabel("Mean damage ratio")
        ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor="none", loc="lower right")
        if title:
            ax.set_title(title, loc="left", fontweight="bold")
    return fig, ax


def plot_hazard_grid(
    result: HazardGridResult,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Heat map of the calibrated factor error over the hazard-parameter grid.

    Parameters
    ----------
    result : HazardGridResult
        Output of :func:`natcat.calibration.calibrate_hazard_grid`.
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    with style_context():
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
        else:
            fig = ax.figure

        table = result.table
        exponents = sorted(table["decay_exponent"].unique())
        asymmetries = sorted(table["asymmetry_factor"].unique())
        grid = table.pivot(
            index="asymmetry_factor", columns="decay_exponent", values="factor_error"
        ).reindex(index=asymmetries, columns=exponents)
        image = ax.imshow(
            grid.to_numpy(), origin="lower", cmap="viridis_r", aspect="auto",
            interpolation="nearest",
        )  # fmt: skip
        ax.set_xticks(range(len(exponents)), [f"{e:g}" for e in exponents])
        ax.set_yticks(range(len(asymmetries)), [f"{a:g}" for a in asymmetries])
        ax.set_xlabel("Rankine decay exponent")
        ax.set_ylabel("Motion asymmetry factor")
        values = grid.to_numpy()
        vmid = np.nanmean(values)
        for i in range(len(asymmetries)):
            for j in range(len(exponents)):
                value = values[i, j]
                if np.isfinite(value):
                    ax.text(
                        j, i, f"{value:.2f}x", ha="center", va="center", fontsize=8,
                        color="white" if value > vmid else PALETTE["ink"],
                    )  # fmt: skip
        best_e, best_a = (
            result.best_hazard["decay_exponent"],
            result.best_hazard["asymmetry_factor"],
        )
        ax.scatter(
            [exponents.index(best_e)], [asymmetries.index(best_a)], s=260, facecolor="none",
            edgecolor=PALETTE["accent_orange"], linewidth=2.0, zorder=5,
        )  # fmt: skip
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Typical factor error after calibration")
        ax.grid(False)
        if title:
            ax.set_title(title, loc="left", fontweight="bold")
    return fig, ax
