"""Non-map figures: wind profiles, vulnerability, RMW heuristic, loss curves.

These functions only depend on numpy/pandas/matplotlib/scipy (no cartopy),
so they are safe to import and call eagerly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from natcat.plotting.style import (
    FIGSIZE_SINGLE,
    PALETTE,
    SAFFIR_SIMPSON,
    SAFFIR_SIMPSON_COLORS,
    SAFFIR_SIMPSON_ORDER,
    format_currency_axis,
    style_context,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "plot_annual_loss_distribution",
    "plot_ep_curve",
    "plot_land_decay",
    "plot_rmw_heuristic",
    "plot_vulnerability",
    "plot_wind_profile",
]

_KT_TO_MS = 0.514444
_C5_PLOT_CEILING = 200.0

#: (upper bound of vmax class in kt, RMW in nm) per the RMW heuristic in the spec.
_RMW_HEURISTIC_STEPS: tuple[tuple[float, float], ...] = (
    (35.0, 80.0),
    (64.0, 60.0),
    (96.0, 40.0),
    (137.0, 25.0),
    (_C5_PLOT_CEILING, 15.0),
)


def _safe_reciprocal(x: np.ndarray) -> np.ndarray:
    """Reciprocal that maps 0 to inf without raising a divide-by-zero warning.

    Used for the probability <-> return-period secondary axis transform,
    which matplotlib probes at axis limits that may include 0.
    """
    with np.errstate(divide="ignore"):
        return np.divide(1.0, np.asarray(x, dtype=float))


def _rankine_profile(radius_nm: np.ndarray, vmax: float, rmw: float, exponent: float) -> np.ndarray:
    """Evaluate the (modified) Rankine vortex wind profile v(r)."""
    r = np.asarray(radius_nm, dtype=float)
    safe_r = np.where(r <= 0, np.nan, r)
    v = np.where(r < rmw, vmax * r / rmw, vmax * (rmw / safe_r) ** exponent)
    return np.where(r == 0, 0.0, v)


def _draw_category_bands(ax: Axes, x_max: float, label_y: float = 0.975) -> None:
    """Shade Saffir-Simpson wind-speed bands lightly in the background."""
    for category in SAFFIR_SIMPSON_ORDER:
        lo, hi = SAFFIR_SIMPSON[category]
        hi_plot = min(hi, x_max)
        if lo >= x_max:
            continue
        ax.axvspan(
            lo, hi_plot, color=SAFFIR_SIMPSON_COLORS[category], alpha=0.15, linewidth=0, zorder=0
        )
        ax.annotate(
            category,
            xy=((lo + hi_plot) / 2, label_y),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=7,
            color=PALETTE["muted"],
            annotation_clip=False,
        )


def _format_short_currency(value: float) -> str:
    """Compact currency label, e.g. 1.23e9 -> '$1.2B', 4.5e7 -> '$45M'."""
    magnitude = abs(float(value))
    if magnitude >= 1e12:
        return f"${magnitude / 1e12:.2f}T"
    if magnitude >= 1e9:
        return f"${magnitude / 1e9:.1f}B"
    if magnitude >= 1e6:
        return f"${magnitude / 1e6:.0f}M"
    if magnitude >= 1e3:
        return f"${magnitude / 1e3:.0f}K"
    return f"${magnitude:.0f}"


def plot_wind_profile(
    vmax: float = 120.0,
    rmw: float = 20.0,
    exponents: tuple[float, ...] = (0.5, 1.0, 2.0),
    *,
    translation_speed: float = 15.0,
    asymmetry_factor: float = 0.5,
    ax: Axes | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot the Rankine wind profile and the motion-asymmetry effect.

    Left panel shows ``v(r)`` for ``r`` in 0-300 nm for each decay exponent.
    Right panel shows how translation speed skews the wind at the radius of
    maximum wind between the right and left side of the track (asymmetry).

    Parameters
    ----------
    vmax : float, default 120.0
        Maximum sustained wind speed (kt).
    rmw : float, default 20.0
        Radius of maximum wind (nm).
    exponents : tuple[float, ...], default (0.5, 1.0, 2.0)
        Rankine decay exponents to compare (0.5 = modified Rankine,
        1.0 = classical Rankine, 2.0 = natcat's current default).
    translation_speed : float, default 15.0
        Storm translation speed (kt) used in the asymmetry panel.
    asymmetry_factor : float, default 0.5
        Asymmetry scaling factor (see ``natcat.hazards.wind_field.motion_asymmetry``).
    ax : array-like of matplotlib.axes.Axes, optional
        Two existing axes ``[left, right]`` to draw on. A new 1x2 figure is
        created when omitted.

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray]
        The figure and an array of the two axes.

    Examples
    --------
    >>> fig, axes = plot_wind_profile()
    """
    with style_context():
        import matplotlib.pyplot as plt

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
        else:
            axes = np.atleast_1d(ax)
            fig = axes[0].figure

        radius = np.linspace(0, 300, 400)
        line_colors = [
            PALETTE["accent"],
            PALETTE["accent_orange"],
            PALETTE["red"],
            PALETTE["purple"],
        ]
        for exponent, color in zip(exponents, line_colors, strict=False):
            wind = _rankine_profile(radius, vmax, rmw, exponent)
            axes[0].plot(radius, wind, color=color, linewidth=1.8, label=f"exponent = {exponent:g}")
        axes[0].axvline(rmw, color=PALETTE["muted"], linestyle="--", linewidth=0.8)
        axes[0].annotate(
            "RMW",
            xy=(rmw, 0.03),
            xycoords=("data", "axes fraction"),
            fontsize=8,
            color=PALETTE["muted"],
        )
        axes[0].set_xlim(0, 300)
        axes[0].set_ylim(bottom=0)
        axes[0].set_xlabel("Radius (nm)")
        axes[0].set_ylabel("Wind speed (kt)")
        axes[0].set_title("Rankine vortex profile", loc="left", fontweight="bold")
        axes[0].legend(loc="upper right")

        bearing = np.linspace(-180, 180, 361)
        delta = ((bearing + 180) % 360) - 180
        asymmetry = asymmetry_factor * translation_speed * np.sin(np.deg2rad(delta))
        symmetric_wind = float(_rankine_profile(np.array([rmw]), vmax, rmw, exponent=1.0)[0])
        total_wind = symmetric_wind + asymmetry
        axes[1].plot(bearing, total_wind, color=PALETTE["purple"], linewidth=1.8)
        axes[1].axhline(
            symmetric_wind,
            color=PALETTE["muted"],
            linestyle="--",
            linewidth=0.8,
            label="Symmetric wind",
        )
        axes[1].axvline(90, color=PALETTE["coast"], linewidth=0.6, alpha=0.7)
        axes[1].axvline(-90, color=PALETTE["coast"], linewidth=0.6, alpha=0.7)
        axes[1].annotate(
            "right of track",
            xy=(90, 0.04),
            xycoords=("data", "axes fraction"),
            ha="center",
            fontsize=7.5,
            color=PALETTE["muted"],
        )
        axes[1].annotate(
            "left of track",
            xy=(-90, 0.04),
            xycoords=("data", "axes fraction"),
            ha="center",
            fontsize=7.5,
            color=PALETTE["muted"],
        )
        axes[1].set_xlim(-180, 180)
        axes[1].set_xlabel("Bearing relative to heading (deg)")
        axes[1].set_ylabel("Wind speed at RMW (kt)")
        axes[1].set_title(
            f"Motion asymmetry (translation = {translation_speed:g} kt)",
            loc="left",
            fontweight="bold",
        )
        axes[1].legend(loc="upper right")
    return fig, axes


def plot_vulnerability(
    vulnerability=None,
    *,
    ax: Axes | None = None,
    construction_types: list[str] | None = None,
    wind_range: tuple[float, float] = (0.0, 180.0),
) -> tuple[Figure, Axes]:
    """Plot damage ratio vs. wind speed per construction type.

    Parameters
    ----------
    vulnerability : object, optional
        Object exposing ``damage_ratio(intensity, construction_types=None)``
        and (ideally) a ``CONSTRUCTION_PARAMS`` mapping of construction type
        name to parameters, e.g. ``natcat.vulnerability.WindVulnerability()``.
        Defaults to a fresh ``WindVulnerability`` instance when omitted.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    construction_types : list[str], optional
        Construction types to plot; defaults to the keys of
        ``vulnerability.CONSTRUCTION_PARAMS``.
    wind_range : tuple[float, float], default (0.0, 180.0)
        Wind speed range (kt) to evaluate.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    Examples
    --------
    >>> fig, ax = plot_vulnerability()  # doctest: +SKIP
    """
    with style_context():
        import matplotlib.pyplot as plt

        if vulnerability is None:
            from natcat.vulnerability import WindVulnerability

            vulnerability = WindVulnerability()

        params = getattr(vulnerability, "CONSTRUCTION_PARAMS", {})
        types = construction_types or list(params.keys()) or ["default"]

        if ax is None:
            fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        else:
            fig = ax.figure

        _draw_category_bands(ax, wind_range[1])

        wind = np.linspace(wind_range[0], wind_range[1], 250)
        line_colors = [PALETTE["accent"], PALETTE["red"], PALETTE["green"], PALETTE["purple"]]
        for i, construction in enumerate(types):
            damage_ratio = np.asarray(
                vulnerability.damage_ratio(wind, construction_types=[construction] * len(wind))
            )
            ax.plot(
                wind,
                damage_ratio,
                color=line_colors[i % len(line_colors)],
                linewidth=1.8,
                label=construction,
            )

        ax.set_xlim(wind_range)
        ax.set_ylim(0, 1.09)
        ax.set_xlabel("Wind speed (kt)")
        ax.set_ylabel("Damage ratio")
        ax.set_title(
            "Vulnerability curves by construction type", loc="left", fontweight="bold", pad=40
        )
        ax.legend(loc="lower right")

        secax = ax.secondary_xaxis(
            "top", functions=(lambda kt: kt * _KT_TO_MS, lambda ms: ms / _KT_TO_MS)
        )
        secax.set_xlabel("Wind speed (m/s)", fontsize=8.5)
        secax.tick_params(labelsize=7.5)
    return fig, ax


def plot_rmw_heuristic(ax: Axes | None = None) -> tuple[Figure, Axes]:
    """Plot the RMW relations used for fixes without an observed radius.

    The Willoughby et al. (2006) relation (default) is drawn for three
    latitudes against the legacy intensity-class step table.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    Examples
    --------
    >>> fig, ax = plot_rmw_heuristic()
    """
    from natcat.tracks.processing import willoughby_rmw

    with style_context():
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        else:
            fig = ax.figure

        _draw_category_bands(ax, _C5_PLOT_CEILING)

        winds = np.linspace(20.0, _C5_PLOT_CEILING, 200)
        for latitude, colour in ((15.0, PALETTE["accent"]), (25.0, "#1E40AF"), (35.0, "#0F172A")):
            ax.plot(
                winds, willoughby_rmw(winds, latitude), color=colour, linewidth=2.0,
                label=f"Willoughby et al. (2006), {latitude:.0f}°N",
            )  # fmt: skip

        edges = [0.0, *[edge for edge, _ in _RMW_HEURISTIC_STEPS]]
        values = [rmw for _, rmw in _RMW_HEURISTIC_STEPS]
        ax.step(
            edges[:-1] + [edges[-1]], values + [values[-1]], where="post",
            color=PALETTE["muted"], linewidth=1.4, linestyle="--", label="legacy step table",
        )  # fmt: skip

        ax.set_xlim(0, _C5_PLOT_CEILING)
        ax.set_ylim(0, 90)
        ax.set_xlabel("Maximum sustained wind speed (kt)")
        ax.set_ylabel("Radius of maximum wind (nm)")
        ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor="none", loc="upper right")
        ax.set_title("RMW relations for missing values", loc="left", fontweight="bold", pad=16)
    return fig, ax


def plot_annual_loss_distribution(
    annual_losses,
    *,
    ax: Axes | None = None,
    bins: int = 50,
    log_y: bool = True,
    show_aal: bool = True,
) -> tuple[Figure, Axes]:
    """Histogram of simulated annual aggregate losses.

    Parameters
    ----------
    annual_losses : array-like
        One aggregate loss per simulated year.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    bins : int, default 50
        Histogram bin count.
    log_y : bool, default True
        Use a log-scaled y axis (recommended: annual loss distributions are
        heavily right-skewed).
    show_aal : bool, default True
        Draw a vertical line at the mean (AAL) with a currency-formatted
        annotation.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    Examples
    --------
    >>> fig, ax = plot_annual_loss_distribution([1e8, 5e8, 2e9])
    """
    with style_context():
        import matplotlib.pyplot as plt

        losses = np.asarray(annual_losses, dtype=float)
        if ax is None:
            fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        else:
            fig = ax.figure

        # Annual losses are heavy-tailed: log-spaced bins on a log axis keep the body
        # and the tail legible at the same time. Zero-loss years are reported in text.
        positive = losses[losses > 0]
        n_zero = int(losses.size - positive.size)
        if positive.size:
            lo = max(float(positive.min()), 1e3)
            hi = float(positive.max())
            edges = np.logspace(np.log10(lo), np.log10(hi * 1.0001), bins + 1)
            ax.hist(
                positive,
                bins=edges,
                color=PALETTE["accent"],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.3,
            )
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        if n_zero:
            ax.annotate(
                f"{n_zero:,} of {losses.size:,} years with no loss",
                xy=(0.02, 0.97),
                xycoords="axes fraction",
                fontsize=8,
                color=PALETTE["muted"],
                ha="left",
                va="top",
            )

        if show_aal and losses.size:
            aal = float(losses.mean())
            ax.axvline(aal, color=PALETTE["red"], linewidth=1.6, linestyle="--", zorder=4)
            ax.annotate(
                f"AAL = ${aal / 1e9:.2f}B",
                xy=(aal, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(6, -4),
                textcoords="offset points",
                fontsize=8.5,
                color=PALETTE["red"],
                fontweight="bold",
                va="top",
            )

        format_currency_axis(ax, axis="x")
        ax.set_xlabel("Annual aggregate loss")
        ax.set_ylabel("Number of years" + (" (log scale)" if log_y else ""))
        ax.set_title("Annual loss distribution", loc="left", fontweight="bold")
    return fig, ax


def plot_ep_curve(
    ep,
    *,
    kinds: tuple[str, ...] = ("aep", "oep"),
    ax: Axes | None = None,
    show_empirical: bool = True,
    show_tail_threshold: bool = True,
    return_period_axis: bool = True,
    highlight_rps: tuple[int, ...] = (10, 50, 100, 250),
) -> tuple[Figure, Axes]:
    """Plot AEP/OEP exceedance-probability curves with return-period markers.

    Parameters
    ----------
    ep : natcat.financial.ExceedanceProbability
        Fitted exceedance-probability object exposing ``curve(kind)``,
        ``aep_empirical``/``oep_empirical``, ``tail_fit(kind)`` and
        ``loss_at_return_period(rp, kind)``.
    kinds : tuple[str, ...], default ("aep", "oep")
        Which curve(s) to draw.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    show_empirical : bool, default True
        Overlay empirical exceedance points as small markers.
    show_tail_threshold : bool, default True
        Draw a vertical dotted line at each curve's GPD tail threshold.
    return_period_axis : bool, default True
        Add a secondary right-hand y axis showing return period in years.
    highlight_rps : tuple[int, ...], default (10, 50, 100, 250)
        Return periods (years) to mark and label, e.g. "1-in-100: $12.3B".

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    Examples
    --------
    >>> fig, ax = plot_ep_curve(ep)  # doctest: +SKIP
    """
    with style_context():
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8.5, 5.0))
        else:
            fig = ax.figure

        colors = {"aep": PALETTE["accent"], "oep": PALETTE["red"]}
        labels = {"aep": "AEP (annual)", "oep": "OEP (occurrence)"}

        # Only the primary (first) curve gets text labels at its highlighted return
        # periods -- labelling every kind causes AEP/OEP text to collide since both
        # sit at the same probability (1/rp) and often at nearby loss values.
        primary_kind = kinds[0] if kinds else None

        for kind in kinds:
            color = colors.get(kind, PALETTE["ink"])
            curve = ep.curve(kind=kind)
            ax.plot(
                curve["loss"],
                curve["probability"],
                color=color,
                linewidth=1.8,
                label=labels.get(kind, kind.upper()),
                zorder=4,
            )

            if show_empirical:
                empirical_fn = getattr(ep, f"{kind}_empirical", None)
                if empirical_fn is not None:
                    losses = curve["loss"].to_numpy()
                    stride = max(1, len(losses) // 40)
                    sampled = losses[::stride]
                    ax.scatter(
                        sampled,
                        np.asarray(empirical_fn(sampled)),
                        s=10,
                        color=color,
                        alpha=0.5,
                        zorder=3,
                    )

            if show_tail_threshold:
                tail_fit = getattr(ep, "tail_fit", None)
                if tail_fit is not None:
                    fit = tail_fit(kind=kind)
                    if fit is not None:
                        threshold = fit[-1]
                        ax.axvline(
                            threshold,
                            color=color,
                            linestyle=":",
                            linewidth=1.0,
                            alpha=0.6,
                            zorder=2,
                        )

            for rp in highlight_rps:
                loss_rp = float(np.atleast_1d(ep.loss_at_return_period(rp, kind=kind))[0])
                prob_rp = 1.0 / rp
                ax.scatter(
                    [loss_rp],
                    [prob_rp],
                    color=color,
                    edgecolor="white",
                    linewidth=0.6,
                    s=30,
                    zorder=6,
                )
                if kind != primary_kind:
                    continue
                ax.annotate(
                    f"1-in-{rp}: {_format_short_currency(loss_rp)}",
                    xy=(loss_rp, prob_rp),
                    xytext=(9, 7),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=color,
                    ha="left",
                    va="bottom",
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.85,
                    },
                    zorder=7,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        positive = np.concatenate(
            [np.asarray(ep.curve(kind=k)["loss"], dtype=float) for k in kinds]
        )
        positive = positive[positive > 0]
        if positive.size:
            ax.set_xlim(left=max(1e6, float(np.percentile(positive, 1))))
        format_currency_axis(ax, axis="x")
        ax.set_xlabel("Loss")
        ax.set_ylabel("Exceedance probability")
        ax.set_title("Exceedance probability curve", loc="left", fontweight="bold")
        ax.legend(loc="upper right")

        if return_period_axis:
            secax = ax.secondary_yaxis("right", functions=(_safe_reciprocal, _safe_reciprocal))
            secax.set_ylabel("Return period (years)")
    return fig, ax


def plot_land_decay(
    model,
    segments: pd.DataFrame,
    *,
    legacy_factor: float | None = 0.92,
    classes: Sequence[tuple[float, float]] = ((34.0, 63.0), (64.0, 95.0), (96.0, 200.0)),
    max_hours: float = 48.0,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Observed inland decay versus the fitted :class:`LandDecayModel`.

    For each landfall-intensity class the median observed wind per 3-hour bin
    after landfall is drawn with its inter-quartile band, together with the
    model curve started from the class's mean landfall wind and, dashed, the
    legacy exponential rule without a background wind.

    Parameters
    ----------
    model : natcat.stochastic.LandDecayModel
        Fitted decay model.
    segments : pandas.DataFrame
        Output of :func:`natcat.stochastic.extract_landfall_segments`.
    legacy_factor : float, optional
        Per-hour factor of the legacy rule to draw for comparison; ``None``
        omits it.
    classes : sequence of (lo, hi)
        Landfall-wind classes in knots.
    max_hours : float, default 48.0
        Time axis limit.
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    with style_context():
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
        else:
            fig = ax.figure

        hours = np.linspace(0.0, max_hours, 200)
        colours = [PALETTE["accent"], PALETTE["accent_orange"], "#B91C1C"]
        for (lo, hi), colour in zip(classes, colours, strict=False):
            part = segments[
                (segments["wind_at_landfall_kt"] >= lo)
                & (segments["wind_at_landfall_kt"] <= hi)
                & (segments["hours"] <= max_hours)
            ]
            if part.empty:
                continue
            bins = (part["hours"] / 3.0).round() * 3.0
            grouped = part.groupby(bins)["max_wind_speed_kt"]
            centres = grouped.median().index.to_numpy()
            ax.fill_between(
                centres, grouped.quantile(0.25).to_numpy(), grouped.quantile(0.75).to_numpy(),
                color=colour, alpha=0.12, lw=0,
            )  # fmt: skip
            n_storms = part["storm"].nunique()
            ax.plot(
                centres, grouped.median().to_numpy(), "o", color=colour, ms=3.5,
                label=f"observed, landfall {lo:.0f}-{min(hi, 150):.0f} kt (n={n_storms})",
            )  # fmt: skip
            wind_0 = float(part["wind_at_landfall_kt"].mean())
            ax.plot(hours, model.step(wind_0, hours), color=colour, lw=1.8)
            if legacy_factor is not None:
                ax.plot(
                    hours, wind_0 * legacy_factor**hours, color=colour, lw=1.0, ls="--", alpha=0.7
                )

        ax.plot([], [], color=PALETTE["ink"], lw=1.8, label="fitted decay with background wind")
        if legacy_factor is not None:
            ax.plot(
                [],
                [],
                color=PALETTE["ink"],
                lw=1.0,
                ls="--",
                label=f"legacy {legacy_factor:g}/h, no floor",
            )
        ax.axhline(64.0, color=PALETTE["muted"], lw=0.7, ls=":")
        ax.axhline(34.0, color=PALETTE["muted"], lw=0.7, ls=":")
        ax.set_xlim(0, max_hours)
        ax.set_ylim(0, None)
        ax.set_xlabel("Hours after landfall")
        ax.set_ylabel("Maximum sustained wind (kt)")
        ax.legend(fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="none", loc="upper right")
        if title:
            ax.set_title(title, loc="left", fontweight="bold")
    return fig, ax
