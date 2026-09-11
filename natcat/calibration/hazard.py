"""Outer loop over hazard parameters around the vulnerability calibration.

The wind-field parameters (Rankine decay exponent, motion-asymmetry factor)
change every footprint, so they cannot be folded into the fast inner
optimisation. Instead the footprints are recomputed for each point of a small
grid, the vulnerability is calibrated at each point, and the combination with
the lowest calibrated objective wins.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from ..hazards.tropical_cyclone import TropicalCycloneHazard
from ..vulnerability.base import VulnerabilityModel
from .calibrator import CalibrationResult, Calibrator
from .cases import CaseInput, cases_from_inputs

__all__ = ["HazardGridResult", "calibrate_hazard_grid", "implied_decay_exponents"]

logger = logging.getLogger(__name__)


@dataclass
class HazardGridResult:
    """Outcome of :func:`calibrate_hazard_grid`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per grid point: ``decay_exponent``, ``asymmetry_factor``,
        ``objective``, ``factor_error`` (``10**sqrt(objective)``) and the
        calibrated parameters.
    results : dict
        ``(decay_exponent, asymmetry_factor) -> CalibrationResult``.
    best_hazard : dict
        Hazard parameters of the best grid point.
    best : CalibrationResult
        Vulnerability calibration at the best grid point.
    """

    table: pd.DataFrame
    results: dict[tuple[float, float], CalibrationResult]
    best_hazard: dict[str, float]
    best: CalibrationResult

    def hazard_factory(self):
        """``partial(TropicalCycloneHazard, **best_hazard)`` for reuse."""
        return partial(TropicalCycloneHazard, **self.best_hazard)

    def summary(self) -> str:
        """Human-readable summary of the grid and the winner."""
        lines = [
            "Hazard grid (calibrated factor error per point):",
            self.table[["decay_exponent", "asymmetry_factor", "factor_error"]]
            .pivot(index="asymmetry_factor", columns="decay_exponent", values="factor_error")
            .round(2)
            .to_string(),
            f"Best: decay_exponent={self.best_hazard['decay_exponent']:g}, "
            f"asymmetry_factor={self.best_hazard['asymmetry_factor']:g}",
            self.best.summary(),
        ]
        return "\n".join(lines)


def calibrate_hazard_grid(
    inputs: Sequence[CaseInput],
    model: VulnerabilityModel,
    *,
    decay_exponents: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0),
    asymmetry_factors: Sequence[float] = (0.3, 0.5, 0.7),
    param_names: Sequence[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    method: str = "global",
    maxiter: int = 200,
    seed: int | None = None,
    progress: bool = True,
) -> HazardGridResult:
    """Calibrate the vulnerability at every point of a hazard-parameter grid.

    Parameters
    ----------
    inputs : sequence of CaseInput
        From :func:`natcat.calibration.prepare_inputs`.
    model : VulnerabilityModel
        Starting vulnerability model (see :class:`Calibrator`).
    decay_exponents : sequence of float
        Rankine decay exponents to try.
    asymmetry_factors : sequence of float
        Motion-asymmetry factors to try.
    param_names, bounds, method, maxiter, seed
        Passed to :class:`Calibrator` / :meth:`Calibrator.fit`.
    progress : bool, default True
        Show a progress bar over grid points.

    Returns
    -------
    HazardGridResult
    """
    if len(inputs) == 0:
        raise ValueError("calibrate_hazard_grid needs at least one input")
    points = [(float(e), float(a)) for e in decay_exponents for a in asymmetry_factors]
    if progress:
        from tqdm.auto import tqdm

        points = tqdm(points, desc="Hazard grid")

    results: dict[tuple[float, float], CalibrationResult] = {}
    rows = []
    for exponent, asymmetry in points:
        factory = partial(
            TropicalCycloneHazard, decay_exponent=exponent, asymmetry_factor=asymmetry
        )
        cases = cases_from_inputs(inputs, factory)
        calibrator = Calibrator(cases, model, param_names=param_names, bounds=bounds)
        result = calibrator.fit(method=method, maxiter=maxiter, seed=seed)
        results[(exponent, asymmetry)] = result
        rows.append(
            {
                "decay_exponent": exponent,
                "asymmetry_factor": asymmetry,
                "initial_objective": result.initial_objective,
                "objective": result.objective,
                "factor_error": float(10 ** np.sqrt(result.objective)),
                **result.params,
            }
        )
        logger.info(
            "exponent=%g asymmetry=%g -> factor error %.2fx",
            exponent, asymmetry, 10 ** np.sqrt(result.objective),
        )  # fmt: skip

    table = pd.DataFrame(rows)
    best_row = table.loc[table["objective"].idxmin()]
    best_key = (float(best_row["decay_exponent"]), float(best_row["asymmetry_factor"]))
    return HazardGridResult(
        table=table,
        results=results,
        best_hazard={"decay_exponent": best_key[0], "asymmetry_factor": best_key[1]},
        best=results[best_key],
    )


def implied_decay_exponents(source: str | Path, *, min_wind_kt: float = 64.0) -> pd.DataFrame:
    """Radial decay exponents implied by observed best-track wind radii.

    For every hurricane-strength fix that carries an observed radius of
    maximum wind and a 34/50/64 kt wind radius, the Rankine profile
    ``v = vmax * (rmw / r) ** n`` is inverted for ``n`` using the mean of the
    reported quadrant radii. This is an independent, loss-free check on the
    decay exponent picked by :func:`calibrate_hazard_grid`.

    Parameters
    ----------
    source : str or pathlib.Path
        A B-deck file or a directory of ``b*.dat`` files.
    min_wind_kt : float, default 64.0
        Only fixes at or above this intensity are used.

    Returns
    -------
    pandas.DataFrame
        One row per fix and threshold: ``storm_id``, ``vmax``, ``rmw``,
        ``threshold``, ``radius`` (mean quadrant radius, nm) and ``exponent``.

    Examples
    --------
    >>> implied_decay_exponents("data/raw")["exponent"].median()  # doctest: +SKIP
    0.53
    """
    source = Path(source)
    paths = sorted(source.glob("b*.dat")) if source.is_dir() else [source]
    rows = []
    for path in paths:
        with path.open(errors="ignore") as handle:
            for line in handle:
                fields = [item.strip() for item in line.split(",")]
                if len(fields) < 20 or fields[4].upper() != "BEST":
                    continue
                try:
                    vmax = float(fields[8])
                    threshold = float(fields[11])
                    quadrants = [float(item) for item in fields[13:17]]
                    rmw = float(fields[19])
                except ValueError:
                    continue
                reported = [radius for radius in quadrants if radius > 0]
                if vmax < min_wind_kt or rmw <= 0 or threshold not in (34.0, 50.0, 64.0):
                    continue
                if not reported or vmax <= threshold:
                    continue
                radius = float(np.mean(reported))
                if radius <= rmw:
                    continue
                storm_id = f"{fields[0]}{fields[1]}{fields[2][:4]}"
                rows.append(
                    (
                        storm_id,
                        vmax,
                        rmw,
                        threshold,
                        radius,
                        np.log(vmax / threshold) / np.log(radius / rmw),
                    )
                )
    return pd.DataFrame(
        rows, columns=["storm_id", "vmax", "rmw", "threshold", "radius", "exponent"]
    )
