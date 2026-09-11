"""Calibration cases: one storm's footprint on its regional exposure.

The wind footprint of a storm does not depend on the vulnerability parameters,
so it is computed once per storm and stored together with the exposure values
it was evaluated on. Re-evaluating the portfolio loss for a new parameter set
is then a single vectorised pass over the stored arrays, which makes an
optimiser with hundreds of evaluations run in seconds.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..hazards.tropical_cyclone import TropicalCycloneHazard
from ..tracks.pipeline import load_best_track
from ..vulnerability.base import VulnerabilityModel

__all__ = [
    "CONUS_BOUNDS",
    "CaseInput",
    "StormCase",
    "build_cases",
    "cases_from_inputs",
    "load_cases",
    "load_inputs",
    "prepare_inputs",
    "save_cases",
    "save_inputs",
    "storm_region",
]

logger = logging.getLogger(__name__)

#: Contiguous United States as ``(lat_min, lat_max, lon_min, lon_max)``.
CONUS_BOUNDS: tuple[float, float, float, float] = (24.0, 50.0, -125.0, -66.0)


@dataclass
class StormCase:
    """A storm's precomputed footprint on its regional exposure.

    Attributes
    ----------
    storm_id : str
        ATCF identifier, e.g. ``'AL142018'``.
    name : str
        Storm name.
    observed_loss : float
        Observed loss in the exposure's reference-year USD.
    intensity : numpy.ndarray
        Maximum wind (kt) at every exposure location, shape ``(N,)``.
    tiv : numpy.ndarray
        Total insured value (USD) per location, shape ``(N,)``.
    construction : numpy.ndarray or None
        Per-location construction classes, if the exposure carries them.
    bounds : tuple of float
        Exposure box ``(lat_min, lat_max, lon_min, lon_max)`` the case covers.
    weight : float
        Objective weight of this storm (default 1).
    """

    storm_id: str
    name: str
    observed_loss: float
    intensity: np.ndarray
    tiv: np.ndarray
    construction: np.ndarray | None = None
    bounds: tuple[float, float, float, float] = CONUS_BOUNDS
    weight: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def n_locations(self) -> int:
        return int(self.intensity.shape[0])

    @property
    def total_value(self) -> float:
        return float(self.tiv.sum())

    def loss(self, vulnerability: VulnerabilityModel) -> float:
        """Portfolio ground-up loss for a vulnerability model."""
        damage_ratio = vulnerability.damage_ratio(self.intensity, self.construction, tiv=self.tiv)
        return float(np.dot(np.asarray(damage_ratio, dtype=np.float64), self.tiv))


@dataclass
class CaseInput:
    """Everything a :class:`StormCase` needs except the wind footprint.

    Tracks and regional exposure are the slow part to load; keeping them lets
    the footprint be recomputed cheaply for different hazard parameters.
    """

    storm_id: str
    name: str
    observed_loss: float
    track: pd.DataFrame
    portfolio: pd.DataFrame
    bounds: tuple[float, float, float, float]
    weight: float = 1.0
    meta: dict = field(default_factory=dict)


def storm_region(
    track: pd.DataFrame,
    *,
    margin_deg: float = 2.0,
    land_bounds: tuple[float, float, float, float] = CONUS_BOUNDS,
) -> tuple[float, float, float, float]:
    """Exposure box around the part of a track that lies inside ``land_bounds``.

    Parameters
    ----------
    track : pandas.DataFrame
        Processed track with ``latitude``/``longitude``.
    margin_deg : float, default 2.0
        Padding around the track's bounding box, in degrees.
    land_bounds : tuple of float, default CONUS_BOUNDS
        ``(lat_min, lat_max, lon_min, lon_max)`` region of interest.

    Returns
    -------
    tuple of float
        ``(lat_min, lat_max, lon_min, lon_max)``, clipped to ``land_bounds``.

    Raises
    ------
    ValueError
        If no track point lies inside ``land_bounds``.
    """
    lat_min, lat_max, lon_min, lon_max = land_bounds
    inside = track["latitude"].between(lat_min, lat_max) & track["longitude"].between(
        lon_min, lon_max
    )
    if not inside.any():
        raise ValueError("Track has no point inside the land bounds")
    part = track[inside]
    return (
        max(lat_min, float(part["latitude"].min()) - margin_deg),
        min(lat_max, float(part["latitude"].max()) + margin_deg),
        max(lon_min, float(part["longitude"].min()) - margin_deg),
        min(lon_max, float(part["longitude"].max()) + margin_deg),
    )


def _default_exposure_loader(bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    from ..exposure.litpop import load_litpop_exposure

    return load_litpop_exposure("USA", bounds=bounds)


def prepare_inputs(
    observed: pd.DataFrame,
    *,
    exposure_loader: Callable[[tuple[float, float, float, float]], pd.DataFrame] | None = None,
    margin_deg: float = 2.0,
    loss_column: str = "observed_loss_ref_usd",
    data_dir: str | Path | None = None,
    download: bool = True,
    progress: bool = True,
) -> list[CaseInput]:
    """Load the track and regional exposure for every row of an observed-loss table.

    Parameters
    ----------
    observed : pandas.DataFrame
        Output of :func:`natcat.calibration.normalise_losses` (needs
        ``loss_column``). Optional ``weight`` column.
    exposure_loader : callable, optional
        ``bounds -> portfolio DataFrame``; defaults to LitPop USA clipped to the
        storm region.
    margin_deg : float, default 2.0
        Padding around the landfall track for the exposure box.
    loss_column : str, default "observed_loss_ref_usd"
        Column holding the calibration target.
    data_dir, download
        Passed to :func:`natcat.tracks.load_best_track`.
    progress : bool, default True
        Show a progress bar.

    Returns
    -------
    list of CaseInput
        Storms whose track never reaches the land bounds, or whose region holds
        no exposure, are skipped with a warning.
    """
    if loss_column not in observed.columns:
        raise KeyError(f"observed table has no {loss_column!r} column; call normalise_losses first")
    loader = exposure_loader or _default_exposure_loader

    rows = list(observed.itertuples(index=False))
    if progress:
        from tqdm.auto import tqdm

        rows = tqdm(rows, desc="Loading storms")

    inputs: list[CaseInput] = []
    for row in rows:
        track = load_best_track(
            int(row.year),
            str(row.basin),
            str(row.storm_number),
            data_dir=data_dir,
            download=download,
        )
        try:
            bounds = storm_region(track, margin_deg=margin_deg)
        except ValueError:
            logger.warning("%s (%s): no track point over land, skipped", row.name, row.storm_id)
            continue
        portfolio = loader(bounds)
        if len(portfolio) == 0:
            logger.warning("%s (%s): empty exposure in %s, skipped", row.name, row.storm_id, bounds)
            continue
        inputs.append(
            CaseInput(
                storm_id=str(row.storm_id),
                name=str(row.name),
                observed_loss=float(getattr(row, loss_column)),
                track=track,
                portfolio=portfolio,
                bounds=bounds,
                weight=float(getattr(row, "weight", 1.0)),
                meta={"year": int(row.year), "n_track_points": int(len(track))},
            )
        )
    return inputs


def cases_from_inputs(
    inputs: Sequence[CaseInput],
    hazard_factory: Callable[[pd.DataFrame], TropicalCycloneHazard] = TropicalCycloneHazard,
) -> list[StormCase]:
    """Evaluate the wind footprint of every input with the given hazard.

    Parameters
    ----------
    inputs : sequence of CaseInput
        From :func:`prepare_inputs`.
    hazard_factory : callable, default TropicalCycloneHazard
        Builds the hazard from a processed track; use ``functools.partial`` to
        set ``decay_exponent`` / ``asymmetry_factor``.

    Returns
    -------
    list of StormCase
    """
    cases: list[StormCase] = []
    for item in inputs:
        coords = item.portfolio[["latitude", "longitude"]].to_numpy(dtype=np.float64)
        intensity = np.asarray(
            hazard_factory(item.track).compute_intensity(coords), dtype=np.float64
        )
        construction = (
            item.portfolio["construction"].to_numpy(dtype=object)
            if "construction" in item.portfolio.columns
            else None
        )
        cases.append(
            StormCase(
                storm_id=item.storm_id,
                name=item.name,
                observed_loss=item.observed_loss,
                intensity=intensity,
                tiv=item.portfolio["tiv"].to_numpy(dtype=np.float64),
                construction=construction,
                bounds=item.bounds,
                weight=item.weight,
                meta=dict(item.meta),
            )
        )
    return cases


def build_cases(
    observed: pd.DataFrame,
    *,
    hazard_factory: Callable[[pd.DataFrame], TropicalCycloneHazard] = TropicalCycloneHazard,
    inputs: Sequence[CaseInput] | None = None,
    **kwargs,
) -> list[StormCase]:
    """Compute one :class:`StormCase` per row of an observed-loss table.

    Shorthand for :func:`cases_from_inputs` applied to :func:`prepare_inputs`.

    Parameters
    ----------
    observed : pandas.DataFrame
        Output of :func:`natcat.calibration.normalise_losses`. Ignored when
        ``inputs`` is given.
    hazard_factory : callable, default TropicalCycloneHazard
        Builds the hazard from a processed track.
    inputs : sequence of CaseInput, optional
        Previously prepared inputs, to skip loading tracks and exposure.
    **kwargs
        Passed to :func:`prepare_inputs`.

    Returns
    -------
    list of StormCase
    """
    if inputs is None:
        inputs = prepare_inputs(observed, **kwargs)
    return cases_from_inputs(inputs, hazard_factory)


def save_inputs(inputs: Sequence[CaseInput], path: str | Path) -> None:
    """Pickle prepared inputs (tracks and regional exposure)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(list(inputs), handle)


def load_inputs(path: str | Path) -> list[CaseInput]:
    """Load inputs written by :func:`save_inputs`."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_cases(cases: Sequence[StormCase], path: str | Path) -> None:
    """Pickle a list of cases (footprints are the expensive part)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(list(cases), handle)


def load_cases(path: str | Path) -> list[StormCase]:
    """Load cases written by :func:`save_cases`."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
