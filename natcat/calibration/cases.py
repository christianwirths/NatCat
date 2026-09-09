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

__all__ = ["CONUS_BOUNDS", "StormCase", "build_cases", "load_cases", "save_cases", "storm_region"]

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


def build_cases(
    observed: pd.DataFrame,
    *,
    exposure_loader: Callable[[tuple[float, float, float, float]], pd.DataFrame] | None = None,
    hazard_factory: Callable[[pd.DataFrame], TropicalCycloneHazard] = TropicalCycloneHazard,
    margin_deg: float = 2.0,
    loss_column: str = "observed_loss_ref_usd",
    data_dir: str | Path | None = None,
    download: bool = True,
    progress: bool = True,
) -> list[StormCase]:
    """Compute one :class:`StormCase` per row of an observed-loss table.

    Parameters
    ----------
    observed : pandas.DataFrame
        Output of :func:`natcat.calibration.normalise_losses` (needs
        ``loss_column``). Optional ``weight`` column.
    exposure_loader : callable, optional
        ``bounds -> portfolio DataFrame``; defaults to LitPop USA clipped to the
        storm region.
    hazard_factory : callable, default TropicalCycloneHazard
        Builds the hazard from a processed track (use ``functools.partial`` to
        change hazard parameters).
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
    list of StormCase
        Storms whose track never reaches the land bounds are skipped with a
        warning.
    """
    if loss_column not in observed.columns:
        raise KeyError(f"observed table has no {loss_column!r} column; call normalise_losses first")
    loader = exposure_loader or _default_exposure_loader

    rows = list(observed.itertuples(index=False))
    if progress:
        from tqdm.auto import tqdm

        rows = tqdm(rows, desc="Building cases")

    cases: list[StormCase] = []
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
        coords = portfolio[["latitude", "longitude"]].to_numpy(dtype=np.float64)
        intensity = np.asarray(hazard_factory(track).compute_intensity(coords), dtype=np.float64)
        construction = (
            portfolio["construction"].to_numpy(dtype=object)
            if "construction" in portfolio.columns
            else None
        )
        cases.append(
            StormCase(
                storm_id=str(row.storm_id),
                name=str(row.name),
                observed_loss=float(getattr(row, loss_column)),
                intensity=intensity,
                tiv=portfolio["tiv"].to_numpy(dtype=np.float64),
                construction=construction,
                bounds=bounds,
                weight=float(getattr(row, "weight", 1.0)),
                meta={"year": int(row.year), "n_track_points": int(len(track))},
            )
        )
    return cases


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
