"""Empirical Markov model for storm motion and intensity change.

The Atlantic is divided into a regular latitude/longitude grid (2 deg by
default).  Every historical time step contributes one transition tuple
``(translation_speed_kt, heading_deg, delta_vmax, delta_rmw)`` to the cell it
starts in.  A synthetic storm advances by drawing one of those tuples uniformly
at random from its current cell.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..utils.geo import destination_point
from ..utils.units import kt2kmh
from .decay import LandDecayModel
from .genesis import is_land

__all__ = ["TRANSITION_COLUMNS", "build_state_table", "TransitionModel", "step_track"]

logger = logging.getLogger(__name__)

#: Order of the four values in a transition tuple.
TRANSITION_COLUMNS: tuple[str, ...] = (
    "translation_speed_kt",
    "heading_deg",
    "delta_vmax",
    "delta_rmw",
)


def _state_id(latitude: float, longitude: float, grid_size: float) -> str:
    """Return the grid-cell identifier containing a point."""
    lat_bin = np.floor(latitude / grid_size) * grid_size
    lon_bin = np.floor(longitude / grid_size) * grid_size
    return f"{lat_bin}_{lon_bin}"


def build_state_table(
    tracks: Sequence[pd.DataFrame],
    grid_size: float = 2.0,
) -> pd.DataFrame:
    """Stack historical tracks into a table of grid-cell transitions.

    Parameters
    ----------
    tracks : sequence of pandas.DataFrame
        Processed tracks carrying ``latitude``, ``longitude``,
        ``max_wind_speed_kt``, ``radius_max_wind_nm``,
        ``translation_speed_kt`` and ``heading_deg``.
    grid_size : float, default 2.0
        Cell size of the state grid, in degrees.

    Returns
    -------
    pandas.DataFrame
        One row per usable time step, with ``state_id``, ``lat_bin``,
        ``lon_bin`` and the four :data:`TRANSITION_COLUMNS`. Empty input yields
        an empty, correctly typed frame.
    """
    columns = ["state_id", "lat_bin", "lon_bin", *TRANSITION_COLUMNS]
    frames = []
    for index, track in enumerate(tracks):
        if len(track) < 2:
            continue
        frame = track.copy()
        if "storm_id" not in frame.columns:
            frame["storm_id"] = f"HIST_{index}"
        frame["storm_id"] = frame["storm_id"].astype(str)
        frames.append(frame)

    if not frames:
        return pd.DataFrame({column: pd.Series(dtype=float) for column in columns})

    master = pd.concat(frames, ignore_index=True)
    master["delta_vmax"] = master.groupby("storm_id")["max_wind_speed_kt"].diff()
    master["delta_rmw"] = master.groupby("storm_id")["radius_max_wind_nm"].diff()
    master = master.dropna(subset=list(TRANSITION_COLUMNS))

    master["lat_bin"] = np.floor(master["latitude"] / grid_size) * grid_size
    master["lon_bin"] = np.floor(master["longitude"] / grid_size) * grid_size
    master["state_id"] = master["lat_bin"].astype(str) + "_" + master["lon_bin"].astype(str)
    return master[columns].reset_index(drop=True)


class TransitionModel:
    """Uniform empirical sampler over per-cell historical transitions.

    Parameters
    ----------
    grid_size : float, default 2.0
        Cell size of the state grid, in degrees.

    Attributes
    ----------
    table : dict of str to numpy.ndarray
        Maps a cell identifier to its ``(n, 4)`` array of transition tuples.

    Examples
    --------
    >>> model = TransitionModel().fit(tracks)   # doctest: +SKIP
    >>> model.n_states                          # doctest: +SKIP
    """

    def __init__(self, *, grid_size: float = 2.0) -> None:
        self.grid_size = float(grid_size)
        self.table: dict[str, np.ndarray] = {}

    def fit(self, tracks: Sequence[pd.DataFrame] | pd.DataFrame) -> TransitionModel:
        """Build the per-cell transition lookup.

        Parameters
        ----------
        tracks : sequence of pandas.DataFrame or pandas.DataFrame
            Historical tracks, or a ready-made state table from
            :func:`build_state_table`.

        Returns
        -------
        TransitionModel
            ``self``, for chaining.
        """
        states = (
            tracks
            if isinstance(tracks, pd.DataFrame) and "state_id" in tracks.columns
            else build_state_table(tracks, grid_size=self.grid_size)
        )
        self.table = {
            state_id: group[list(TRANSITION_COLUMNS)].to_numpy(dtype=np.float64)
            for state_id, group in states.groupby("state_id")
        }
        return self

    @property
    def n_states(self) -> int:
        """Number of grid cells with at least one historical transition."""
        return len(self.table)

    def state_id(self, latitude: float, longitude: float) -> str:
        """Return the grid-cell identifier containing a point.

        Parameters
        ----------
        latitude, longitude : float
            Position in degrees.

        Returns
        -------
        str
            Cell identifier, e.g. ``'24.0_-80.0'``.
        """
        return _state_id(latitude, longitude, self.grid_size)

    def sample(
        self,
        latitude: float,
        longitude: float,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        """Draw one transition tuple for the cell containing a point.

        Parameters
        ----------
        latitude, longitude : float
            Current storm position, in degrees.
        rng : numpy.random.Generator
            Random generator; the only source of randomness used.

        Returns
        -------
        numpy.ndarray or None
            Shape ``(4,)`` array of :data:`TRANSITION_COLUMNS`, or ``None``
            when no historical storm has ever occupied that cell.
        """
        options = self.table.get(self.state_id(latitude, longitude))
        if options is None or len(options) == 0:
            return None
        return options[rng.integers(0, len(options))]

    def __repr__(self) -> str:
        return f"TransitionModel(grid_size={self.grid_size}, n_states={self.n_states})"


def step_track(
    state: dict[str, float],
    model: TransitionModel,
    rng: np.random.Generator,
    *,
    time_step_h: float = 3.0,
    land_decay: float | LandDecayModel | None = None,
    land_rmw_growth: float = 1.02,
    min_rmw_nm: float = 5.0,
    max_wind_kt: float = 185.0,
    max_rmw_nm: float = 150.0,
) -> tuple[dict[str, float], dict[str, float]] | None:
    """Advance a synthetic storm by one time step.

    Over water the intensity and radius follow the sampled historical deltas.
    Over land the wind follows the :class:`LandDecayModel` (decay towards a
    background wind) and the radius grows as ``land_rmw_growth ** dt``. A step
    that crosses the coast is split: the land rule acts for the fraction of
    the step spent over land (half, when only one end point is on land) and
    the sampled delta is scaled by the remainder.

    Parameters
    ----------
    state : dict
        Current state with ``latitude``, ``longitude``, ``max_wind_speed_kt``
        and ``radius_max_wind_nm``. Never mutated.
    model : TransitionModel
        Fitted transition sampler.
    rng : numpy.random.Generator
        Random generator; the only source of randomness used.
    time_step_h : float, default 3.0
        Length of the step, in hours.
    land_decay : float or LandDecayModel, optional
        Inland decay. A :class:`LandDecayModel` (default: its fitted Atlantic
        parameters) or, for the legacy behaviour, a per-hour multiplicative
        factor such as ``0.92`` (no background wind).
    land_rmw_growth : float, default 1.02
        Per-hour multiplicative radius growth over land.
    min_rmw_nm : float, default 5.0
        Floor on the radius of maximum wind.
    max_wind_kt : float, default 185.0
        Physical cap on the maximum sustained wind. The unbounded random walk
        of intensity deltas would otherwise occasionally exceed the strongest
        storm on record (about 165 kt in the Atlantic).
    max_rmw_nm : float, default 150.0
        Physical cap on the radius of maximum wind, which would otherwise drift
        to several hundred nautical miles in long-lived synthetic storms.

    Returns
    -------
    tuple of (dict, dict) or None
        ``(current, next)`` where ``current`` is ``state`` annotated with the
        sampled ``translation_speed_kt`` and ``heading_deg`` used to leave it,
        and ``next`` is the new state. ``None`` when the cell has no historical
        transitions, i.e. the storm dies.
    """
    transition = model.sample(state["latitude"], state["longitude"], rng)
    if transition is None:
        return None

    speed_kt, heading_deg, delta_vmax, delta_rmw = transition

    current = dict(state)
    current["translation_speed_kt"] = float(speed_kt)
    current["heading_deg"] = float(heading_deg)

    distance_km = kt2kmh(speed_kt) * time_step_h
    next_lat, next_lon = destination_point(
        state["latitude"], state["longitude"], heading_deg, distance_km
    )
    next_lat = float(next_lat)
    next_lon = float(next_lon)

    decay = (
        LandDecayModel()
        if land_decay is None
        else land_decay
        if isinstance(land_decay, LandDecayModel)
        else LandDecayModel.from_rate(float(land_decay))
    )
    land_now = bool(
        state.get(
            "_over_land", is_land(np.array([state["latitude"]]), np.array([state["longitude"]]))[0]
        )
    )
    land_next = bool(is_land(np.array([next_lat]), np.array([next_lon]))[0])
    land_fraction = (float(land_now) + float(land_next)) / 2.0

    if land_fraction > 0.0:
        land_hours = land_fraction * time_step_h
        next_vmax = float(decay.step(state["max_wind_speed_kt"], land_hours))
        next_vmax += (1.0 - land_fraction) * delta_vmax
        next_rmw = state["radius_max_wind_nm"] * (land_rmw_growth**land_hours)
        next_rmw += (1.0 - land_fraction) * delta_rmw
    else:
        next_vmax = state["max_wind_speed_kt"] + delta_vmax
        next_rmw = state["radius_max_wind_nm"] + delta_rmw

    following = dict(current)
    following["latitude"] = next_lat
    following["longitude"] = next_lon
    following["max_wind_speed_kt"] = float(np.clip(next_vmax, 0.0, max_wind_kt))
    following["radius_max_wind_nm"] = float(np.clip(next_rmw, min_rmw_nm, max_rmw_nm))
    following["_over_land"] = float(land_next)
    return current, following
