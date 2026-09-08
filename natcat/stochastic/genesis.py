"""Cyclogenesis model: where storms form and how strong they start.

Genesis locations are drawn from a Gaussian kernel density estimate fitted to
the first fix of every historical track, with over-land samples rejected.  The
initial intensity and size are copied from one of the ``n_neighbors`` nearest
historical genesis points, chosen at random.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["GENESIS_COLUMNS", "extract_genesis_points", "is_land", "GenesisModel"]

logger = logging.getLogger(__name__)

#: Columns of the genesis-point table.
GENESIS_COLUMNS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "time",
    "max_wind_speed_kt",
    "radius_max_wind_nm",
)


def is_land(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Boolean land mask for arrays of coordinates.

    Uses ``global_land_mask``, imported lazily so that ``import natcat`` does
    not require it. Coordinates outside the valid range are treated as land so
    that they get rejected by the genesis sampler.

    Parameters
    ----------
    lat, lon : numpy.ndarray
        Latitude and longitude in degrees.

    Returns
    -------
    numpy.ndarray
        ``True`` where the point is over land or out of range.

    Raises
    ------
    ImportError
        If ``global_land_mask`` is not installed.
    """
    try:
        from global_land_mask import globe
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "global_land_mask is required for the stochastic model. "
            "Install it with: pip install global-land-mask"
        ) from exc

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    valid = (lat >= -90.0) & (lat <= 90.0) & (lon >= -180.0) & (lon <= 180.0)
    out = np.ones(lat.shape, dtype=bool)
    if valid.any():
        out[valid] = globe.is_land(lat[valid], lon[valid])
    return out


def extract_genesis_points(tracks: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Collect the first fix of every historical track.

    Parameters
    ----------
    tracks : sequence of pandas.DataFrame
        Processed tracks, each sorted by ``time``.

    Returns
    -------
    pandas.DataFrame
        One row per track with columns ``latitude``, ``longitude``, ``time``,
        ``max_wind_speed_kt``, ``radius_max_wind_nm`` and ``year``. Rows with a
        non-finite position are dropped.

    Examples
    --------
    >>> extract_genesis_points([])
    Empty DataFrame
    Columns: [latitude, longitude, time, max_wind_speed_kt, radius_max_wind_nm, year]
    Index: []
    """
    rows = [track.iloc[0] for track in tracks if len(track) > 0]
    if not rows:
        empty = pd.DataFrame(columns=[*GENESIS_COLUMNS, "year"])
        return empty

    out = pd.DataFrame(
        {column: [row.get(column, np.nan) for row in rows] for column in GENESIS_COLUMNS}
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["latitude", "longitude"])
    out["time"] = pd.to_datetime(out["time"])
    out["year"] = out["time"].dt.year
    return out.reset_index(drop=True)


class GenesisModel:
    """Sampler for synthetic cyclogenesis points.

    Parameters
    ----------
    n_neighbors : int, default 5
        Number of nearest historical genesis points from which the initial
        intensity and radius are drawn.

    Attributes
    ----------
    points : pandas.DataFrame or None
        The historical genesis points this model was fitted on.
    kde : scipy.stats.gaussian_kde or None
        Fitted density over ``(latitude, longitude)``.

    Examples
    --------
    >>> model = GenesisModel().fit(tracks)                 # doctest: +SKIP
    >>> model.sample(10, rng=np.random.default_rng(0))     # doctest: +SKIP
    """

    def __init__(self, *, n_neighbors: int = 5) -> None:
        self.n_neighbors = int(n_neighbors)
        self.points: pd.DataFrame | None = None
        self.kde = None
        self._scaler = None
        self._knn = None

    def fit(self, tracks: Sequence[pd.DataFrame] | pd.DataFrame) -> GenesisModel:
        """Fit the genesis KDE and the nearest-neighbour intensity lookup.

        Parameters
        ----------
        tracks : sequence of pandas.DataFrame or pandas.DataFrame
            Historical tracks, or a ready-made genesis-point table.

        Returns
        -------
        GenesisModel
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If fewer than three usable genesis points are available.
        """
        from scipy import stats

        points = tracks if isinstance(tracks, pd.DataFrame) else extract_genesis_points(tracks)
        if len(points) < 3:
            raise ValueError(f"Need at least 3 genesis points to fit a KDE, got {len(points)}")

        self.points = points.reset_index(drop=True)
        self.kde = stats.gaussian_kde(
            np.vstack([self.points["latitude"].to_numpy(), self.points["longitude"].to_numpy()])
        )
        self._fit_neighbors()
        return self

    def _fit_neighbors(self) -> None:
        """Fit the standardised k-nearest-neighbour index over genesis points."""
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        features = self.points[["latitude", "longitude"]].to_numpy(dtype=np.float64)
        self._scaler = StandardScaler().fit(features)
        k = min(self.n_neighbors, len(self.points))
        self._knn = NearestNeighbors(n_neighbors=k).fit(self._scaler.transform(features))

    def sample_locations(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` over-ocean genesis locations from the fitted KDE.

        Parameters
        ----------
        n : int
            Number of locations to draw.
        rng : numpy.random.Generator
            Random generator; the only source of randomness used.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, 2)`` array of ``[latitude, longitude]``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if self.kde is None:
            raise RuntimeError("GenesisModel is not fitted. Call fit() first.")
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)

        batches: list[np.ndarray] = []
        collected = 0
        to_sample = max(1, int(n * 1.2))
        # Bounded retries: over-land rejection converges quickly in practice.
        for _ in range(100):
            drawn = self.kde.resample(to_sample, seed=rng)
            lat, lon = drawn[0], drawn[1]
            ocean = ~is_land(lat, lon)
            batch = np.column_stack((lat[ocean], lon[ocean]))
            batches.append(batch)
            collected += len(batch)
            if collected >= n:
                break
            to_sample = max(1, int((n - collected) * 1.2))
        else:  # pragma: no cover - defensive
            logger.warning(
                "Genesis sampling stopped after 100 batches with %d/%d points.", collected, n
            )

        return np.vstack(batches)[:n] if batches else np.empty((0, 2), dtype=np.float64)

    def sample(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Draw ``n`` synthetic genesis points with intensity and size.

        Parameters
        ----------
        n : int
            Number of genesis points.
        rng : numpy.random.Generator
            Random generator; the only source of randomness used.

        Returns
        -------
        pandas.DataFrame
            Columns ``latitude``, ``longitude``, ``max_wind_speed_kt`` and
            ``radius_max_wind_nm``. Empty (but correctly typed) when ``n <= 0``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        columns = ["latitude", "longitude", "max_wind_speed_kt", "radius_max_wind_nm"]
        if n <= 0:
            return pd.DataFrame({column: pd.Series(dtype=float) for column in columns})

        locations = self.sample_locations(n, rng)
        if len(locations) == 0:  # pragma: no cover - defensive
            return pd.DataFrame({column: pd.Series(dtype=float) for column in columns})

        scaled = self._scaler.transform(locations)
        _, indices = self._knn.kneighbors(scaled)
        choice = rng.integers(0, indices.shape[1], size=len(locations))
        picked = indices[np.arange(len(locations)), choice]

        return pd.DataFrame(
            {
                "latitude": locations[:, 0],
                "longitude": locations[:, 1],
                "max_wind_speed_kt": self.points["max_wind_speed_kt"].to_numpy()[picked],
                "radius_max_wind_nm": self.points["radius_max_wind_nm"].to_numpy()[picked],
            }
        )

    def __repr__(self) -> str:
        n = 0 if self.points is None else len(self.points)
        return f"GenesisModel(n_neighbors={self.n_neighbors}, n_historical_points={n})"
