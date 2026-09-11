"""Synthetic tropical cyclone catalog: fit on history, generate new storms.

The catalog chains three fitted components:

1. :class:`~natcat.stochastic.genesis.GenesisModel` -- where and how strongly
   storms form;
2. :class:`~natcat.stochastic.transitions.TransitionModel` -- how they move and
   change intensity, as an empirical Markov chain over a lat/lon grid;
3. :class:`~natcat.stochastic.frequency.PoissonFrequency` -- how many form per
   season.

All randomness flows through a single :class:`numpy.random.Generator` stored on
the instance, so a given ``seed`` fully determines the generated catalog.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.atcf import read_best_track
from ..data.quality import validate_track
from ..tracks.processing import prepare_track
from .decay import LandDecayModel
from .frequency import PoissonFrequency
from .genesis import GenesisModel, extract_genesis_points
from .transitions import TransitionModel, step_track

__all__ = ["CATALOG_COLUMNS", "REFERENCE_TIME", "SyntheticTCCatalog"]

logger = logging.getLogger(__name__)

#: Columns of a generated catalog (``year`` is added when ``n_years`` is used).
CATALOG_COLUMNS: tuple[str, ...] = (
    "storm_id",
    "time",
    "hour",
    "latitude",
    "longitude",
    "max_wind_speed_kt",
    "radius_max_wind_nm",
    "translation_speed_kt",
    "heading_deg",
)

#: Synthetic tracks carry relative times measured from this epoch; only offsets
#: within a storm are meaningful.
REFERENCE_TIME = pd.Timestamp("1900-01-01")


class SyntheticTCCatalog:
    """Fit an empirical stochastic TC model and generate synthetic catalogs.

    Parameters
    ----------
    basin : str, default 'al'
        Two-letter basin code used to select ``b<basin>*.dat`` files.
    grid_size : float, default 2.0
        Cell size of the Markov state grid, in degrees.
    time_step_h : float, default 3.0
        Track time step, in hours.
    max_hours : int, default 720
        Maximum simulated lifetime of one storm, in hours (30 days).
    n_neighbors : int, default 5
        Nearest historical genesis points used for the initial intensity.
    min_wind_kt : float, default 15.0
        Wind speed below which a storm is considered dissipated.
    land_decay : float or LandDecayModel, optional
        Inland decay rule. ``None`` (default) fits a :class:`LandDecayModel`
        (decay rate and background wind) to the historical landfalls during
        :meth:`fit`; a model instance is used as given; a float such as
        ``0.92`` restores the legacy per-hour multiplicative decay with no
        background wind.
    land_remnant_hours : float, default 24.0
        A storm over land that has stayed below ``remnant_wind_kt`` for this
        long is terminated. Best tracks carry inland remnants for days; they
        cause no modelled damage and would otherwise run to ``max_hours``.
    remnant_wind_kt : float, default 34.0
        Wind below which an over-land storm counts as a remnant.
    land_rmw_growth : float, default 1.02
        Per-hour multiplicative radius growth over land.
    max_wind_kt : float, default 185.0
        Physical cap on synthetic maximum sustained wind.
    max_rmw_nm : float, default 150.0
        Physical cap on the synthetic radius of maximum wind.
    seed : int, optional
        Seed for the single internal :class:`numpy.random.Generator`.

    Attributes
    ----------
    genesis : GenesisModel
    transitions : TransitionModel
    frequency : PoissonFrequency
    land_decay : LandDecayModel
        Inland decay, fitted in :meth:`fit` unless given.
    tracks : list of pandas.DataFrame
        The historical tracks the model was fitted on.
    rng : numpy.random.Generator
        The only source of randomness.

    Examples
    --------
    >>> catalog = SyntheticTCCatalog(seed=0).fit("data/raw")   # doctest: +SKIP
    >>> storms = catalog.generate(n_years=100)                 # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        basin: str = "al",
        grid_size: float = 2.0,
        time_step_h: float = 3.0,
        max_hours: int = 720,
        n_neighbors: int = 5,
        min_wind_kt: float = 15.0,
        land_decay: float | LandDecayModel | None = None,
        land_remnant_hours: float = 24.0,
        remnant_wind_kt: float = 34.0,
        land_rmw_growth: float = 1.02,
        max_wind_kt: float = 185.0,
        max_rmw_nm: float = 150.0,
        seed: int | None = None,
    ) -> None:
        self.basin = basin
        self.grid_size = float(grid_size)
        self.time_step_h = float(time_step_h)
        self.max_hours = int(max_hours)
        self.n_neighbors = int(n_neighbors)
        self.min_wind_kt = float(min_wind_kt)
        self.land_decay: LandDecayModel | None = (
            None
            if land_decay is None
            else land_decay
            if isinstance(land_decay, LandDecayModel)
            else LandDecayModel.from_rate(float(land_decay))
        )
        self._fit_land_decay = land_decay is None
        self.land_remnant_hours = float(land_remnant_hours)
        self.remnant_wind_kt = float(remnant_wind_kt)
        self.land_rmw_growth = float(land_rmw_growth)
        self.max_wind_kt = float(max_wind_kt)
        self.max_rmw_nm = float(max_rmw_nm)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.genesis = GenesisModel(n_neighbors=self.n_neighbors)
        self.transitions = TransitionModel(grid_size=self.grid_size)
        self.frequency = PoissonFrequency()
        self.tracks: list[pd.DataFrame] = []
        self.failed_files: list[tuple[str, str]] = []
        self._is_fitted = False

    # -- fitting -----------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed successfully."""
        return self._is_fitted

    @property
    def freq(self) -> str:
        """Pandas frequency string matching :attr:`time_step_h`."""
        return f"{self.time_step_h:g}h"

    def load_tracks(
        self,
        data_dir: str | Path,
        *,
        progress: bool = False,
        max_files: int | None = None,
    ) -> list[pd.DataFrame]:
        """Read and process every B-deck file in a directory.

        Parameters
        ----------
        data_dir : str or pathlib.Path
            Directory containing ``b<basin>*.dat`` files.
        progress : bool, default False
            Show a tqdm progress bar.
        max_files : int, optional
            Read only the first ``max_files`` files (sorted by name). Useful to
            keep tests fast.

        Returns
        -------
        list of pandas.DataFrame
            Processed tracks that passed quality control. Files that fail are
            recorded on :attr:`failed_files`.

        Raises
        ------
        FileNotFoundError
            If the directory does not exist or holds no matching files.
        """
        directory = Path(data_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"Not a directory: {directory}")

        paths = sorted(directory.glob(f"b{self.basin}*.dat"))
        if max_files is not None:
            paths = paths[:max_files]
        if not paths:
            raise FileNotFoundError(f"No b{self.basin}*.dat files found in {directory}")

        iterator = paths
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(paths, desc="Loading best tracks", unit="file")

        tracks: list[pd.DataFrame] = []
        self.failed_files = []
        for path in iterator:
            try:
                raw = read_best_track(path)
                ok, reason = validate_track(raw)
                if not ok:
                    self.failed_files.append((str(path), reason))
                    continue
                # The full life cycle (including decay) is needed to learn transitions;
                # truncation is applied by the LossSimulator when losses are evaluated.
                tracks.append(prepare_track(raw, freq=self.freq, truncate_after_hurricane=False))
            except (ValueError, KeyError, OSError) as exc:
                self.failed_files.append((str(path), repr(exc)))

        logger.info("Loaded %d tracks (%d skipped).", len(tracks), len(self.failed_files))
        return tracks

    def fit(
        self,
        data_dir: str | Path,
        *,
        progress: bool = False,
        max_files: int | None = None,
    ) -> SyntheticTCCatalog:
        """Calibrate genesis, transition and frequency models on history.

        Parameters
        ----------
        data_dir : str or pathlib.Path
            Directory of ATCF B-deck files.
        progress : bool, default False
            Show a tqdm progress bar while reading files.
        max_files : int, optional
            Read only the first ``max_files`` files.

        Returns
        -------
        SyntheticTCCatalog
            ``self``, for chaining.

        Raises
        ------
        FileNotFoundError
            If no usable files are found.
        ValueError
            If too few tracks survive quality control to fit the genesis KDE.
        """
        self.tracks = self.load_tracks(data_dir, progress=progress, max_files=max_files)
        if self._fit_land_decay:
            self.land_decay = LandDecayModel.fit(self.tracks)
        genesis_points = extract_genesis_points(self.tracks)

        self.genesis.fit(genesis_points)
        self.transitions.fit(self.tracks)
        self.frequency.fit(genesis_points["year"])

        self._is_fitted = True
        logger.info(
            "Fitted on %d tracks: %d states, lambda=%.2f storms/year.",
            len(self.tracks),
            self.transitions.n_states,
            self.frequency.rate,
        )
        return self

    # -- generation --------------------------------------------------------
    def _empty_catalog(self, *, with_year: bool) -> pd.DataFrame:
        """Return a correctly typed, empty catalog frame."""
        columns = {
            "storm_id": pd.Series(dtype=object),
            "time": pd.Series(dtype="datetime64[ns]"),
        }
        for name in CATALOG_COLUMNS:
            if name not in columns:
                columns[name] = pd.Series(dtype=float)
        if with_year:
            columns["year"] = pd.Series(dtype=int)
        return pd.DataFrame(columns)

    def _walk(self, start: dict[str, float]) -> pd.DataFrame:
        """Propagate one storm from genesis until it dissipates."""
        n_steps = max(1, int(self.max_hours // self.time_step_h))

        state = dict(start)
        state.setdefault("translation_speed_kt", 0.0)
        state.setdefault("heading_deg", 0.0)
        state["hour"] = 0.0

        states: list[dict[str, float]] = []
        remnant_hours = 0.0
        for step in range(1, n_steps + 1):
            outcome = step_track(
                state,
                self.transitions,
                self.rng,
                time_step_h=self.time_step_h,
                land_decay=self.land_decay,
                land_rmw_growth=self.land_rmw_growth,
                max_wind_kt=self.max_wind_kt,
                max_rmw_nm=self.max_rmw_nm,
            )
            if outcome is None:
                states.append(state)
                break
            current, following = outcome
            states.append(current)
            following["hour"] = step * self.time_step_h
            state = following
            if state["max_wind_speed_kt"] < self.min_wind_kt:
                states.append(state)
                break
            if state.get("_over_land") and state["max_wind_speed_kt"] < self.remnant_wind_kt:
                remnant_hours += self.time_step_h
                if remnant_hours >= self.land_remnant_hours:
                    states.append(state)
                    break
            else:
                remnant_hours = 0.0
        else:
            states.append(state)

        track = pd.DataFrame(states).drop(columns=["_over_land"], errors="ignore")
        track["time"] = REFERENCE_TIME + pd.to_timedelta(track["hour"], unit="h")
        return track

    def generate(
        self,
        n_storms: int | None = None,
        n_years: int | None = None,
        *,
        progress: bool = False,
    ) -> pd.DataFrame:
        """Generate a synthetic storm catalog.

        Parameters
        ----------
        n_storms : int, optional
            Exact number of storms to generate.
        n_years : int, optional
            Number of synthetic seasons; the storm count of each is drawn from
            the fitted Poisson frequency. Takes precedence over ``n_storms``.
        progress : bool, default False
            Show a tqdm progress bar.

        Returns
        -------
        pandas.DataFrame
            Columns :data:`CATALOG_COLUMNS`, plus ``year`` when ``n_years`` is
            given. ``n_storms=0`` (or a run of empty Poisson years) returns an
            empty frame with the same columns.

        Raises
        ------
        RuntimeError
            If the catalog has not been fitted.
        ValueError
            If neither ``n_storms`` nor ``n_years`` is provided, or if either
            is negative.

        Examples
        --------
        >>> catalog.generate(n_storms=0).empty   # doctest: +SKIP
        True
        """
        if not self._is_fitted:
            raise RuntimeError("SyntheticTCCatalog is not fitted. Call fit() first.")

        years: np.ndarray | None = None
        if n_years is not None:
            if n_years < 0:
                raise ValueError(f"n_years must be non-negative, got {n_years}")
            counts = self.frequency.sample(n_years, self.rng)
            years = np.repeat(np.arange(len(counts)), counts)
            n_storms = int(counts.sum())
        elif n_storms is None:
            raise ValueError("Provide either n_storms or n_years.")
        if n_storms < 0:
            raise ValueError(f"n_storms must be non-negative, got {n_storms}")

        if n_storms == 0:
            return self._empty_catalog(with_year=years is not None)

        origins = self.genesis.sample(n_storms, self.rng)
        iterator = range(len(origins))
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc="Generating storms", unit="storm")

        frames = []
        for i in iterator:
            start = origins.iloc[i].to_dict()
            track = self._walk(start)
            track["storm_id"] = f"SYN_{i:06d}"
            if years is not None:
                track["year"] = int(years[i])
            frames.append(track)

        if not frames:  # pragma: no cover - defensive
            return self._empty_catalog(with_year=years is not None)

        catalog = pd.concat(frames, ignore_index=True)
        ordered = [c for c in CATALOG_COLUMNS if c in catalog.columns]
        extras = [c for c in catalog.columns if c not in ordered]
        return catalog[[*ordered, *extras]]

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Pickle the fitted catalog to disk.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file.

        Returns
        -------
        pathlib.Path
            The path written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)
        return path

    @staticmethod
    def load(path: str | Path) -> SyntheticTCCatalog:
        """Load a catalog previously written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            Pickle file to read.

        Returns
        -------
        SyntheticTCCatalog
            The restored catalog.

        Raises
        ------
        TypeError
            If the pickle does not contain a :class:`SyntheticTCCatalog`.
        """
        with open(path, "rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, SyntheticTCCatalog):
            raise TypeError(f"{path} does not contain a SyntheticTCCatalog")
        return obj

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"SyntheticTCCatalog(basin={self.basin!r}, grid_size={self.grid_size}, "
            f"time_step_h={self.time_step_h}, max_hours={self.max_hours}, status={status})"
        )
