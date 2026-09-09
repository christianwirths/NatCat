"""Download ATCF deck files from the NHC public FTP mirror.

All network calls carry an explicit timeout and raise on HTTP errors; nothing
returns ``None`` silently.  Downloaded ``.gz`` archives are extracted next to
themselves and the archive is removed, so callers always receive a ``.dat``
path.
"""

from __future__ import annotations

import gzip
import logging
import shutil
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from ..config import get_data_dir

__all__ = [
    "ARCHIVE_URL",
    "REALTIME_URL",
    "NHCClient",
    "download_best_track",
    "download_a_deck",
    "list_archive_files",
    "download_archive",
    "extract_gzip",
]

logger = logging.getLogger(__name__)

#: Yearly archive of finalised deck files.
ARCHIVE_URL = "https://ftp.nhc.noaa.gov/atcf/archive/"
#: Real-time directory used for the current season.
REALTIME_URL = "https://ftp.nhc.noaa.gov/atcf/aid_public/"


def _normalise_storm_number(storm_number: str | int) -> str:
    """Return a zero-padded two-digit ATCF storm number."""
    text = f"{int(storm_number):02d}" if not isinstance(storm_number, str) else storm_number.strip()
    if len(text) != 2 or not text.isdigit():
        raise ValueError(f"storm_number must be two digits, got {storm_number!r}")
    return text


def _raw_dir(data_dir: str | Path | None) -> Path:
    """Return the raw-data directory, creating it if needed."""
    base = Path(data_dir) if data_dir is not None else get_data_dir()
    raw = base / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    return raw


def extract_gzip(path: str | Path, *, remove_archive: bool = True) -> Path:
    """Decompress a ``.gz`` file in place.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.gz`` archive.
    remove_archive : bool, default True
        Delete the ``.gz`` file after a successful extraction.

    Returns
    -------
    pathlib.Path
        Path of the extracted file (the input path without ``.gz``).
    """
    path = Path(path)
    target = path.with_suffix("") if path.suffix == ".gz" else Path(str(path) + ".dat")
    with gzip.open(path, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    if remove_archive:
        path.unlink(missing_ok=True)
    return target


class NHCClient:
    """Small HTTP client for the NHC ATCF archive.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Root data directory. Files land in ``<data_dir>/raw``. Defaults to
        :func:`natcat.config.get_data_dir`.
    timeout : float, default 30
        Per-request timeout in seconds.

    Examples
    --------
    >>> client = NHCClient()                       # doctest: +SKIP
    >>> client.download_deck(2018, "al", "14")     # doctest: +SKIP
    PosixPath('.../data/raw/bal142018.dat')
    """

    def __init__(self, *, data_dir: str | Path | None = None, timeout: float = 30) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else get_data_dir()
        self.timeout = timeout

    # -- internals ---------------------------------------------------------
    def _get(self, url: str, *, stream: bool = False):
        """Issue a GET request, raising a descriptive error on failure."""
        import requests

        try:
            response = requests.get(url, timeout=self.timeout, stream=stream)
        except requests.RequestException as exc:  # pragma: no cover - network
            raise RuntimeError(f"Request to {url} failed: {exc}") from exc
        if response.status_code == 404:
            raise FileNotFoundError(f"Not found on NHC server: {url}")
        response.raise_for_status()
        return response

    def _deck_url(self, year: int, basin: str, storm_number: str, deck: str) -> tuple[str, str]:
        """Return ``(url, filename)`` for one storm's deck file."""
        filename = f"{deck.lower()}{basin.lower()}{storm_number}{year}.dat.gz"
        if year == datetime.now().year:
            return f"{REALTIME_URL}{filename}", filename
        return f"{ARCHIVE_URL}{year}/{filename}", filename

    # -- public API --------------------------------------------------------
    def download_deck(
        self,
        year: int,
        basin: str,
        storm_number: str | int,
        *,
        deck: str = "b",
        overwrite: bool = False,
    ) -> Path:
        """Download and extract one deck file.

        Parameters
        ----------
        year : int
            Season year, e.g. ``2018``.
        basin : str
            Two-letter basin code, e.g. ``'al'``.
        storm_number : str or int
            ATCF cyclone number within the season, e.g. ``'14'``.
        deck : {'b', 'a'}, default 'b'
            ``'b'`` for best track, ``'a'`` for forecast aids.
        overwrite : bool, default False
            Re-download even when the extracted file already exists.

        Returns
        -------
        pathlib.Path
            Path to the extracted ``.dat`` file.

        Raises
        ------
        FileNotFoundError
            If the NHC server has no such file.
        RuntimeError
            If the request fails for any other reason.
        """
        storm_number = _normalise_storm_number(storm_number)
        raw = _raw_dir(self.data_dir)
        url, filename = self._deck_url(year, basin, storm_number, deck)

        archive_path = raw / filename
        dat_path = raw / filename[: -len(".gz")]
        if dat_path.exists() and not overwrite:
            logger.debug("Using cached deck file %s", dat_path)
            return dat_path

        logger.info("Downloading %s", url)
        response = self._get(url, stream=True)
        with open(archive_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                handle.write(chunk)
        return extract_gzip(archive_path)

    def list_archive_files(self, year: int, basin: str = "al", deck: str = "b") -> list[str]:
        """List archived deck filenames for one season.

        Parameters
        ----------
        year : int
            Season year.
        basin : str, default 'al'
            Two-letter basin code.
        deck : {'b', 'a'}, default 'b'
            Deck type.

        Returns
        -------
        list of str
            Sorted ``.dat.gz`` filenames, e.g. ``['bal012018.dat.gz', ...]``.
        """
        from bs4 import BeautifulSoup

        url = f"{ARCHIVE_URL}{year}/"
        response = self._get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        prefix = f"{deck.lower()}{basin.lower()}"
        suffix = f"{year}.dat.gz"
        names = {
            href
            for link in soup.find_all("a")
            if (href := link.get("href")) and href.startswith(prefix) and href.endswith(suffix)
        }
        return sorted(names)

    def download_archive(
        self,
        years: Iterable[int],
        basin: str = "al",
        *,
        deck: str = "b",
        progress: bool = True,
    ) -> list[Path]:
        """Download every deck file for a range of seasons.

        Parameters
        ----------
        years : iterable of int
            Season years to fetch.
        basin : str, default 'al'
            Two-letter basin code.
        deck : {'b', 'a'}, default 'b'
            Deck type.
        progress : bool, default True
            Show a tqdm progress bar.

        Returns
        -------
        list of pathlib.Path
            Paths of every successfully extracted ``.dat`` file.
        """
        from tqdm.auto import tqdm

        jobs: list[tuple[int, str]] = []
        for year in years:
            for name in self.list_archive_files(year, basin, deck):
                jobs.append((year, name))

        if not jobs:
            logger.warning("No archive files found for the requested years.")
            return []

        raw = _raw_dir(self.data_dir)
        paths: list[Path] = []
        iterator = tqdm(jobs, desc="Downloading decks", unit="file") if progress else jobs
        for year, name in iterator:
            dat_path = raw / name[: -len(".gz")]
            if dat_path.exists():
                paths.append(dat_path)
                continue
            try:
                storm_number = name[3:5]
                paths.append(self.download_deck(year, basin, storm_number, deck=deck))
            except (FileNotFoundError, RuntimeError) as exc:
                logger.warning("Skipping %s: %s", name, exc)
        return paths


def download_best_track(
    year: int,
    basin: str,
    storm_number: str | int,
    *,
    data_dir: str | Path | None = None,
    overwrite: bool = False,
    timeout: float = 30,
) -> Path:
    """Download and extract one B-deck (best track) file.

    Parameters
    ----------
    year : int
        Season year, e.g. ``2018``.
    basin : str
        Two-letter basin code, e.g. ``'al'``.
    storm_number : str or int
        ATCF cyclone number, e.g. ``'14'``.
    data_dir : str or pathlib.Path, optional
        Root data directory; the file lands in ``<data_dir>/raw``.
    overwrite : bool, default False
        Re-download even when the extracted file exists.
    timeout : float, default 30
        Per-request timeout in seconds.

    Returns
    -------
    pathlib.Path
        Path to the extracted ``.dat`` file.
    """
    client = NHCClient(data_dir=data_dir, timeout=timeout)
    return client.download_deck(year, basin, storm_number, deck="b", overwrite=overwrite)


def download_a_deck(
    year: int,
    basin: str,
    storm_number: str | int,
    *,
    data_dir: str | Path | None = None,
    overwrite: bool = False,
    timeout: float = 30,
) -> Path:
    """Download and extract one A-deck (forecast aids) file.

    Parameters
    ----------
    year : int
        Season year.
    basin : str
        Two-letter basin code.
    storm_number : str or int
        ATCF cyclone number.
    data_dir : str or pathlib.Path, optional
        Root data directory.
    overwrite : bool, default False
        Re-download even when the extracted file exists.
    timeout : float, default 30
        Per-request timeout in seconds.

    Returns
    -------
    pathlib.Path
        Path to the extracted ``.dat`` file.
    """
    client = NHCClient(data_dir=data_dir, timeout=timeout)
    return client.download_deck(year, basin, storm_number, deck="a", overwrite=overwrite)


def list_archive_files(
    year: int,
    basin: str = "al",
    deck: str = "b",
    *,
    timeout: float = 30,
) -> list[str]:
    """List archived deck filenames for one season.

    Parameters
    ----------
    year : int
        Season year.
    basin : str, default 'al'
        Two-letter basin code.
    deck : {'b', 'a'}, default 'b'
        Deck type.
    timeout : float, default 30
        Per-request timeout in seconds.

    Returns
    -------
    list of str
        Sorted ``.dat.gz`` filenames.
    """
    return NHCClient(timeout=timeout).list_archive_files(year, basin, deck)


def download_archive(
    years: Iterable[int],
    basin: str = "al",
    *,
    data_dir: str | Path | None = None,
    deck: str = "b",
    progress: bool = True,
) -> list[Path]:
    """Download every deck file for a range of seasons.

    Parameters
    ----------
    years : iterable of int
        Season years to fetch.
    basin : str, default 'al'
        Two-letter basin code.
    data_dir : str or pathlib.Path, optional
        Root data directory.
    deck : {'b', 'a'}, default 'b'
        Deck type.
    progress : bool, default True
        Show a tqdm progress bar.

    Returns
    -------
    list of pathlib.Path
        Paths of every extracted ``.dat`` file.
    """
    client = NHCClient(data_dir=data_dir)
    return client.download_archive(years, basin, deck=deck, progress=progress)
