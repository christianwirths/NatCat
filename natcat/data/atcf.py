"""Readers for ATCF A-deck (forecast) and B-deck (best track) files.

ATCF deck files are comma-separated with a *positional* layout and a variable
number of trailing fields: modern B-decks carry 38-40 fields while historic
(pre-1950) files may carry as few as 25.  Both decks share the same layout for
the leading fields, so a single positional specification is used and columns are
selected by index rather than by a fixed header.

References
----------
https://www.nrlmry.navy.mil/atcf_web/docs/database/new/abdeck.txt
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "ATCF_FIELDS",
    "MAX_ATCF_FIELDS",
    "parse_lat_lon",
    "read_best_track",
    "read_a_deck",
]

logger = logging.getLogger(__name__)

#: Number of positional slots read from every deck file. Generous enough to
#: cover the longest observed record; missing trailing fields become ``NaN``.
MAX_ATCF_FIELDS: int = 45

#: Positional index -> canonical ATCF field name (shared by A- and B-decks).
ATCF_FIELDS: dict[int, str] = {
    0: "BASIN",
    1: "CY",
    2: "YYYYMMDDHH",
    3: "TECHNUM",
    4: "TECH",
    5: "TAU",
    6: "LAT",
    7: "LON",
    8: "VMAX",
    9: "MSLP",
    10: "TY",
    11: "RAD",
    12: "WINDCODE",
    13: "RAD1",
    14: "RAD2",
    15: "RAD3",
    16: "RAD4",
    17: "POUTER",
    18: "ROUTER",
    19: "RMW",
    20: "GUSTS",
    21: "EYE",
    22: "SUBREGION",
    23: "MAXSEAS",
    24: "INITIALS",
    25: "DIR",
    26: "SPEED",
    27: "STORMNAME",
    28: "DEPTH",
}


def parse_lat_lon(s: object) -> float:
    """Parse an ATCF latitude or longitude token into signed decimal degrees.

    ATCF encodes coordinates as tenths of a degree followed by a hemisphere
    letter, e.g. ``'178N'`` -> ``17.8`` and ``'866W'`` -> ``-86.6``.

    Parameters
    ----------
    s : object
        Raw token. ``None``, ``NaN`` and empty strings return ``NaN``.

    Returns
    -------
    float
        Signed decimal degrees, or ``NaN`` if the token cannot be parsed.

    Examples
    --------
    >>> parse_lat_lon("178N")
    17.8
    >>> parse_lat_lon("866W")
    -86.6
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return float("nan")
    text = str(s).strip()
    if not text:
        return float("nan")

    hemisphere = text[-1].upper()
    if hemisphere in ("N", "S", "E", "W"):
        body = text[:-1]
        sign = -1.0 if hemisphere in ("S", "W") else 1.0
    else:
        body = text
        sign = 1.0

    try:
        value = float(body) / 10.0
    except ValueError:
        return float("nan")
    return sign * value


def _read_raw(path: str | Path) -> pd.DataFrame:
    """Read a deck file positionally and strip whitespace from object columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ATCF file not found: {path}")

    df = pd.read_csv(
        path,
        header=None,
        names=list(range(MAX_ATCF_FIELDS)),
        sep=",",
        skipinitialspace=True,
        on_bad_lines="skip",
        index_col=False,
        dtype=object,
    )
    return df.apply(lambda col: col.str.strip() if col.dtype == object else col)


def _numeric(series: pd.Series) -> pd.Series:
    """Coerce a raw ATCF column to float, mapping blanks to ``NaN``."""
    return pd.to_numeric(series.replace("", np.nan), errors="coerce")


def _decode(df: pd.DataFrame) -> pd.DataFrame:
    """Translate positional deck columns into the canonical track schema."""
    out = pd.DataFrame(index=df.index)
    out["time"] = pd.to_datetime(df[2], format="%Y%m%d%H", errors="coerce")
    out["latitude"] = df[6].map(parse_lat_lon)
    out["longitude"] = df[7].map(parse_lat_lon)
    out["max_wind_speed_kt"] = _numeric(df[8])
    out["radius_max_wind_nm"] = _numeric(df[19])
    out["min_pressure_mb"] = _numeric(df[9])
    out["storm_type"] = df[10].replace("", np.nan)
    out["storm_name"] = df[27].replace("", np.nan) if 27 in df.columns else np.nan
    out["basin"] = df[0].replace("", np.nan)
    out["storm_number"] = df[1].replace("", np.nan)
    return out


def read_best_track(path: str | Path) -> pd.DataFrame:
    """Read an ATCF B-deck (best track) file.

    B-decks repeat every fix once per wind radius (``RAD`` = 34/50/64 kt); those
    duplicates are collapsed so that exactly one row per timestamp remains.
    Blank ``VMAX``/``RMW`` fields become ``NaN`` rather than zero, and a zero
    ``RMW`` (the ATCF "no data" sentinel) is also mapped to ``NaN`` so that
    :func:`natcat.tracks.fill_missing_rmw` can fill it.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.dat`` file, e.g. ``data/raw/bal142018.dat``.

    Returns
    -------
    pandas.DataFrame
        Columns ``time``, ``latitude``, ``longitude``, ``max_wind_speed_kt``,
        ``radius_max_wind_nm``, ``min_pressure_mb``, ``storm_type``,
        ``storm_name``, ``basin``, ``storm_number``; sorted by ``time`` with a
        reset index.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.

    Examples
    --------
    >>> track = read_best_track("data/raw/bal142018.dat")  # doctest: +SKIP
    >>> track["time"].is_unique  # doctest: +SKIP
    True
    """
    raw = _read_raw(path)
    if raw.empty:
        return _decode(raw)

    best = raw[raw[4].astype(str).str.upper() == "BEST"]
    if best.empty:
        logger.warning("No BEST records found in %s", path)

    out = _decode(best)
    out = out.dropna(subset=["time", "latitude", "longitude"])
    out = out.sort_values("time", kind="stable")
    out = out.drop_duplicates(subset=["time"], keep="first")

    # ATCF uses 0 as the "missing" sentinel for RMW.
    out.loc[out["radius_max_wind_nm"] <= 0, "radius_max_wind_nm"] = np.nan
    return out.reset_index(drop=True)


def read_a_deck(path: str | Path, techs: Sequence[str] | None = None) -> pd.DataFrame:
    """Read an ATCF A-deck (forecast/aid) file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.dat`` file, e.g. ``data/raw/aal142018.dat``.
    techs : sequence of str, optional
        Objective-aid identifiers to keep (e.g. ``['OFCL', 'HWRF']``). If
        ``None``, every aid in the file is returned.

    Returns
    -------
    pandas.DataFrame
        The best-track schema plus ``tech`` (aid identifier) and ``tau``
        (forecast lead time in hours), sorted by ``time`` then ``tau``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    raw = _read_raw(path)
    if raw.empty:
        out = _decode(raw)
        out["tech"] = pd.Series(dtype=object)
        out["tau"] = pd.Series(dtype=float)
        return out

    if techs is not None:
        wanted = {t.strip().upper() for t in techs}
        raw = raw[raw[4].astype(str).str.upper().isin(wanted)]

    out = _decode(raw)
    out["tech"] = raw[4]
    out["tau"] = _numeric(raw[5])
    out = out.dropna(subset=["time", "latitude", "longitude"])
    out.loc[out["radius_max_wind_nm"] <= 0, "radius_max_wind_nm"] = np.nan
    return out.sort_values(["time", "tau"], kind="stable").reset_index(drop=True)
