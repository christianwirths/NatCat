"""Observed storm losses used as calibration targets.

The bundled table ``data/nhc_us_landfall_losses.csv`` lists US landfalls with
the total damage estimate from the NHC Tropical Cyclone Report of each storm
(nominal USD of the loss year). Every row carries a ``damage_driver`` flag
(``wind``, ``mixed``, ``surge``, ``flood``) and an ``include`` flag; storms
dominated by surge or rainfall flooding are excluded by default because the
model is wind-only.

.. warning::
   The bundled values are a curated starting point, not an authoritative loss
   database. Check them against the cited reports before using a calibration
   for anything beyond exploration, and prefer your own (e.g. insured) losses
   via :func:`load_observed_losses` where available.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "REQUIRED_COLUMNS",
    "US_CPI",
    "load_observed_losses",
    "normalise_losses",
]

#: Columns every observed-loss table must carry.
REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"storm_id", "name", "year", "basin", "storm_number", "observed_loss_usd", "loss_year"}
)

#: US CPI-U annual averages (1982-84 = 100), used for the default price
#: normalisation. Source: US Bureau of Labor Statistics.
US_CPI: dict[int, float] = {
    1989: 124.0, 1990: 130.7, 1991: 136.2, 1992: 140.3, 1993: 144.5, 1994: 148.2,
    1995: 152.4, 1996: 156.9, 1997: 160.5, 1998: 163.0, 1999: 166.6, 2000: 172.2,
    2001: 177.1, 2002: 179.9, 2003: 184.0, 2004: 188.9, 2005: 195.3, 2006: 201.6,
    2007: 207.3, 2008: 215.3, 2009: 214.5, 2010: 218.1, 2011: 224.9, 2012: 229.6,
    2013: 233.0, 2014: 236.7, 2015: 237.0, 2016: 240.0, 2017: 245.1, 2018: 251.1,
    2019: 255.7, 2020: 258.8, 2021: 271.0, 2022: 292.7, 2023: 304.7, 2024: 313.7,
}  # fmt: skip


def _bundled_path() -> Path:
    return Path(str(resources.files("natcat.calibration") / "data" / "nhc_us_landfall_losses.csv"))


def load_observed_losses(
    path: str | Path | None = None,
    *,
    included_only: bool = True,
) -> pd.DataFrame:
    """Load an observed-loss table (the bundled NHC table by default).

    Parameters
    ----------
    path : str or pathlib.Path, optional
        CSV with at least :data:`REQUIRED_COLUMNS`. Optional columns:
        ``damage_driver``, ``include`` (bool), ``normalisation_factor``
        (overrides the CPI factor), ``weight`` (objective weight),
        ``source``, ``notes``.
    included_only : bool, default True
        Drop rows whose ``include`` flag is false.

    Returns
    -------
    pandas.DataFrame
        One row per storm, ``storm_number`` zero-padded to two characters.

    Raises
    ------
    KeyError
        If a required column is missing.
    """
    source = Path(path) if path is not None else _bundled_path()
    df = pd.read_csv(source, dtype={"storm_number": str})
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(f"Observed-loss table is missing columns: {sorted(missing)}")

    df = df.copy()
    df["storm_number"] = df["storm_number"].astype(str).str.zfill(2)
    df["basin"] = df["basin"].astype(str).str.lower()
    df["year"] = df["year"].astype(int)
    df["loss_year"] = df["loss_year"].astype(int)
    df["observed_loss_usd"] = df["observed_loss_usd"].astype(float)
    if "include" not in df.columns:
        df["include"] = True
    df["include"] = df["include"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    if "weight" not in df.columns:
        df["weight"] = 1.0
    if included_only:
        df = df[df["include"]]
    return df.reset_index(drop=True)


def normalise_losses(
    df: pd.DataFrame,
    *,
    reference_year: int = 2018,
    method: str = "cpi",
) -> pd.DataFrame:
    """Bring observed losses to the price level of the exposure's reference year.

    Adds ``normalisation_factor`` (unless the table already carries one, which
    is then kept) and ``observed_loss_ref_usd``.

    Parameters
    ----------
    df : pandas.DataFrame
        Table from :func:`load_observed_losses`. Never mutated.
    reference_year : int, default 2018
        Year the exposure values refer to (LitPop's default reference year).
    method : {"cpi", "none"}, default "cpi"
        ``"cpi"`` scales by the ratio of US CPI-U annual averages; ``"none"``
        uses a factor of one. Neither accounts for exposure growth, so old
        storms are under-normalised relative to today's building stock.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        For an unknown method or a year outside :data:`US_CPI`.
    """
    out = df.copy()
    if "normalisation_factor" in out.columns and out["normalisation_factor"].notna().all():
        factor = out["normalisation_factor"].astype(float).to_numpy()
    elif method == "none":
        factor = np.ones(len(out))
    elif method == "cpi":
        try:
            ref = US_CPI[int(reference_year)]
            factor = np.array([ref / US_CPI[int(y)] for y in out["loss_year"]])
        except KeyError as exc:
            raise ValueError(
                f"No CPI value for year {exc}; extend US_CPI or pass a normalisation_factor"
            ) from exc
    else:
        raise ValueError(f"Unknown normalisation method {method!r}; use 'cpi' or 'none'")
    out["normalisation_factor"] = factor
    out["observed_loss_ref_usd"] = out["observed_loss_usd"].to_numpy() * factor
    return out
