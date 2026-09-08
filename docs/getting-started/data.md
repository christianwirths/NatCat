# Data

NatCat reads track data in the NHC Automated Tropical Cyclone Forecasting (ATCF) format, and
optionally exposure data from CLIMADA's LitPop dataset.

## ATCF best-track archive

The National Hurricane Center (NHC) publishes best-track ("B-deck") and forecast/model ("A-deck")
files per storm, one plain-text file per storm per basin per season, under the ATCF naming
convention:

```
b al 14 2018 .dat
│  │  │  │
│  │  │  └─ season year
│  │  └──── storm number within the basin/season (01, 02, ..., 14, ...)
│  └─────── basin (al = North Atlantic, ep = East Pacific, ...)
└────────── deck: b = best track, a = forecast/model
```

i.e. Hurricane Michael (2018), the 14th named Atlantic storm of the season, is `bal142018.dat`.

### Deck A vs deck B

| Deck | Content | Rows per fix | Typical use |
|------|---------|---------------|-------------|
| A (forecast) | Multiple forecast models/techniques (`OFCL`, `AVNO`, `EMXI`, `HWRF`, `CTCX`, `UKX`, ...) and forecast lead times (`TAU`) per synoptic time | One row per (tech, tau, wind-radius quadrant) | Real-time / operational forecast comparison |
| B (best track) | The post-season reanalysis "official" track, `TECH == "BEST"` | One row per (fix time, wind-radius threshold) | Historical calibration, regression testing, the `load_best_track` pipeline |

Both decks repeat each fix once per reported wind-radius threshold (`RAD` = 34/50/64 kt) &#8212;
`natcat.data.atcf.read_best_track` de-duplicates on `time`, keeping the first row per timestamp.
`read_a_deck` keeps every `(tech, tau, wind-radius quadrant)` row as-is; the wind-radius
duplication is instead resolved downstream, in `natcat.tracks.extract_past_trajectory` /
`extract_future_trajectory`, which filter to a single `tech` and de-duplicate on `time` (or
`tau`, then the reconstructed `time`) before interpolation.

### Downloading data

```python
from natcat.data import download_best_track, download_archive

# Single storm
path = download_best_track(2018, "al", 14)

# Bulk archive, e.g. the full offline calibration set for the stochastic model
paths = download_archive(range(1900, 2021), basin="al")
```

`download_best_track` / `download_a_deck` write into `NATCAT_DATA_DIR/raw/` (or `data_dir` if
given), skip the request if the file is already cached (unless `overwrite=True`), and raise
`FileNotFoundError`/`RuntimeError` with the failing URL on an HTTP error rather than failing
silently.

### `NATCAT_DATA_DIR`

All I/O in `natcat.data` and `natcat.tracks` resolves a base data directory: the
`NATCAT_DATA_DIR` environment variable if set, otherwise `<repository root>/data`. Raw ATCF files
live under `<data dir>/raw/`.

```bash
export NATCAT_DATA_DIR=/path/to/data
```

```python
from natcat.tracks import load_best_track

# Uses NATCAT_DATA_DIR/raw/bal142018.dat if present, otherwise downloads it there.
track = load_best_track(2018, "al", "14")
```

## Exposure: LitPop via CLIMADA

Real (non-synthetic) exposure comes from [CLIMADA](https://climada-python.readthedocs.io)'s
LitPop dataset &#8212; gridded asset value disaggregated from nightlight intensity and population.

```python
from natcat.exposure import load_litpop_exposure

exposure = load_litpop_exposure(country="USA", bounds=(24.0, 31.5, -88.0, -79.0))
```

`load_litpop_exposure` imports `climada` lazily, so `import natcat` does not require it to be
installed. CLIMADA caches LitPop rasters on first request (typically under
`~/climada/data/exposures/litpop/`); subsequent calls for the same country reuse the cache.
Requires the `geo` extra (`pip install -e ".[geo]"`).

!!! tip "No CLIMADA? Use a synthetic portfolio"
    `natcat.exposure.synthetic_portfolio` generates a random portfolio inside a bounding box with
    no external dependencies &#8212; the fastest way to get a runnable example, and what the
    quickstart and most of the user guide use.
