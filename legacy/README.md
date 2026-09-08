# Legacy code

This directory holds the first iteration of the project, superseded by the
installable `natcat` package at the repository root. It is kept for reference
only: it is **not maintained**, not tested, not covered by the documentation
site, and it will not be updated as `natcat` evolves.

- `legacy/src/` — the original flat modules (`preprocess_TC_data.py`,
  `wind_fields.py`, `loss.py`, ...), previously `src/deprecated/`.
- `legacy/notebooks/` — the exploratory notebooks `nb1`–`nb3`.

These files import each other as flat top-level modules and expect
`sys.path.append('../src')`-style setup rather than a package import, so they
will not run against the current layout without adjusting `sys.path`. For
anything new, use `natcat` instead.
