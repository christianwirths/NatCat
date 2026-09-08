# Contributing

NatCat is a personal, actively developed project; contributions and issue reports are welcome via
GitHub.

## Development setup

```bash
git clone https://github.com/christianwirths/NatCat.git
cd NatCat
conda env create -f environment.yml
conda activate NatCat
pip install -e ".[dev,geo,docs]"
```

## Code standards

- Python 3.10+, type hints on all public functions.
- Docstrings in **numpy style** (`Parameters` / `Returns` / `Raises` / `Examples`) &#8212;
  `mkdocstrings` renders these directly into the [API reference](api/data.md).
- Every module has a module docstring and an explicit `__all__`; no wildcard imports.
- Pure functions: never mutate an input DataFrame &#8212; copy first.
- No `print` in library code; use `logging.getLogger(__name__)`. Long-running loops accept a
  `progress: bool` argument controlling a `tqdm` bar.
- Heavy dependencies (`climada`, `cartopy` &#8212; the optional `geo` extra &#8212; plus the core
  `global_land_mask` and `scikit-learn`) are imported lazily inside the functions that need them,
  so `import natcat` never pays their import cost and never requires `climada`/`cartopy` to be
  installed.
- Any randomness takes a `seed` or an explicit `numpy.random.Generator`; never call the global
  `numpy.random` API from library code.

## Linting

```bash
ruff check natcat tests
```

Configuration lives in `pyproject.toml`: line length 100, rule sets `E, F, I, W, UP, B, NPY`.

## Tests

```bash
pytest
```

Tests must pass offline. Tests that require the historical archive
(`data/raw/bal142018.dat`, the full `data/raw/bal*.dat` set) skip gracefully via `pytest.skip` when
the fixture data is absent; tests that require network access are marked and skipped by default.

## Building the documentation

```bash
mkdocs serve
```

opens a live-reloading local copy of this site at `http://127.0.0.1:8000`. To build the static
site (as CI does):

```bash
mkdocs build --strict
```

`--strict` treats warnings (broken internal links, missing nav targets) as errors. Missing
`natcat.*` modules or missing example notebooks/figures are expected only while a given piece of
the refactor is still in progress; a green `--strict` build is exit-criteria once `natcat/`,
`notebooks/` and `docs/assets/` are all in place.

## Pull requests

- Keep changes focused; prefer several small PRs over one large one.
- Add or update tests for any behavior change.
- Run `ruff check` and `pytest` locally before opening a PR.
- Update `docs/changelog.md` for any user-facing change.
