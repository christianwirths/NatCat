# Installation

NatCat is installed as an editable package inside a conda environment. Core (always-installed)
dependencies are numpy, pandas, scipy, matplotlib, requests, tqdm, beautifulsoup4, scikit-learn
and global-land-mask; `cartopy` and `climada` are kept as an optional extra since they are heavy
and only needed for mapping and real (LitPop) exposure data.

## Conda environment

```bash
git clone https://github.com/christianwirths/NatCat.git
cd NatCat
conda env create -f environment.yml
conda activate NatCat
```

## Install the package

```bash
pip install -e .
```

This installs `natcat` in editable mode, so changes to the source tree are picked up immediately
without reinstalling.

## Optional extras

The core install is enough for track processing, hazard, vulnerability, loss and financial
calculations on a synthetic portfolio &#8212; including the stochastic genesis model's land/sea
rejection, which uses the core `global-land-mask` and `scikit-learn` dependencies. Two families
of functionality need optional extras:

| Extra    | Installs                                   | Needed for |
|----------|---------------------------------------------|------------|
| `geo`    | `cartopy`, `climada`                        | Real exposure via LitPop (`natcat.exposure.litpop`), basemap plotting (`natcat.plotting.maps`) |
| `docs`   | `mkdocs-material`, `mkdocstrings[python]`, `mkdocs-jupyter` | Building this documentation site |
| `dev`    | `pytest`, `ruff`                            | Running the test suite and linting |

```bash
pip install -e ".[geo]"
pip install -e ".[docs]"
pip install -e ".[dev]"
```

!!! note "climada and cartopy are optional"
    `import natcat` never imports `climada` or `cartopy` at module load time &#8212; those (and the
    core dependencies `global_land_mask`/`scikit-learn`) are imported lazily inside the specific
    functions that need them &#8212; `natcat.exposure.litpop.load_litpop_exposure`,
    `natcat.plotting.maps`, and the land-mask/KNN checks in `natcat.stochastic.genesis`. If you
    only work with
    synthetic portfolios and best-track data, you can skip the `geo` extra entirely.

## Verify the install

```bash
python -c "import natcat; print(natcat.__version__)"
```

## Data directory

NatCat resolves a data directory for ATCF archives and cached downloads, in order of precedence:

1. The `NATCAT_DATA_DIR` environment variable, if set.
2. `<repository root>/data`.

```bash
export NATCAT_DATA_DIR=/path/to/data
```

See [Data](data.md) for the layout NatCat expects inside that directory.
