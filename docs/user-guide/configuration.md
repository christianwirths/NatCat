# Configuration

NatCat keeps configuration minimal and explicit: a resolved data directory plus a handful of
physical/statistical constants, all overridable per call rather than through global state.

## Data directory

`natcat.config` resolves the data directory once, in order of precedence:

1. `NATCAT_DATA_DIR` environment variable.
2. `<repository root>/data`.

```bash
export NATCAT_DATA_DIR=/path/to/data
```

```python
from natcat.config import get_data_dir
print(get_data_dir())
```

Every function that touches disk (`download_best_track`, `load_best_track`,
`SyntheticTCCatalog.fit`, ...) accepts an explicit `data_dir` argument that overrides this
resolution for a single call, which is useful in tests and notebooks that point at a fixture
directory.

## Constants and parameters

Rather than a global settings object, NatCat exposes physical/statistical parameters as explicit,
overridable keyword arguments on the relevant class or function:

| Where | Parameter | Default | Meaning |
|-------|-----------|---------|---------|
| `TropicalCycloneHazard` | `decay_exponent` | `2.0` | Rankine vortex outer-decay exponent |
| `TropicalCycloneHazard` | `asymmetry_factor` | `0.5` | Fraction of translation speed added asymmetrically |
| `WindVulnerability` | `threshold_kt` | `40.0` | Wind speed below which damage ratio is 0 |
| `WindVulnerability` | `CONSTRUCTION_PARAMS` | `{"Frame": {...}, "Masonry": {...}}` | `v_50`/`k` per construction type |
| `SyntheticTCCatalog` | `grid_size` | `2.0` | Degrees per Markov state-grid cell |
| `SyntheticTCCatalog` | `land_decay` / `land_rmw_growth` | `0.92` / `1.02` | Per-timestep over-land decay/growth |
| `ExceedanceProbability` | `tail_quantile` | `0.95` | Empirical/GPD tail split point |

This keeps every run fully reproducible from its call site &#8212; no hidden global mutation &#8212;
and each parameter is documented in context in the [Methodology](../methodology/wind-field.md)
pages rather than only here.

## Reproducibility

Every stochastic component takes a `seed` (or accepts an explicit `numpy.random.Generator`) and
never touches the global `numpy.random` state:

```python
catalog = SyntheticTCCatalog(seed=42)
simulator = LossSimulator(catalog, portfolio, vulnerability, seed=7)
```

Two runs with the same seed and the same historical fit input produce byte-identical synthetic
catalogs and simulated loss arrays.

## Logging

Library code uses the standard `logging` module (`logger = logging.getLogger("natcat...")`)
rather than `print`, so output is controllable by the caller:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Long-running loops (`SyntheticTCCatalog.generate`, `LossSimulator.run`) additionally expose a
`progress: bool` argument that toggles a `tqdm` progress bar, independent of the logging level.
