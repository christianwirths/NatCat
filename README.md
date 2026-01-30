# NatCat

Natural Catastrophe Modelling Framework
Currently only covering tropical cyclone loss estimation using real-time forecast data and parametric wind field models.

## What this does

This project downloads operational hurricane forecast data (A-deck files from NHC), applies a Rankine vortex wind field model with storm motion asymmetry, and estimates insured losses on exposure portfolios. The damage calculation tracks maximum wind speeds at each location over the storm's passage and applies a sigmoid vulnerability curve to compute damage ratios.

Currently supports Atlantic basin storms with exposure data pulled via CLIMADA's LitPop API.

## Notebooks

**nb1_data_investigation.ipynb** - Explore and visualize raw track data, compare best-track vs forecast models.

**nb2_realtime_TC_loss.ipynb** - Basic loss calculation workflow using best-track data.

**nb3_realworld_assets.ipynb** - Full pipeline: download forecasts, load exposure via CLIMADA, compute time-evolving losses across multiple forecast models (OFCL, AVNO, EMXI, HWRF, CTCX, UKX), and generate animated GIFs showing spatial damage evolution.

## Key capabilities

- Download and parse NHC A-deck forecast files
- Interpolate storm tracks to 5-minute resolution
- Rankine vortex wind field with configurable asymmetry factor
- Heuristic radius of maximum wind when missing from data
- Vulnerability curves for Frame and Masonry construction types
- Cumulative max damage tracking (damage only increases over time)
- Animated visualization of damage footprint evolution

## Setup

```
conda env create -f environment.yml
conda activate natcat
```

## Project structure

```
src/
    download_TC_data.py    - Fetch forecast data from NHC
    preprocess_TC_data.py  - Parse and clean track files
    wind_fields.py         - Rankine vortex model
    vulnerbility.py        - Damage ratio curves
    loss.py                - Portfolio damage calculations
    utils.py               - Track interpolation, bearing/velocity
    visualization.py       - Trajectory animations
```


