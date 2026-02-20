# NatCat

Natural Catastrophe Modelling Framework. This is a personal project to implement clean code standards and to familiarize myself with the utilization of weather and climate data for real world applications.
Currently only covering tropical cyclone loss estimation using real-time forecast data and parametric wind field models.

## What this does

This project downloads operational hurricane forecast data (A-deck files from NHC), applies a Rankine vortex wind field model with storm motion asymmetry, and estimates insured losses on exposure portfolios. The damage calculation tracks maximum wind speeds at each location over the storm's passage and applies a sigmoid vulnerability curve to compute damage ratios.

Currently supports Atlantic basin storms with exposure data pulled via CLIMADA's LitPop API.

## Notebooks

**nb1_data_investigation.ipynb** - Explore and visualize raw track data, compare best-track vs forecast models.

**nb2_realtime_TC_loss.ipynb** - Basic loss calculation workflow using best-track data.

**nb3_realworld_assets.ipynb** - Full pipeline: download forecasts, load exposure via CLIMADA, compute time-evolving losses across multiple forecast models (OFCL, AVNO, EMXI, HWRF, CTCX, UKX), and generate animated GIFs showing spatial damage evolution.

**nb4_refactored_pipeline.ipynb** - Refactored pipeline showing the modularized `src/` workflow and an end-to-end loss example using the current codebase.

**nb5_probabilistic_storms.ipynb** - Generating probabilistic tropical cyclone tracks from historical data using an emperical MCMC.

Note on older notebooks

- `nb1_data_investigation.ipynb`, `nb2_realtime_TC_loss.ipynb`, `nb3_realworld_assets.ipynb`: use the deprecated A/B-deck codepath and related helpers; their original code has been moved to `src/deprecated/` but descriptions remain here for reference.

## Next features:
- Synthetic storm tracks to make probabilistic assessments
- Full parameter uncertainty assessment
- Integration of automated unit tests using pytest
