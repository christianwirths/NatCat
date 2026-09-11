#!/usr/bin/env python
"""Calibrate the value-dependent vulnerability against observed US losses.

Usage::

    python scripts/calibrate.py [--inputs .cache/calibration_inputs.pkl] [--seed 0]
                                [--maxiter 200] [--reference-year 2018]
                                [--normalisation gdp|cpi|none] [--observed path.csv]
                                [--no-hazard-grid] [--no-download]

Loads track and regional LitPop exposure for every storm in the observed-loss
table (cached to ``--inputs``; delete the file to rebuild), then

1. runs the hazard grid (Rankine decay exponent x motion asymmetry factor),
   calibrating ``ValueDependentVulnerability`` with differential evolution
   plus a local polish at every grid point, and
2. reports the best combination.

Writes ``docs/assets/figures/calibration_{scatter,curves,hazard_grid}.png``
and ``docs/assets/data/calibration_result.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from natcat import plotting
from natcat.calibration import (
    Calibrator,
    calibrate_hazard_grid,
    cases_from_inputs,
    load_inputs,
    load_observed_losses,
    normalise_losses,
    prepare_inputs,
    save_inputs,
)
from natcat.vulnerability import ValueDependentVulnerability, WindVulnerability

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "assets" / "figures"
DATA = ROOT / "docs" / "assets" / "data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("calibrate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", type=Path, default=ROOT / ".cache" / "calibration_inputs.pkl")
    parser.add_argument("--observed", type=Path, default=None, help="Custom observed-loss CSV")
    parser.add_argument("--reference-year", type=int, default=2018)
    parser.add_argument("--normalisation", default="gdp", choices=("gdp", "cpi", "none"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--no-hazard-grid", action="store_true", help="Default hazard only")
    parser.add_argument(
        "--exponents", type=float, nargs="+", default=(0.5, 0.75, 1.0, 1.5, 2.0),
        help="Decay exponents of the hazard grid",
    )  # fmt: skip
    parser.add_argument(
        "--asymmetries", type=float, nargs="+", default=(0.3, 0.5, 0.7),
        help="Asymmetry factors of the hazard grid",
    )  # fmt: skip
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    observed = normalise_losses(
        load_observed_losses(args.observed),
        reference_year=args.reference_year,
        method=args.normalisation,
    )
    log.info("%d storms in the calibration set", len(observed))

    if args.inputs.exists():
        inputs = load_inputs(args.inputs)
        log.info("Loaded %d cached inputs from %s", len(inputs), args.inputs)
        # observed losses may have been re-normalised since the cache was written
        targets = dict(zip(observed["storm_id"], observed["observed_loss_ref_usd"], strict=True))
        for item in inputs:
            item.observed_loss = float(targets[item.storm_id])
    else:
        t0 = time.time()
        inputs = prepare_inputs(observed, download=not args.no_download)
        save_inputs(inputs, args.inputs)
        log.info(
            "Prepared %d inputs in %.0f s (cached to %s)",
            len(inputs),
            time.time() - t0,
            args.inputs,
        )
    for item in inputs:
        log.info(
            "  %-8s %-8s %6d locations, TIV $%.2e, observed $%.2e",
            item.storm_id, item.name, len(item.portfolio), item.portfolio["tiv"].sum(),
            item.observed_loss,
        )  # fmt: skip

    start = ValueDependentVulnerability()
    hazard_payload: dict = {}
    if args.no_hazard_grid:
        cases = cases_from_inputs(inputs)
        result = Calibrator(cases, start).fit(method="global", maxiter=args.maxiter, seed=args.seed)
        hazard = {"decay_exponent": 2.0, "asymmetry_factor": 0.5}
    else:
        t0 = time.time()
        grid = calibrate_hazard_grid(
            inputs, start, decay_exponents=args.exponents, asymmetry_factors=args.asymmetries,
            maxiter=args.maxiter, seed=args.seed,
        )  # fmt: skip
        log.info("Hazard grid finished in %.0f s", time.time() - t0)
        print(grid.summary())
        result, hazard = grid.best, grid.best_hazard
        fig, _ = plotting.plot_hazard_grid(grid, title="Calibrated error over hazard parameters")
        plotting.save(fig, FIGURES / "calibration_hazard_grid.png")
        hazard_payload = {"hazard_grid": grid.table.to_dict(orient="records")}

    print(result.summary())
    print(result.table.to_string(index=False, float_format=lambda v: f"{v:,.3g}"))

    plotting.apply_style()
    fig, _ = plotting.plot_calibration(result, title="Modelled vs observed storm loss")
    plotting.add_source_note(
        fig,
        f"Observed: NHC Tropical Cyclone Reports, {args.normalisation.upper()}-normalised to "
        f"{args.reference_year}; exposure: LitPop (CLIMADA); hazard: exponent "
        f"{hazard['decay_exponent']:g}, asymmetry {hazard['asymmetry_factor']:g}",
    )
    plotting.save(fig, FIGURES / "calibration_scatter.png")

    fig, _ = plotting.plot_value_dependent_curves(
        result.model,
        reference=WindVulnerability("Frame"),
        title="Calibrated vulnerability by tile value",
    )
    plotting.save(fig, FIGURES / "calibration_curves.png")

    DATA.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_year": args.reference_year,
        "normalisation": args.normalisation,
        "n_storms": len(inputs),
        "hazard": hazard,
        "method": result.method,
        "n_evaluations": result.n_evaluations,
        "initial_params": result.initial_params,
        "params": result.params,
        "initial_objective": result.initial_objective,
        "objective": result.objective,
        "typical_factor_error_initial": 10 ** (result.initial_objective**0.5),
        "typical_factor_error": 10**result.rmse_log10,
        "storms": result.table.to_dict(orient="records"),
        **hazard_payload,
    }
    (DATA / "calibration_result.json").write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", DATA / "calibration_result.json")


if __name__ == "__main__":
    main()
