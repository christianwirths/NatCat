#!/usr/bin/env python
"""Calibrate the value-dependent vulnerability against observed US losses.

Usage::

    python scripts/calibrate.py [--cases .cache/calibration_cases.pkl] [--seed 0]
                                [--maxiter 200] [--reference-year 2018]
                                [--observed path.csv] [--no-download]

Builds one footprint per storm in the observed-loss table (cached to
``--cases``; delete the file to rebuild), fits ``ValueDependentVulnerability``
with differential evolution plus a local polish, and writes

* ``docs/assets/figures/calibration_scatter.png``
* ``docs/assets/figures/calibration_curves.png``
* ``docs/assets/data/calibration_result.json``
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
    build_cases,
    load_cases,
    load_observed_losses,
    normalise_losses,
    save_cases,
)
from natcat.vulnerability import ValueDependentVulnerability, WindVulnerability

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "assets" / "figures"
DATA = ROOT / "docs" / "assets" / "data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("calibrate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", type=Path, default=ROOT / ".cache" / "calibration_cases.pkl")
    parser.add_argument("--observed", type=Path, default=None, help="Custom observed-loss CSV")
    parser.add_argument("--reference-year", type=int, default=2018)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    observed = normalise_losses(
        load_observed_losses(args.observed), reference_year=args.reference_year
    )
    log.info("%d storms in the calibration set", len(observed))

    if args.cases.exists():
        cases = load_cases(args.cases)
        log.info("Loaded %d cached cases from %s", len(cases), args.cases)
    else:
        t0 = time.time()
        cases = build_cases(observed, download=not args.no_download)
        save_cases(cases, args.cases)
        log.info(
            "Built %d cases in %.0f s (cached to %s)", len(cases), time.time() - t0, args.cases
        )
    for case in cases:
        log.info(
            "  %-8s %s: %6d locations, TIV $%.2e, observed $%.2e",
            case.storm_id, case.name, case.n_locations, case.total_value, case.observed_loss,
        )  # fmt: skip

    start = ValueDependentVulnerability()  # threshold 40 kt, v50 105 kt, slope 0, k 0.12
    calibrator = Calibrator(cases, start)
    t0 = time.time()
    result = calibrator.fit(method="global", maxiter=args.maxiter, seed=args.seed)
    log.info("Fit finished in %.0f s", time.time() - t0)
    print(result.summary())
    print(result.table.to_string(index=False, float_format=lambda v: f"{v:,.3g}"))

    plotting.apply_style()
    fig, _ = plotting.plot_calibration(result, title="Modelled vs observed storm loss")
    plotting.add_source_note(
        fig, "Observed: NHC Tropical Cyclone Reports, CPI-normalised; exposure: LitPop (CLIMADA)"
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
        "n_storms": len(cases),
        "method": result.method,
        "n_evaluations": result.n_evaluations,
        "initial_params": result.initial_params,
        "params": result.params,
        "initial_objective": result.initial_objective,
        "objective": result.objective,
        "typical_factor_error_initial": 10 ** (result.initial_objective**0.5),
        "typical_factor_error": 10**result.rmse_log10,
        "storms": result.table.to_dict(orient="records"),
    }
    (DATA / "calibration_result.json").write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", DATA / "calibration_result.json")


if __name__ == "__main__":
    main()
