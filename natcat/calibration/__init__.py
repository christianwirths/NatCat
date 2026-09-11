"""Calibrate vulnerability parameters against observed storm losses.

Workflow::

    observed = normalise_losses(load_observed_losses(), reference_year=2018)
    cases = build_cases(observed)                      # footprint per storm, once
    result = Calibrator(cases, ValueDependentVulnerability()).fit(seed=0)
    print(result.summary())
"""

from .calibrator import CalibrationResult, Calibrator
from .cases import (
    CONUS_BOUNDS,
    CaseInput,
    StormCase,
    build_cases,
    cases_from_inputs,
    load_cases,
    load_inputs,
    prepare_inputs,
    save_cases,
    save_inputs,
    storm_region,
)
from .hazard import HazardGridResult, calibrate_hazard_grid, implied_decay_exponents
from .observed import (
    REQUIRED_COLUMNS,
    US_CPI,
    US_GDP_NOMINAL_TN,
    load_observed_losses,
    normalise_losses,
)

__all__ = [
    "CONUS_BOUNDS",
    "REQUIRED_COLUMNS",
    "US_CPI",
    "US_GDP_NOMINAL_TN",
    "CalibrationResult",
    "Calibrator",
    "CaseInput",
    "HazardGridResult",
    "StormCase",
    "build_cases",
    "calibrate_hazard_grid",
    "implied_decay_exponents",
    "cases_from_inputs",
    "load_cases",
    "load_inputs",
    "load_observed_losses",
    "normalise_losses",
    "prepare_inputs",
    "save_cases",
    "save_inputs",
    "storm_region",
]
