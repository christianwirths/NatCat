"""Tests for natcat.calibration and the value-dependent vulnerability model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from natcat.calibration import (
    Calibrator,
    StormCase,
    load_cases,
    load_observed_losses,
    normalise_losses,
    save_cases,
    storm_region,
)
from natcat.loss import LossCalculator
from natcat.vulnerability import ValueDependentVulnerability, WindVulnerability


# -- ValueDependentVulnerability ---------------------------------------------
def test_zero_slope_matches_wind_vulnerability():
    winds = np.linspace(0, 180, 50)
    value_model = ValueDependentVulnerability(v50_ref=100.0, k=0.12, v50_slope=0.0)
    frame = WindVulnerability("Frame")
    assert np.allclose(
        value_model.damage_ratio(winds, tiv=np.full(50, 3e5)), frame.damage_ratio(winds)
    )


def test_positive_slope_makes_valuable_tiles_sturdier():
    model = ValueDependentVulnerability(v50_ref=100.0, v50_slope=10.0)
    cheap, dear = model.damage_ratio([110.0, 110.0], tiv=[1e5, 1e9])
    assert cheap > dear


def test_v50_is_clipped_to_bounds():
    model = ValueDependentVulnerability(v50_ref=100.0, v50_slope=100.0, v50_bounds=(60.0, 140.0))
    assert model.v50([1.0, 1e12]).tolist() == [60.0, 140.0]


def test_threshold_zeroes_damage_below_it():
    model = ValueDependentVulnerability(threshold_kt=40.0)
    assert model.damage_ratio([39.9, 40.0], tiv=[1e6, 1e6])[0] == 0.0
    assert model.damage_ratio([39.9, 40.0], tiv=[1e6, 1e6])[1] > 0.0


def test_params_round_trip_and_validation():
    model = ValueDependentVulnerability(k=0.2)
    updated = model.with_params(v50_ref=120.0, k=0.1)
    assert updated.params == {
        "threshold_kt": 40.0, "v50_ref": 120.0, "v50_slope": 0.0, "k": 0.1, "scale": 1.0
    }  # fmt: skip
    assert ValueDependentVulnerability(scale=0.5).damage_ratio(200.0, tiv=1e7) <= 0.5
    with pytest.raises(ValueError, match="scale"):
        ValueDependentVulnerability(scale=1.5)
    assert model.params["k"] == 0.2  # original untouched
    with pytest.raises(ValueError, match="Unknown parameter"):
        model.with_params(nope=1.0)
    with pytest.raises(ValueError):
        ValueDependentVulnerability(k=0.0)


def test_wind_vulnerability_flat_params():
    model = WindVulnerability("Frame")
    assert model.params["Frame_v_50"] == 100.0
    updated = model.with_params(Frame_v_50=90.0, threshold_kt=35.0)
    assert updated.params_table["Frame"]["v_50"] == 90.0
    assert updated.threshold_kt == 35.0
    assert model.params_table["Frame"]["v_50"] == 100.0
    with pytest.raises(ValueError, match="Unknown parameter"):
        model.with_params(Brick_v_50=1.0)


def test_loss_calculator_passes_tiv(tiny_processed):
    from natcat.exposure import synthetic_portfolio
    from natcat.hazards import TropicalCycloneHazard

    portfolio = synthetic_portfolio(200, bounds=(25.0, 28.0, -80.6, -79.4), seed=7)
    hazard = TropicalCycloneHazard(tiny_processed)
    steep = ValueDependentVulnerability(v50_ref=100.0, v50_slope=40.0, tiv_ref=1e4)
    flat = ValueDependentVulnerability(v50_ref=100.0, v50_slope=0.0, tiv_ref=1e4)
    loss_steep = LossCalculator(hazard, steep).compute(portfolio)["loss"].sum()
    loss_flat = LossCalculator(hazard, flat).compute(portfolio)["loss"].sum()
    assert loss_flat > 0
    assert loss_steep < loss_flat  # every tile is worth more than tiv_ref, so sturdier


# -- observed losses ----------------------------------------------------------
def test_bundled_table_loads_and_filters():
    everything = load_observed_losses(included_only=False)
    included = load_observed_losses()
    assert {"storm_id", "observed_loss_usd", "loss_year", "include"} <= set(everything.columns)
    assert len(included) < len(everything)
    assert included["include"].all()
    assert (included["storm_number"].str.len() == 2).all()
    assert "AL142018" in set(included["storm_id"])
    assert "AL092017" not in set(included["storm_id"])  # Harvey: flood, excluded


def test_normalise_losses_inflates_old_storms():
    df = load_observed_losses()
    out = normalise_losses(df, reference_year=2018)  # default: nominal GDP
    cpi = normalise_losses(df, reference_year=2018, method="cpi")
    andrew = out[out["storm_id"] == "AL041992"].iloc[0]
    michael = out[out["storm_id"] == "AL142018"].iloc[0]
    assert andrew["normalisation_factor"] == pytest.approx(20.66 / 6.52, rel=1e-3)
    assert cpi[cpi["storm_id"] == "AL041992"].iloc[0]["normalisation_factor"] > 1.5
    assert andrew["normalisation_factor"] > 1.5
    assert michael["normalisation_factor"] == pytest.approx(1.0)
    assert andrew["observed_loss_ref_usd"] == pytest.approx(
        andrew["observed_loss_usd"] * andrew["normalisation_factor"]
    )
    assert "normalisation_factor" not in df.columns  # input untouched


def test_normalise_losses_explicit_factor_and_errors():
    df = pd.DataFrame(
        {
            "storm_id": ["X"], "name": ["x"], "year": [2000], "basin": ["al"],
            "storm_number": ["01"], "observed_loss_usd": [2.0], "loss_year": [2000],
            "normalisation_factor": [3.0],
        }
    )  # fmt: skip
    assert normalise_losses(df)["observed_loss_ref_usd"].iloc[0] == 6.0
    df = df.drop(columns="normalisation_factor").assign(loss_year=[1850])
    with pytest.raises(ValueError, match="GDP"):
        normalise_losses(df)
    with pytest.raises(ValueError, match="method"):
        normalise_losses(df.assign(loss_year=[2000]), method="magic")


def test_load_observed_losses_requires_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("storm_id,name\nA,B\n")
    with pytest.raises(KeyError, match="missing columns"):
        load_observed_losses(path)


# -- cases ---------------------------------------------------------------------
def test_storm_region_clips_and_pads():
    track = pd.DataFrame({"latitude": [20.0, 26.0, 30.0], "longitude": [-70.0, -80.0, -85.0]})
    lat_min, lat_max, lon_min, lon_max = storm_region(track, margin_deg=1.0)
    assert (lat_min, lat_max) == (25.0, 31.0)
    assert (lon_min, lon_max) == (-86.0, -79.0)
    with pytest.raises(ValueError):
        storm_region(track.iloc[:1])


def _synthetic_cases(true_model, rng, n_storms=6, n_locations=400):
    """Storms hitting exposure of different value mixes, so the slope is identifiable."""
    cases = []
    for i in range(n_storms):
        intensity = rng.uniform(30.0, 150.0, n_locations)
        low = 4.5 + 4.0 * i / max(n_storms - 1, 1)  # storm i hits tiles worth 10**low..10**(low+2)
        tiv = 10 ** rng.uniform(low, low + 2.0, n_locations)
        case = StormCase(f"S{i}", f"storm{i}", 1.0, intensity, tiv)
        case.observed_loss = case.loss(true_model)
        cases.append(case)
    return cases


def test_case_loss_and_pickle_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    cases = _synthetic_cases(ValueDependentVulnerability(), rng, n_storms=2)
    save_cases(cases, tmp_path / "cases.pkl")
    loaded = load_cases(tmp_path / "cases.pkl")
    assert [c.storm_id for c in loaded] == ["S0", "S1"]
    assert loaded[0].loss(ValueDependentVulnerability()) == pytest.approx(cases[0].observed_loss)
    assert loaded[0].n_locations == 400


# -- calibrator ------------------------------------------------------------------
def test_calibrator_objective_is_zero_at_truth_and_recovers_parameters():
    rng = np.random.default_rng(1)
    truth = ValueDependentVulnerability(threshold_kt=45.0, v50_ref=110.0, v50_slope=8.0, k=0.10)
    cases = _synthetic_cases(truth, rng)
    start = ValueDependentVulnerability(threshold_kt=40.0, v50_ref=95.0, v50_slope=0.0, k=0.14)

    calibrator = Calibrator(cases, start)
    assert calibrator.objective(np.array(list(truth.params.values()))) == pytest.approx(0.0)

    result = calibrator.fit(method="global", maxiter=60, seed=0)
    assert result.objective < result.initial_objective
    assert result.objective < 1e-4
    assert result.params["v50_ref"] == pytest.approx(110.0, abs=5.0)
    assert result.params["v50_slope"] == pytest.approx(8.0, abs=2.0)
    assert result.table["modelled"].to_numpy() == pytest.approx(
        result.table["observed"].to_numpy(), rel=0.05
    )
    assert "objective" in result.summary()


def test_calibrator_global_respects_bounds_and_seed():
    rng = np.random.default_rng(2)
    truth = ValueDependentVulnerability(v50_ref=120.0, v50_slope=5.0)
    cases = _synthetic_cases(truth, rng, n_storms=4, n_locations=200)
    calibrator = Calibrator(
        cases, ValueDependentVulnerability(), param_names=("v50_ref", "v50_slope"),
        bounds={"v50_ref": (90.0, 150.0)},
    )  # fmt: skip
    a = calibrator.fit(method="global", maxiter=30, seed=0, polish=False)
    b = calibrator.fit(method="global", maxiter=30, seed=0, polish=False)
    assert a.params == b.params
    assert 90.0 <= a.params["v50_ref"] <= 150.0
    assert a.params["threshold_kt"] == 40.0  # not optimised
    assert a.params["scale"] == 1.0  # not optimised
    assert a.objective <= a.initial_objective


def test_calibrator_validation():
    rng = np.random.default_rng(3)
    cases = _synthetic_cases(ValueDependentVulnerability(), rng, n_storms=1)
    with pytest.raises(ValueError, match="at least one"):
        Calibrator([], ValueDependentVulnerability())
    with pytest.raises(ValueError, match="Unknown parameter"):
        Calibrator(cases, ValueDependentVulnerability(), param_names=("nope",))
    cases[0].observed_loss = 0.0
    with pytest.raises(ValueError, match="positive observed"):
        Calibrator(cases, ValueDependentVulnerability())
    with pytest.raises(ValueError, match="method"):
        Calibrator(
            _synthetic_cases(ValueDependentVulnerability(), rng, 1), ValueDependentVulnerability()
        ).fit(method="nope")


# -- hazard grid ------------------------------------------------------------------
def test_hazard_grid_recovers_true_exponent(tiny_processed):
    from functools import partial

    from natcat.calibration import CaseInput, calibrate_hazard_grid, cases_from_inputs
    from natcat.exposure import synthetic_portfolio
    from natcat.hazards import TropicalCycloneHazard

    truth_vuln = ValueDependentVulnerability(v50_ref=100.0, v50_slope=5.0)
    truth_hazard = partial(TropicalCycloneHazard, decay_exponent=1.0, asymmetry_factor=0.5)
    inputs = []
    for i in range(3):
        lon = -80.0 + 0.3 * (i - 1)
        portfolio = synthetic_portfolio(150, bounds=(25.0, 28.0, lon - 0.6, lon + 0.6), seed=i)
        # each "storm" is the same track shifted in longitude, hitting a different portfolio
        track = tiny_processed.assign(longitude=tiny_processed["longitude"] + 0.1 * i)
        inputs.append(CaseInput(f"S{i}", f"s{i}", 1.0, track, portfolio, (25, 28, -81, -79)))
    for item, case in zip(inputs, cases_from_inputs(inputs, truth_hazard), strict=True):
        item.observed_loss = case.loss(truth_vuln)
        assert item.observed_loss > 0

    grid = calibrate_hazard_grid(
        inputs,
        ValueDependentVulnerability(),
        decay_exponents=(1.0, 2.0),
        asymmetry_factors=(0.5,),
        param_names=("v50_ref", "v50_slope"),
        method="local",
        maxiter=400,
        progress=False,
    )
    assert set(grid.results) == {(1.0, 0.5), (2.0, 0.5)}
    assert grid.best_hazard == {"decay_exponent": 1.0, "asymmetry_factor": 0.5}
    assert grid.best.objective < grid.results[(2.0, 0.5)].objective
    assert len(grid.table) == 2 and "factor_error" in grid.table.columns
    assert "Best: decay_exponent=1" in grid.summary()
    assert grid.hazard_factory().keywords == {"decay_exponent": 1.0, "asymmetry_factor": 0.5}
    with pytest.raises(ValueError, match="at least one"):
        calibrate_hazard_grid([], ValueDependentVulnerability(), progress=False)


def test_inputs_round_trip_and_build_cases_reuse(tmp_path, tiny_processed):
    from natcat.calibration import CaseInput, build_cases, load_inputs, save_inputs
    from natcat.exposure import synthetic_portfolio

    portfolio = synthetic_portfolio(50, bounds=(25.0, 28.0, -80.6, -79.4), seed=1)
    inputs = [CaseInput("S0", "s0", 5.0, tiny_processed, portfolio, (25, 28, -81, -79))]
    save_inputs(inputs, tmp_path / "inputs.pkl")
    loaded = load_inputs(tmp_path / "inputs.pkl")
    cases = build_cases(pd.DataFrame(), inputs=loaded)
    assert cases[0].storm_id == "S0" and cases[0].n_locations == 50
    assert cases[0].observed_loss == 5.0


def test_implied_decay_exponents_from_bdeck(tmp_path):
    from natcat.calibration import implied_decay_exponents

    # vmax 100 kt, RMW 20 nm, 50 kt radius 80 nm in every quadrant -> n = ln(2) / ln(4) = 0.5
    line = "AL, 14, 2018101012,   , BEST,   0, 300N,  850W, 100,  950, HU,  50, NEQ,   80,   80,   80,   80, 1010,  200,  20,\n"
    weak = line.replace(" 100,  950, HU,  50", "  50,  990, TS,  34")
    path = tmp_path / "bal142018.dat"
    path.write_text(line + weak)
    out = implied_decay_exponents(path)
    assert len(out) == 1  # the tropical-storm fix is below min_wind_kt
    assert out["exponent"].iloc[0] == pytest.approx(0.5)
    assert out["storm_id"].iloc[0] == "AL142018"
    assert implied_decay_exponents(tmp_path)["exponent"].tolist() == out["exponent"].tolist()
