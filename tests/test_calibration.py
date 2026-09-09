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


def test_normalise_losses_cpi_inflates_old_storms():
    df = load_observed_losses()
    out = normalise_losses(df, reference_year=2018)
    andrew = out[out["storm_id"] == "AL041992"].iloc[0]
    michael = out[out["storm_id"] == "AL142018"].iloc[0]
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
    with pytest.raises(ValueError, match="CPI"):
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
