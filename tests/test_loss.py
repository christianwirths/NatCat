"""Tests for natcat.loss and natcat.exposure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from natcat.exposure import synthetic_portfolio
from natcat.hazards import TropicalCycloneHazard
from natcat.loss import LossCalculator, SimulationResults
from natcat.vulnerability import WindVulnerability


@pytest.fixture
def calculator(michael_track):
    return LossCalculator(TropicalCycloneHazard(michael_track), WindVulnerability())


# -- exposure ---------------------------------------------------------------
def test_synthetic_portfolio_columns():
    portfolio = synthetic_portfolio(50, seed=1)
    assert list(portfolio.columns) == [
        "location_id",
        "latitude",
        "longitude",
        "tiv",
        "construction",
    ]
    assert len(portfolio) == 50


def test_synthetic_portfolio_is_deterministic():
    pd.testing.assert_frame_equal(synthetic_portfolio(20, seed=3), synthetic_portfolio(20, seed=3))


def test_synthetic_portfolio_respects_bounds():
    bounds = (10.0, 12.0, -60.0, -58.0)
    portfolio = synthetic_portfolio(200, bounds=bounds, seed=2)
    assert portfolio["latitude"].between(bounds[0], bounds[1]).all()
    assert portfolio["longitude"].between(bounds[2], bounds[3]).all()


def test_synthetic_portfolio_rejects_negative_n():
    with pytest.raises(ValueError):
        synthetic_portfolio(-1)


# -- LossCalculator ---------------------------------------------------------
def test_compute_adds_columns(calculator, portfolio):
    results = calculator.compute(portfolio)
    assert {"intensity", "damage_ratio", "loss"}.issubset(results.columns)
    assert len(results) == len(portfolio)


def test_compute_does_not_mutate_portfolio(calculator, portfolio):
    before = portfolio.copy(deep=True)
    calculator.compute(portfolio)
    pd.testing.assert_frame_equal(portfolio, before)


def test_loss_equals_damage_ratio_times_tiv(calculator, portfolio):
    results = calculator.compute(portfolio)
    assert np.allclose(results["loss"], results["damage_ratio"] * results["tiv"])


def test_total_loss_matches_sum(calculator, portfolio):
    results = calculator.compute(portfolio)
    assert calculator.total_loss == pytest.approx(results["loss"].sum())
    assert calculator.total_loss > 0.0


def test_total_loss_before_compute_raises(calculator):
    with pytest.raises(RuntimeError, match="compute"):
        _ = calculator.total_loss


def test_compute_requires_columns(calculator):
    with pytest.raises(KeyError):
        calculator.compute(pd.DataFrame({"latitude": [25.0]}))


def test_calculate_portfolio_loss_is_deprecated_alias(calculator, portfolio):
    with pytest.warns(DeprecationWarning):
        legacy = calculator.calculate_portfolio_loss(portfolio)
    pd.testing.assert_frame_equal(legacy, calculator.compute(portfolio))


def test_compute_history_is_long_format(calculator, portfolio, michael_track):
    times = pd.date_range(michael_track["time"].min(), michael_track["time"].max(), freq="12h")
    history = calculator.compute_history(portfolio, times)
    assert len(history) == len(times) * len(portfolio)
    assert {"time", "intensity", "damage_ratio", "loss"}.issubset(history.columns)


def test_compute_history_is_monotone(calculator, portfolio, michael_track):
    times = pd.date_range(michael_track["time"].min(), michael_track["time"].max(), freq="12h")
    totals = calculator.compute_history(portfolio, times).groupby("time")["loss"].sum()
    assert totals.is_monotonic_increasing


def test_compute_history_final_total_matches_compute(calculator, portfolio, michael_track):
    times = pd.date_range(michael_track["time"].min(), michael_track["time"].max(), freq="12h")
    history = calculator.compute_history(portfolio, times)
    final = history[history["time"] == history["time"].max()]["loss"].sum()
    assert final == pytest.approx(calculator.compute(portfolio)["loss"].sum())


def test_compute_history_sorts_unsorted_times(calculator, portfolio, michael_track):
    times = pd.date_range(michael_track["time"].min(), michael_track["time"].max(), freq="12h")
    shuffled = pd.Series(times).sample(frac=1.0, random_state=0)
    history = calculator.compute_history(portfolio, shuffled)
    assert history["time"].is_monotonic_increasing


def test_compute_history_rejects_empty_times(calculator, portfolio):
    with pytest.raises(ValueError, match="timestamp"):
        calculator.compute_history(portfolio, [])


# -- SimulationResults ------------------------------------------------------
def test_simulation_results_aal_and_frame():
    results = SimulationResults(
        annual_losses=np.array([0.0, 10.0, 20.0, 30.0]),
        max_event_losses=np.array([0.0, 10.0, 15.0, 30.0]),
        events=pd.DataFrame(columns=["year", "storm_id", "loss"]),
        n_years=4,
    )
    assert results.aal == pytest.approx(15.0)
    frame = results.to_frame()
    assert list(frame.columns) == ["year", "annual_loss", "max_event_loss"]
    assert len(frame) == 4


def test_simulation_results_empty_aal():
    results = SimulationResults(
        annual_losses=np.zeros(0),
        max_event_losses=np.zeros(0),
        events=pd.DataFrame(columns=["year", "storm_id", "loss"]),
        n_years=0,
    )
    assert results.aal == 0.0
