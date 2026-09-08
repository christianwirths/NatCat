"""Tests for natcat.financial.

The curves are exercised on a synthetic lognormal loss sample, so these tests
need no data files and run offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from natcat.financial import EPCurve, EventResult, ExceedanceProbability, YearResult
from natcat.loss import SimulationResults

N_YEARS = 2000
TAIL_QUANTILE = 0.95


@pytest.fixture(scope="module")
def sample() -> tuple[np.ndarray, np.ndarray]:
    """Annual and max-event losses from a lognormal model."""
    rng = np.random.default_rng(0)
    annual = rng.lognormal(mean=16.0, sigma=1.4, size=N_YEARS)
    max_event = annual * rng.uniform(0.4, 1.0, size=N_YEARS)
    return annual, max_event


@pytest.fixture(scope="module")
def ep(sample) -> ExceedanceProbability:
    annual, max_event = sample
    return ExceedanceProbability(annual, max_event, tail_quantile=TAIL_QUANTILE)


@pytest.fixture(scope="module")
def grid(sample) -> np.ndarray:
    annual, _ = sample
    return np.linspace(0.0, float(annual.max()) * 1.5, 400)


# -- construction -----------------------------------------------------------
def test_epcurve_is_an_alias():
    assert EPCurve is ExceedanceProbability


def test_rejects_empty_sample():
    with pytest.raises(ValueError, match="empty"):
        ExceedanceProbability([])


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        ExceedanceProbability([1.0, 2.0], [1.0])


def test_rejects_bad_tail_quantile():
    with pytest.raises(ValueError, match="tail_quantile"):
        ExceedanceProbability([1.0, 2.0], tail_quantile=1.5)


def test_from_simulation():
    import pandas as pd

    results = SimulationResults(
        annual_losses=np.array([1.0, 2.0, 3.0, 4.0]),
        max_event_losses=np.array([1.0, 1.5, 3.0, 2.0]),
        events=pd.DataFrame(columns=["year", "storm_id", "loss"]),
        n_years=4,
    )
    curve = ExceedanceProbability.from_simulation(results)
    assert curve.n_years == 4
    assert curve.aal == pytest.approx(2.5)


# -- AEP --------------------------------------------------------------------
def test_aep_at_zero_is_one(ep):
    assert ep.aep(0.0).item() == pytest.approx(1.0)


def test_aep_is_monotone_non_increasing(ep, grid):
    values = ep.aep(grid)
    assert np.all(np.diff(values) <= 1e-12)


def test_aep_is_bounded(ep, grid):
    values = ep.aep(grid)
    assert np.all((values >= 0.0) & (values <= 1.0))


def test_aep_is_vectorised(ep):
    assert ep.aep([1e6, 1e7, 1e8]).shape == (3,)
    assert ep.aep(1e7).shape == (1,)


def test_aep_empirical_matches_definition(sample, ep):
    annual, _ = sample
    threshold = float(np.quantile(annual, 0.5))
    assert ep.aep_empirical(threshold).item() == pytest.approx(np.mean(annual >= threshold))


# -- OEP --------------------------------------------------------------------
def test_oep_is_monotone_non_increasing(ep, grid):
    assert np.all(np.diff(ep.oep(grid)) <= 1e-12)


def test_oep_never_exceeds_aep(ep, grid):
    assert np.all(ep.oep(grid) <= ep.aep(grid) + 1e-12)


def test_oep_without_data_raises(sample):
    annual, _ = sample
    with pytest.raises(ValueError, match="max_event_losses"):
        ExceedanceProbability(annual).oep(1.0)


def test_oep_empirical_matches_definition(sample, ep):
    _, max_event = sample
    threshold = float(np.quantile(max_event, 0.5))
    assert ep.oep_empirical(threshold).item() == pytest.approx(np.mean(max_event >= threshold))


# -- GPD tail ---------------------------------------------------------------
def test_tail_fit_returns_parameters(ep):
    shape, loc, scale, threshold = ep.tail_fit("aep")
    assert loc == 0.0
    assert scale > 0.0
    assert threshold > 0.0
    assert np.isfinite(shape)


def test_tail_is_continuous_at_threshold(ep):
    _, _, _, threshold = ep.tail_fit("aep")
    step = max(threshold * 1e-9, 1e-9)
    below = ep.aep(threshold).item()
    above = ep.aep(threshold + step).item()
    assert above == pytest.approx(below, rel=1e-6)


def test_tail_extrapolates_beyond_the_sample(sample, ep):
    annual, _ = sample
    beyond = float(annual.max()) * 3.0
    # The empirical curve is exactly zero out here; the GPD tail is not.
    assert ep.aep_empirical(beyond).item() == 0.0
    assert 0.0 < ep.aep(beyond).item() < 1e-3


def test_tail_fit_raises_when_sample_too_small():
    curve = ExceedanceProbability([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="too few excesses"):
        curve.tail_fit("aep")


def test_small_sample_still_produces_a_curve():
    curve = ExceedanceProbability([1.0, 2.0, 3.0])
    assert curve.aep(0.0).item() == pytest.approx(1.0)


# -- return periods ---------------------------------------------------------
def test_return_period_is_inverse_of_probability(ep):
    losses = np.array([1e6, 1e7, 5e7])
    assert np.allclose(ep.return_period(losses), 1.0 / ep.aep(losses))


def test_return_period_is_infinite_where_probability_is_zero():
    curve = ExceedanceProbability([1.0, 2.0, 3.0])
    assert np.isinf(curve.return_period(100.0)[0])


def test_return_period_increases_with_loss(ep, grid):
    values = ep.return_period(grid[grid > 0])
    assert np.all(np.diff(values) >= -1e-9)


@pytest.mark.parametrize("rp", [50.0, 100.0, 250.0, 500.0])
def test_loss_at_return_period_inverts_return_period(ep, rp):
    loss = ep.loss_at_return_period(rp).item()
    assert ep.return_period(loss).item() == pytest.approx(rp, rel=1e-6)


def test_loss_at_return_period_is_monotone(ep):
    periods = np.array([2.0, 5.0, 10.0, 50.0, 100.0, 250.0, 1000.0])
    losses = ep.loss_at_return_period(periods)
    assert np.all(np.diff(losses) > 0)


def test_loss_at_return_period_rejects_sub_annual(ep):
    with pytest.raises(ValueError, match=">= 1"):
        ep.loss_at_return_period(0.5)


def test_oep_return_period_exceeds_aep_return_period(ep):
    loss = ep.loss_at_return_period(100.0, kind="aep").item()
    assert ep.return_period(loss, kind="oep") >= ep.return_period(loss, kind="aep")


def test_unknown_kind_raises(ep):
    with pytest.raises(ValueError, match="kind"):
        ep.return_period(1.0, kind="pep")


# -- curve / aggregates -----------------------------------------------------
def test_curve_shape_and_ordering(ep):
    frame = ep.curve("aep", n_points=50)
    assert list(frame.columns) == ["loss", "probability", "return_period"]
    assert len(frame) == 50
    assert frame["loss"].is_monotonic_increasing
    assert np.all(np.diff(frame["probability"]) <= 1e-12)


def test_curve_rejects_too_few_points(ep):
    with pytest.raises(ValueError, match="n_points"):
        ep.curve(n_points=1)


def test_aal_matches_mean(sample, ep):
    annual, _ = sample
    assert ep.aal == pytest.approx(float(np.mean(annual)))


def test_repr(ep):
    assert "ExceedanceProbability" in repr(ep)


# -- result containers ------------------------------------------------------
def test_event_result_retained_loss():
    event = EventResult(storm_id="SYN_1", year=3, gross_loss=100.0, ceded_loss=40.0)
    assert event.retained_loss == pytest.approx(60.0)


def test_year_result_aggregates():
    year = YearResult(
        year=1,
        events=[
            EventResult("a", 1, 100.0),
            EventResult("b", 1, 250.0),
        ],
    )
    assert year.aggregate_loss == pytest.approx(350.0)
    assert year.max_event_loss == pytest.approx(250.0)


def test_year_result_without_events():
    assert YearResult(year=0).max_event_loss == 0.0


def test_oep_loss_at_return_period_never_exceeds_aep(ep):
    """The OEP inversion is capped by the AEP inversion, also far in the tail."""
    rps = np.array([2.0, 10.0, 100.0, 1_000.0, 10_000.0])
    oep_loss = ep.loss_at_return_period(rps, kind="oep")
    aep_loss = ep.loss_at_return_period(rps, kind="aep")
    assert np.all(oep_loss <= aep_loss * (1 + 1e-9))


def test_oep_probability_never_exceeds_aep_far_in_tail(ep, grid):
    """Clipping keeps OEP <= AEP even where both tails are pure GPD extrapolation."""
    far = np.asarray(grid, dtype=float).max() * np.array([10.0, 100.0, 1_000.0])
    assert np.all(ep.oep(far) <= ep.aep(far) + 1e-12)
