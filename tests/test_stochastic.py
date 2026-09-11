"""Tests for natcat.stochastic.

The catalog is fitted on a small subset of the local ATCF archive to keep the
suite fast; the whole module skips when no archive is available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from natcat.stochastic import (
    PoissonFrequency,
    SyntheticTCCatalog,
    TransitionModel,
    build_state_table,
    extract_genesis_points,
)

N_FILES = 60


@pytest.fixture(scope="module")
def fitted_catalog(request) -> SyntheticTCCatalog:
    """A catalog fitted on the first ``N_FILES`` Atlantic best tracks."""
    data_dir = request.getfixturevalue("raw_data_dir")
    return SyntheticTCCatalog(seed=0).fit(data_dir, max_files=N_FILES)


# -- helpers ----------------------------------------------------------------
def test_extract_genesis_points_empty():
    out = extract_genesis_points([])
    assert out.empty
    assert "latitude" in out.columns


def test_extract_genesis_points_takes_first_fix(tiny_processed):
    points = extract_genesis_points([tiny_processed, tiny_processed])
    assert len(points) == 2
    assert points["latitude"].iloc[0] == pytest.approx(tiny_processed["latitude"].iloc[0])
    assert points["year"].iloc[0] == 2020


def test_build_state_table_empty():
    table = build_state_table([])
    assert table.empty
    assert "state_id" in table.columns


def test_build_state_table_bins_positions(tiny_processed):
    table = build_state_table([tiny_processed], grid_size=2.0)
    assert not table.empty
    assert set(table["lat_bin"].unique()) <= {24.0, 26.0, 28.0}
    assert (table["lon_bin"] == -80.0).all()


def test_transition_model_returns_none_for_unknown_cell(tiny_processed):
    model = TransitionModel().fit([tiny_processed])
    assert model.sample(-60.0, 120.0, np.random.default_rng(0)) is None


def test_transition_model_sample_shape(tiny_processed):
    model = TransitionModel().fit([tiny_processed])
    sample = model.sample(25.5, -79.5, np.random.default_rng(0))
    assert sample is not None
    assert sample.shape == (4,)


def test_poisson_frequency_fit_and_sample():
    years = np.repeat([2000, 2001, 2002], [3, 3, 3])
    model = PoissonFrequency().fit(years)
    assert model.rate == pytest.approx(3.0)
    counts = model.sample(100, np.random.default_rng(0))
    assert counts.shape == (100,)
    assert np.all(counts >= 0)


def test_poisson_frequency_unfitted_raises():
    with pytest.raises(RuntimeError):
        PoissonFrequency().sample(3, np.random.default_rng(0))


def test_poisson_frequency_empty_years_raises():
    with pytest.raises(ValueError):
        PoissonFrequency().fit([])


# -- catalog ----------------------------------------------------------------
def test_fit_populates_components(fitted_catalog):
    assert fitted_catalog.is_fitted
    assert len(fitted_catalog.tracks) > 10
    assert fitted_catalog.transitions.n_states > 0
    assert fitted_catalog.frequency.rate > 0
    assert fitted_catalog.genesis.points is not None


def test_generate_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        SyntheticTCCatalog().generate(n_storms=1)


def test_generate_requires_an_argument(fitted_catalog):
    with pytest.raises(ValueError, match="n_storms or n_years"):
        fitted_catalog.generate()


def test_generate_zero_storms_returns_empty_frame(fitted_catalog):
    catalog = fitted_catalog.generate(n_storms=0)
    assert catalog.empty
    assert {"storm_id", "time", "latitude", "longitude"}.issubset(catalog.columns)


def test_generate_columns_and_size(fitted_catalog):
    catalog = fitted_catalog.generate(n_storms=8)
    expected = {
        "storm_id",
        "time",
        "hour",
        "latitude",
        "longitude",
        "max_wind_speed_kt",
        "radius_max_wind_nm",
        "translation_speed_kt",
        "heading_deg",
    }
    assert expected.issubset(catalog.columns)
    assert catalog["storm_id"].nunique() == 8


def test_generate_is_deterministic_under_seed(raw_data_dir):
    a = SyntheticTCCatalog(seed=42).fit(raw_data_dir, max_files=N_FILES).generate(n_storms=5)
    b = SyntheticTCCatalog(seed=42).fit(raw_data_dir, max_files=N_FILES).generate(n_storms=5)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_catalogs(raw_data_dir):
    a = SyntheticTCCatalog(seed=1).fit(raw_data_dir, max_files=N_FILES).generate(n_storms=5)
    b = SyntheticTCCatalog(seed=2).fit(raw_data_dir, max_files=N_FILES).generate(n_storms=5)
    assert not a["latitude"].to_numpy().tolist() == b["latitude"].to_numpy().tolist()


def test_tracks_stay_within_max_hours(raw_data_dir):
    catalog = SyntheticTCCatalog(seed=0, max_hours=120).fit(raw_data_dir, max_files=N_FILES)
    storms = catalog.generate(n_storms=20)
    assert storms["hour"].max() <= 120
    assert storms.groupby("storm_id").size().max() <= 120 / catalog.time_step_h + 1


def test_generated_physics_is_plausible(fitted_catalog):
    storms = fitted_catalog.generate(n_storms=20)
    assert (storms["max_wind_speed_kt"] >= 0).all()
    assert (storms["radius_max_wind_nm"] >= 5.0).all()
    assert storms["latitude"].between(-90, 90).all()
    assert storms["longitude"].between(-180, 180).all()


def test_generate_with_years_adds_year_column(fitted_catalog):
    storms = fitted_catalog.generate(n_years=20)
    if storms.empty:
        pytest.skip("Poisson sampling produced no storms for this subset")
    assert "year" in storms.columns
    assert storms["year"].between(0, 19).all()


def test_generate_rejects_negative_counts(fitted_catalog):
    with pytest.raises(ValueError):
        fitted_catalog.generate(n_storms=-1)
    with pytest.raises(ValueError):
        fitted_catalog.generate(n_years=-1)


def test_save_and_load_round_trip(fitted_catalog, tmp_path):
    path = fitted_catalog.save(tmp_path / "catalog.pkl")
    restored = SyntheticTCCatalog.load(path)
    assert restored.is_fitted
    assert restored.transitions.n_states == fitted_catalog.transitions.n_states


def test_load_rejects_wrong_pickle(tmp_path):
    import pickle

    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps({"not": "a catalog"}))
    with pytest.raises(TypeError):
        SyntheticTCCatalog.load(path)


def test_fit_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        SyntheticTCCatalog().fit(tmp_path / "nope")


def test_fit_rejects_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        SyntheticTCCatalog().fit(tmp_path)


# -- LossSimulator ----------------------------------------------------------
def test_loss_simulator_runs(fitted_catalog, portfolio):
    from natcat.loss import LossSimulator
    from natcat.vulnerability import WindVulnerability

    simulator = LossSimulator(fitted_catalog, portfolio, WindVulnerability(), seed=0)
    results = simulator.run(5, progress=False)
    assert results.n_years == 5
    assert results.annual_losses.shape == (5,)
    assert results.max_event_losses.shape == (5,)
    assert np.all(results.max_event_losses <= results.annual_losses + 1e-6)
    assert list(results.events.columns) == ["year", "storm_id", "loss"]
    assert results.aal >= 0.0


def test_loss_simulator_rejects_negative_years(fitted_catalog, portfolio):
    from natcat.loss import LossSimulator
    from natcat.vulnerability import WindVulnerability

    simulator = LossSimulator(fitted_catalog, portfolio, WindVulnerability())
    with pytest.raises(ValueError):
        simulator.run(-1)


def test_generated_intensity_and_radius_respect_caps(raw_data_dir):
    """The random walk is bounded by the physical caps on wind and RMW."""
    catalog = SyntheticTCCatalog(seed=5, max_wind_kt=120.0, max_rmw_nm=60.0).fit(
        raw_data_dir, max_files=N_FILES
    )
    tracks = catalog.generate(n_storms=40)
    stepped = tracks[tracks["hour"] > 0]  # genesis fixes come from the historical record
    assert stepped["max_wind_speed_kt"].max() <= 120.0
    assert stepped["radius_max_wind_nm"].max() <= 60.0
    assert stepped["radius_max_wind_nm"].min() >= 5.0


# -- inland decay ------------------------------------------------------------------
def test_land_decay_model_step_floor_and_legacy():
    from natcat.stochastic import LandDecayModel

    model = LandDecayModel(rate_per_h=0.07, floor_kt=25.0)
    assert float(model.step(100.0, 0.0)) == 100.0
    assert float(model.step(100.0, 1e6)) == pytest.approx(25.0)
    assert float(model.step(20.0, 6.0)) == 20.0  # never intensifies over land
    assert model.hours_to(100.0, 64.0) == pytest.approx(np.log(75.0 / 39.0) / 0.07)
    legacy = LandDecayModel.from_rate(0.92)
    assert legacy.floor_kt == 0.0
    assert float(legacy.step(100.0, 3.0)) == pytest.approx(100.0 * 0.92**3)
    with pytest.raises(ValueError):
        LandDecayModel(rate_per_h=0.0)
    with pytest.raises(ValueError):
        LandDecayModel.from_rate(1.5)


def test_land_decay_fit_recovers_parameters():
    from natcat.stochastic import LandDecayModel, extract_landfall_segments

    truth = LandDecayModel(rate_per_h=0.08, floor_kt=22.0)
    rng = np.random.default_rng(0)
    tracks = []
    for k in range(25):
        wind_0 = 60.0 + 3.0 * k
        hours = np.arange(0.0, 60.0, 3.0)
        # start over the Gulf, move north-east onto Florida; land from index 4 on
        lat = 27.0 + 0.4 * np.arange(len(hours))
        lon = -85.5 + 0.35 * np.arange(len(hours))
        wind = np.where(np.arange(len(hours)) < 4, wind_0, truth.step(wind_0, hours - hours[3]))
        wind = wind + rng.normal(0.0, 2.0, len(hours))
        tracks.append(
            pd.DataFrame(
                {
                    "time": pd.Timestamp("2000-01-01") + pd.to_timedelta(hours, unit="h"),
                    "latitude": lat,
                    "longitude": lon,
                    "max_wind_speed_kt": wind,
                }
            )
        )
    segments = extract_landfall_segments(tracks)
    assert segments["storm"].nunique() == 25
    assert (segments["hours"] > 0).all()
    fitted = LandDecayModel.fit(tracks)
    assert fitted.rate_per_h == pytest.approx(0.08, abs=0.015)
    assert fitted.floor_kt == pytest.approx(22.0, abs=4.0)
    assert fitted.n_points == len(segments[segments["hours"] <= 48.0])
    with pytest.raises(ValueError, match="at least 20"):
        LandDecayModel.fit(tracks[:1])


def test_step_track_applies_land_decay_over_land():
    from natcat.stochastic import LandDecayModel, TransitionModel, build_state_table, step_track

    # a slow north-westward track over central Florida, constant intensity
    florida = pd.DataFrame(
        {
            "time": pd.date_range("2000-01-01", periods=5, freq="3h"),
            "latitude": [28.0, 28.1, 28.2, 28.3, 28.4],
            "longitude": [-81.4, -81.5, -81.6, -81.7, -81.8],
            "max_wind_speed_kt": [100.0] * 5,
            "radius_max_wind_nm": [20.0] * 5,
            "translation_speed_kt": [4.0] * 5,
            "heading_deg": [320.0] * 5,
        }
    )
    model = TransitionModel(grid_size=2.0).fit(build_state_table([florida]))
    decay = LandDecayModel(rate_per_h=0.1, floor_kt=0.0)
    rng = np.random.default_rng(0)
    state = {
        "latitude": 28.2,
        "longitude": -81.6,
        "max_wind_speed_kt": 100.0,
        "radius_max_wind_nm": 20.0,
        "_over_land": 1.0,
    }
    _, after = step_track(state, model, rng, time_step_h=3.0, land_decay=decay)
    assert after["_over_land"] == 1.0  # a 4 kt step keeps it inland
    assert after["max_wind_speed_kt"] == pytest.approx(100.0 * np.exp(-0.3))
    assert after["radius_max_wind_nm"] == pytest.approx(20.0 * 1.02**3)
    # legacy float rule gives the old multiplicative decay
    _, legacy = step_track(state, model, rng, time_step_h=3.0, land_decay=0.92)
    assert legacy["max_wind_speed_kt"] == pytest.approx(100.0 * 0.92**3)


def test_fitted_catalog_has_a_land_decay_model_and_terminates_remnants(fitted_catalog):
    from natcat.stochastic import LandDecayModel

    assert isinstance(fitted_catalog.land_decay, LandDecayModel)
    assert fitted_catalog.land_decay.n_points > 100
    assert 0.03 < fitted_catalog.land_decay.rate_per_h < 0.2
    assert 10.0 < fitted_catalog.land_decay.floor_kt < 40.0
    storms = fitted_catalog.generate(n_storms=40)
    assert "_over_land" not in storms.columns
    assert storms["hour"].max() <= fitted_catalog.max_hours
