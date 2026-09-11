"""Tests for natcat.hazards.wind_field."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from natcat.hazards.wind_field import (
    DEFAULT_ASYMMETRY_FACTOR,
    DEFAULT_DECAY_EXPONENT,
    max_wind_footprint,
    max_wind_history,
    motion_asymmetry,
    rankine_vortex,
)
from natcat.utils.geo import bearing, haversine_distance


def brute_force_footprint(
    track, coords, *, asymmetry_factor=DEFAULT_ASYMMETRY_FACTOR, exponent=DEFAULT_DECAY_EXPONENT
):
    """Reference implementation: one Python loop over track points."""
    lat = np.asarray(coords)[:, 0]
    lon = np.asarray(coords)[:, 1]
    out = np.zeros(len(lat))
    for _, row in track.iterrows():
        distances = haversine_distance(row["latitude"], row["longitude"], lat, lon, unit="nm")
        v_sym = rankine_vortex(
            distances, row["max_wind_speed_kt"], row["radius_max_wind_nm"], exponent=exponent
        )
        angles = bearing(row["latitude"], row["longitude"], lat, lon)
        v_asym = motion_asymmetry(
            angles, row["heading_deg"], row["translation_speed_kt"], factor=asymmetry_factor
        )
        out = np.maximum(out, v_sym + v_asym)
    return out


# -- rankine_vortex ---------------------------------------------------------
def test_rankine_vortex_zero_at_centre():
    assert float(rankine_vortex(0.0, 100.0, 20.0)) == 0.0


def test_rankine_vortex_peaks_at_rmw():
    assert float(rankine_vortex(20.0, 100.0, 20.0)) == pytest.approx(100.0)


def test_rankine_vortex_linear_inside_eye():
    assert float(rankine_vortex(10.0, 100.0, 20.0)) == pytest.approx(50.0)


@pytest.mark.parametrize(
    ("exponent", "expected"),
    [(2.0, 25.0), (1.0, 50.0), (0.5, 100.0 / np.sqrt(2.0))],
)
def test_rankine_vortex_decay_exponent(exponent, expected):
    assert float(rankine_vortex(40.0, 100.0, 20.0, exponent=exponent)) == pytest.approx(expected)


def test_rankine_vortex_is_monotone_outside_eye():
    r = np.linspace(20.0, 400.0, 100)
    values = rankine_vortex(r, 120.0, 20.0)
    assert np.all(np.diff(values) < 0)


def test_rankine_vortex_broadcasts():
    r = np.array([[10.0, 40.0], [10.0, 40.0]])
    vmax = np.array([[100.0], [50.0]])
    out = rankine_vortex(r, vmax, 20.0)
    assert out.shape == (2, 2)
    assert out[1, 0] == pytest.approx(25.0)


# -- motion_asymmetry -------------------------------------------------------
def test_motion_asymmetry_right_of_track_is_positive():
    # Storm heading due north; a point due east lies to its right.
    assert float(motion_asymmetry(90.0, 0.0, 20.0)) == pytest.approx(10.0)


def test_motion_asymmetry_left_of_track_is_negative():
    assert float(motion_asymmetry(270.0, 0.0, 20.0)) == pytest.approx(-10.0)


def test_motion_asymmetry_zero_along_track():
    assert float(motion_asymmetry(0.0, 0.0, 20.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(motion_asymmetry(180.0, 0.0, 20.0)) == pytest.approx(0.0, abs=1e-12)


def test_motion_asymmetry_wraps_around_north():
    # 350 deg relative to a heading of 10 deg is -20 deg, i.e. left of track.
    assert float(motion_asymmetry(350.0, 10.0, 10.0)) < 0


def test_motion_asymmetry_scales_with_factor():
    a = float(motion_asymmetry(90.0, 0.0, 20.0, factor=1.0))
    b = float(motion_asymmetry(90.0, 0.0, 20.0, factor=0.5))
    assert a == pytest.approx(2.0 * b)


# -- footprint --------------------------------------------------------------
def test_footprint_matches_brute_force(tiny_processed, random_coords):
    fast = max_wind_footprint(tiny_processed, random_coords)
    slow = brute_force_footprint(tiny_processed, random_coords)
    assert np.max(np.abs(fast - slow)) < 1e-9


def test_footprint_matches_brute_force_on_michael(michael_track, random_coords):
    fast = max_wind_footprint(michael_track, random_coords)
    slow = brute_force_footprint(michael_track, random_coords)
    assert np.max(np.abs(fast - slow)) < 1e-8


def test_footprint_is_chunk_size_invariant(tiny_processed, random_coords):
    a = max_wind_footprint(tiny_processed, random_coords, chunk_size=101)
    b = max_wind_footprint(tiny_processed, random_coords, chunk_size=10_000_000)
    assert np.allclose(a, b)


def test_footprint_is_non_negative(tiny_processed, random_coords):
    assert np.all(max_wind_footprint(tiny_processed, random_coords) >= 0.0)


def test_footprint_asymmetry_right_exceeds_left(tiny_processed):
    # The track runs due north along -80 deg; sample symmetric points either side.
    coords = np.array([[26.5, -78.5], [26.5, -81.5]])
    right, left = max_wind_footprint(tiny_processed, coords, asymmetry_factor=0.5)
    symmetric = max_wind_footprint(tiny_processed, coords, asymmetry_factor=0.0)
    assert right > left
    assert symmetric[0] == pytest.approx(symmetric[1], rel=1e-6)


def test_footprint_empty_inputs(tiny_processed):
    assert max_wind_footprint(tiny_processed, np.empty((0, 2))).shape == (0,)
    assert np.all(max_wind_footprint(tiny_processed.head(0), np.zeros((3, 2))) == 0.0)


def test_footprint_rejects_bad_coords(tiny_processed):
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        max_wind_footprint(tiny_processed, np.zeros((3, 3)))


def test_footprint_rejects_unknown_vortex(tiny_processed, random_coords):
    with pytest.raises(ValueError, match="Unknown vortex"):
        max_wind_footprint(tiny_processed, random_coords, vortex="mystery")


def test_footprint_rejects_unprepared_track(tiny_track, random_coords):
    with pytest.raises(KeyError, match="missing column"):
        max_wind_footprint(tiny_track, random_coords)


# -- history ----------------------------------------------------------------
def test_history_is_monotone_non_decreasing(tiny_processed, random_coords):
    times = tiny_processed["time"].to_numpy()[::3]
    history = max_wind_history(tiny_processed, random_coords, times)
    assert history.shape == (len(times), len(random_coords))
    assert np.all(np.diff(history, axis=0) >= -1e-12)


def test_history_last_row_equals_footprint(tiny_processed, random_coords):
    times = tiny_processed["time"].to_numpy()[::3]
    history = max_wind_history(tiny_processed, random_coords, times)
    footprint = max_wind_footprint(tiny_processed, random_coords)
    assert np.allclose(history[-1], footprint)


def test_history_on_michael_matches_footprint(michael_track, random_coords):
    times = pd.date_range(
        michael_track["time"].min(), michael_track["time"].max(), freq="6h"
    ).to_numpy()
    history = max_wind_history(michael_track, random_coords, times)
    assert np.all(np.diff(history, axis=0) >= -1e-12)
    assert np.allclose(history[-1], max_wind_footprint(michael_track, random_coords))


def test_history_before_track_start_is_zero(tiny_processed, random_coords):
    early = np.array([np.datetime64("2019-01-01T00:00:00")])
    assert np.all(max_wind_history(tiny_processed, random_coords, early) == 0.0)


def test_history_matches_brute_force_per_timestep(tiny_processed, random_coords):
    times = tiny_processed["time"].to_numpy()[::4]
    history = max_wind_history(tiny_processed, random_coords, times)
    for i, stamp in enumerate(times):
        subset = tiny_processed[tiny_processed["time"] <= stamp]
        expected = np.maximum(brute_force_footprint(subset, random_coords), 0.0)
        assert np.max(np.abs(history[i] - expected)) < 1e-9


def test_history_requires_time_column(tiny_processed, random_coords):
    with pytest.raises(KeyError, match="time"):
        max_wind_history(tiny_processed.drop(columns=["time"]), random_coords, [0])
