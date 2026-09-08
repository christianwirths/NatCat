"""Tests for natcat.hazards.tropical_cyclone and the hazard base class."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from natcat.hazards import HazardModel, TropicalCycloneHazard
from natcat.hazards.wind_field import max_wind_footprint


def test_peril_type(tiny_processed):
    assert TropicalCycloneHazard(tiny_processed).peril_type == "TC"


def test_is_a_hazard_model(tiny_processed):
    assert isinstance(TropicalCycloneHazard(tiny_processed), HazardModel)


def test_hazard_model_is_abstract():
    with pytest.raises(TypeError):
        HazardModel()


def test_compute_intensity_matches_footprint(tiny_processed, random_coords):
    hazard = TropicalCycloneHazard(tiny_processed)
    assert np.allclose(
        hazard.compute_intensity(random_coords), max_wind_footprint(tiny_processed, random_coords)
    )


def test_compute_intensity_shape(michael_track, portfolio):
    hazard = TropicalCycloneHazard(michael_track)
    coords = portfolio[["latitude", "longitude"]].to_numpy()
    intensity = hazard.compute_intensity(coords)
    assert intensity.shape == (len(portfolio),)
    assert intensity.max() > 60.0  # Michael raked the Panhandle portfolio


def test_decay_exponent_changes_far_field(tiny_processed):
    coords = np.array([[27.0, -82.0]])
    steep = TropicalCycloneHazard(tiny_processed, decay_exponent=2.0).compute_intensity(coords)
    shallow = TropicalCycloneHazard(tiny_processed, decay_exponent=0.5).compute_intensity(coords)
    assert shallow[0] > steep[0]


def test_asymmetry_factor_is_respected(tiny_processed):
    coords = np.array([[26.5, -78.5]])
    with_asym = TropicalCycloneHazard(tiny_processed, asymmetry_factor=0.5).compute_intensity(
        coords
    )
    without = TropicalCycloneHazard(tiny_processed, asymmetry_factor=0.0).compute_intensity(coords)
    assert with_asym[0] > without[0]


def test_compute_intensity_history_shape_and_monotonicity(tiny_processed, random_coords):
    hazard = TropicalCycloneHazard(tiny_processed)
    times = tiny_processed["time"].to_numpy()[::3]
    history = hazard.compute_intensity_history(random_coords, times)
    assert history.shape == (len(times), len(random_coords))
    assert np.all(np.diff(history, axis=0) >= -1e-12)


def test_compute_intensity_history_ends_at_footprint(michael_track, random_coords):
    hazard = TropicalCycloneHazard(michael_track)
    times = pd.date_range(michael_track["time"].min(), michael_track["time"].max(), freq="12h")
    history = hazard.compute_intensity_history(random_coords, times.to_numpy())
    assert np.allclose(history[-1], hazard.compute_intensity(random_coords))


def test_hazard_does_not_mutate_track(tiny_processed, random_coords):
    before = tiny_processed.copy(deep=True)
    hazard = TropicalCycloneHazard(tiny_processed)
    hazard.compute_intensity(random_coords)
    hazard.compute_intensity_history(random_coords, tiny_processed["time"].to_numpy())
    pd.testing.assert_frame_equal(tiny_processed, before)


def test_times_property(tiny_processed):
    assert len(TropicalCycloneHazard(tiny_processed).times) == len(tiny_processed)


def test_repr(tiny_processed):
    assert "TropicalCycloneHazard" in repr(TropicalCycloneHazard(tiny_processed))
