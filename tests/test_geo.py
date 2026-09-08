"""Tests for natcat.utils.geo."""

from __future__ import annotations

import numpy as np
import pytest

from natcat.utils.geo import bearing, destination_point, haversine_distance


def test_haversine_known_distance_km():
    # One degree of latitude is ~111.19 km on a sphere of radius 6371 km.
    assert haversine_distance(0.0, 0.0, 1.0, 0.0, unit="km") == pytest.approx(111.195, abs=0.01)


def test_haversine_known_distance_nm():
    # New York (JFK) to London (LHR) is ~3000 nm.
    distance = haversine_distance(40.6413, -73.7781, 51.4700, -0.4543, unit="nm")
    assert distance == pytest.approx(3000.0, rel=0.01)


def test_haversine_zero_distance():
    assert haversine_distance(12.3, -45.6, 12.3, -45.6) == pytest.approx(0.0, abs=1e-9)


def test_haversine_is_symmetric():
    a = haversine_distance(10.0, 20.0, -5.0, 100.0)
    b = haversine_distance(-5.0, 100.0, 10.0, 20.0)
    assert a == pytest.approx(b)


def test_haversine_broadcasts_to_grid():
    lat_t = np.array([[10.0], [20.0], [30.0]])
    lon_t = np.array([[-80.0], [-80.0], [-80.0]])
    lat_p = np.array([[11.0, 12.0]])
    lon_p = np.array([[-80.0, -80.0]])
    assert haversine_distance(lat_t, lon_t, lat_p, lon_p).shape == (3, 2)


def test_haversine_rejects_bad_unit():
    with pytest.raises(ValueError, match="Invalid unit"):
        haversine_distance(0.0, 0.0, 1.0, 1.0, unit="miles")


@pytest.mark.parametrize(
    ("lat2", "lon2", "expected"),
    [(1.0, 0.0, 0.0), (0.0, 1.0, 90.0), (-1.0, 0.0, 180.0), (0.0, -1.0, 270.0)],
)
def test_bearing_cardinal_directions(lat2, lon2, expected):
    assert float(bearing(0.0, 0.0, lat2, lon2)) == pytest.approx(expected, abs=1e-6)


def test_bearing_northeast_is_about_45():
    assert float(bearing(0.0, 0.0, 1.0, 1.0)) == pytest.approx(45.0, abs=0.01)


def test_bearing_is_in_range():
    rng = np.random.default_rng(0)
    lat = rng.uniform(-80, 80, 200)
    lon = rng.uniform(-180, 180, 200)
    values = bearing(0.0, 0.0, lat, lon)
    assert np.all((values >= 0.0) & (values < 360.0))


def test_destination_point_round_trip():
    lat, lon = destination_point(25.0, -80.0, 45.0, 200.0)
    back = haversine_distance(25.0, -80.0, lat, lon, unit="km")
    assert float(back) == pytest.approx(200.0, rel=1e-6)
    assert float(bearing(25.0, -80.0, lat, lon)) == pytest.approx(45.0, abs=1e-6)


def test_destination_point_due_east():
    lat, lon = destination_point(0.0, 0.0, 90.0, 111.195)
    assert float(lat) == pytest.approx(0.0, abs=1e-9)
    assert float(lon) == pytest.approx(1.0, abs=1e-3)
