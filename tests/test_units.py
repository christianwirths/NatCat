"""Tests for natcat.utils.units."""

from __future__ import annotations

import numpy as np
import pytest

from natcat.utils.units import (
    km2nm,
    kmh2kt,
    kt2kmh,
    kt2ms,
    nm2km,
    saffir_simpson_category,
)


def test_nm2km():
    assert nm2km(1.0) == pytest.approx(1.852)


def test_km2nm_inverts_nm2km():
    assert km2nm(nm2km(37.5)) == pytest.approx(37.5)


def test_kt2kmh():
    assert kt2kmh(10.0) == pytest.approx(18.52)


def test_kmh2kt_inverts_kt2kmh():
    assert kmh2kt(kt2kmh(64.0)) == pytest.approx(64.0)


def test_kt2ms():
    # 1 knot is exactly 1852 m / 3600 s.
    assert kt2ms(1.0) == pytest.approx(0.5144444, abs=1e-6)


def test_conversions_are_vectorised():
    values = np.array([1.0, 2.0, 3.0])
    assert np.allclose(nm2km(values), values * 1.852)


@pytest.mark.parametrize(
    ("wind_kt", "expected"),
    [
        (10.0, "TD"),
        (33.9, "TD"),
        (34.0, "TS"),
        (63.0, "TS"),
        (64.0, "C1"),
        (83.0, "C2"),
        (96.0, "C3"),
        (113.0, "C4"),
        (137.0, "C5"),
        (160.0, "C5"),
    ],
)
def test_saffir_simpson_category(wind_kt, expected):
    assert saffir_simpson_category(wind_kt) == expected


def test_saffir_simpson_category_vectorised():
    out = saffir_simpson_category(np.array([20.0, 40.0, 140.0]))
    assert out.tolist() == ["TD", "TS", "C5"]
