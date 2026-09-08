"""Tests for natcat.data.atcf and natcat.data.quality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from natcat.data import parse_lat_lon, read_a_deck, read_best_track, validate_track

MODERN_LINE = (
    "AL, 14, 2018100618,   , BEST,   0, 178N,  866W,  25, 1006, LO,   0,    ,    0,    0, "
    "   0,    0, 1009,  180,  90,  35,   0,   L,   0,    ,   0,   0,     INVEST, M,  0,    , "
    "   0,    0,    0,    0, genesis-num, 036,\n"
)
HISTORIC_LINE = (
    "AL, 01, 1900082700,   , BEST,   0, 150N,  421W,  35,     , TS, , , , , , , , , , , , , "
    ",    UNNAMED\n"
)


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("178N", 17.8),
        ("178S", -17.8),
        ("866W", -86.6),
        ("866E", 86.6),
        ("  178N ", 17.8),
        ("0N", 0.0),
    ],
)
def test_parse_lat_lon(token, expected):
    assert parse_lat_lon(token) == pytest.approx(expected)


@pytest.mark.parametrize("token", [None, np.nan, "", "   ", "abc"])
def test_parse_lat_lon_missing(token):
    assert np.isnan(parse_lat_lon(token))


def test_read_best_track_handles_modern_and_historic_widths(tmp_path):
    path = tmp_path / "bmix.dat"
    path.write_text(MODERN_LINE + HISTORIC_LINE)

    track = read_best_track(path)
    assert len(track) == 2
    assert track["latitude"].tolist() == [15.0, 17.8]
    assert track["longitude"].tolist() == [-42.1, -86.6]
    # The 25-field historic record carries no RMW and no MSLP.
    historic = track.iloc[0]
    assert np.isnan(historic["radius_max_wind_nm"])
    assert np.isnan(historic["min_pressure_mb"])
    assert historic["max_wind_speed_kt"] == 35.0
    modern = track.iloc[1]
    assert modern["radius_max_wind_nm"] == 90.0
    assert modern["storm_name"] == "INVEST"


def test_read_best_track_maps_zero_rmw_to_nan(tmp_path):
    path = tmp_path / "bzero.dat"
    path.write_text(MODERN_LINE.replace(",  90,", ",   0,"))
    assert np.isnan(read_best_track(path)["radius_max_wind_nm"].iloc[0])


def test_read_best_track_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_best_track(tmp_path / "nope.dat")


def test_read_best_track_michael_dedupes_timestamps(michael_raw):
    # The raw file repeats each fix once per wind radius (RAD 34/50/64).
    assert len(michael_raw) == 38
    assert michael_raw["time"].is_unique
    assert michael_raw["time"].is_monotonic_increasing


def test_read_best_track_michael_schema(michael_raw):
    expected = [
        "time",
        "latitude",
        "longitude",
        "max_wind_speed_kt",
        "radius_max_wind_nm",
        "min_pressure_mb",
        "storm_type",
        "storm_name",
        "basin",
        "storm_number",
    ]
    assert list(michael_raw.columns) == expected
    assert michael_raw["max_wind_speed_kt"].max() == 140.0
    assert "MICHAEL" in set(michael_raw["storm_name"])


def test_read_a_deck_filters_techs(raw_data_dir):
    path = raw_data_dir / "aal142018.dat"
    if not path.exists():
        pytest.skip("A-deck for Michael not available")
    deck = read_a_deck(path, techs=["OFCL"])
    assert not deck.empty
    assert set(deck["tech"].unique()) == {"OFCL"}
    assert {"tech", "tau", "time"}.issubset(deck.columns)


def test_validate_track_accepts_good_track(michael_raw):
    ok, reason = validate_track(michael_raw)
    assert ok, reason


def test_validate_track_rejects_empty():
    ok, reason = validate_track(pd.DataFrame())
    assert not ok
    assert "empty" in reason.lower()


def test_validate_track_rejects_missing_column():
    ok, reason = validate_track(pd.DataFrame({"time": [1], "latitude": [1.0]}))
    assert not ok
    assert "Missing required column" in reason


def test_validate_track_rejects_out_of_bounds():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2020-01-01"]),
            "latitude": [200.0],
            "longitude": [0.0],
            "max_wind_speed_kt": [50.0],
        }
    )
    ok, reason = validate_track(df)
    assert not ok
    assert "Latitude" in reason
