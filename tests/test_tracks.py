"""Tests for natcat.tracks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from natcat.tracks import (
    add_heading,
    add_translation_velocity,
    fill_missing_rmw,
    interpolate_track,
    prepare_track,
    truncate_after_hurricane,
    willoughby_rmw,
)

PROCESSED_COLUMNS = {
    "time",
    "latitude",
    "longitude",
    "max_wind_speed_kt",
    "radius_max_wind_nm",
    "translation_speed_kt",
    "heading_deg",
}


# -- interpolate_track ------------------------------------------------------
def test_interpolate_track_does_not_mutate_input(tiny_track):
    before = tiny_track.copy(deep=True)
    interpolate_track(tiny_track, freq="1h")
    pd.testing.assert_frame_equal(tiny_track, before)


def test_interpolate_track_time_is_monotone_and_regular(tiny_track):
    out = interpolate_track(tiny_track, freq="1h")
    assert out["time"].is_monotonic_increasing
    assert out["time"].is_unique
    assert len(out) == 19  # 18 hours inclusive
    assert (out["time"].diff().dropna() == pd.Timedelta(hours=1)).all()


def test_interpolate_track_has_no_nan_in_numeric_columns(tiny_track):
    out = interpolate_track(tiny_track, freq="1h")
    numeric = out.select_dtypes(include="number")
    assert not numeric.isna().any().any()


def test_interpolate_track_reproduces_original_fixes(tiny_track):
    out = interpolate_track(tiny_track, freq="1h")
    merged = out.merge(tiny_track, on="time", suffixes=("_new", "_old"))
    assert len(merged) == len(tiny_track)
    assert np.allclose(merged["latitude_new"], merged["latitude_old"])
    assert np.allclose(merged["max_wind_speed_kt_new"], merged["max_wind_speed_kt_old"])


def test_interpolate_track_is_linear_between_fixes(tiny_track):
    out = interpolate_track(tiny_track, freq="1h")
    midpoint = out.loc[out["time"] == pd.Timestamp("2020-01-01 03:00"), "latitude"].iloc[0]
    assert midpoint == pytest.approx(25.5)


def test_interpolate_track_carries_non_numeric_columns(tiny_track):
    out = interpolate_track(tiny_track, freq="1h")
    assert out.loc[0, "storm_type"] == "TS"
    assert out.iloc[-1]["storm_type"] == "HU"


def test_interpolate_track_cubic_falls_back_for_short_tracks(tiny_track):
    short = tiny_track.head(3)
    out = interpolate_track(short, freq="1h", method="cubic")
    assert len(out) == 13


def test_interpolate_track_cubic_runs(tiny_track):
    out = interpolate_track(tiny_track, freq="2h", method="cubic")
    assert out["time"].is_monotonic_increasing
    assert not out["latitude"].isna().any()


def test_interpolate_track_single_point_returns_copy(tiny_track):
    out = interpolate_track(tiny_track.head(1), freq="1h")
    assert len(out) == 1


def test_interpolate_track_rejects_duplicates(tiny_track):
    duplicated = pd.concat([tiny_track, tiny_track.head(1)], ignore_index=True)
    duplicated = duplicated.sort_values("time")
    with pytest.raises(ValueError, match="unique"):
        interpolate_track(duplicated, freq="1h")


def test_interpolate_track_rejects_unsorted(tiny_track):
    with pytest.raises(ValueError, match="sorted"):
        interpolate_track(tiny_track.iloc[::-1], freq="1h")


def test_interpolate_track_rejects_unknown_method(tiny_track):
    with pytest.raises(ValueError, match="method"):
        interpolate_track(tiny_track, method="quadratic")


def test_interpolate_track_requires_time_column(tiny_track):
    with pytest.raises(KeyError):
        interpolate_track(tiny_track.drop(columns=["time"]))


# -- fill_missing_rmw -------------------------------------------------------
@pytest.mark.parametrize(
    ("wind", "expected"),
    [
        (10.0, 80.0),
        (34.9, 80.0),
        (35.0, 60.0),
        (63.0, 60.0),
        (64.0, 40.0),
        (95.0, 40.0),
        (96.0, 25.0),
        (136.0, 25.0),
        (137.0, 15.0),
        (180.0, 15.0),
    ],
)
def test_fill_missing_rmw_step_thresholds(wind, expected):
    df = pd.DataFrame({"max_wind_speed_kt": [wind], "radius_max_wind_nm": [np.nan]})
    out = fill_missing_rmw(df, method="step")
    assert out["radius_max_wind_nm"].iloc[0] == pytest.approx(expected)


def test_willoughby_rmw_values_and_trends():
    # Willoughby et al. (2006): 100 kt at 25N -> 31.9 km = 17.2 nm
    assert willoughby_rmw(100.0, 25.0) == pytest.approx(17.2, abs=0.1)
    assert willoughby_rmw(140.0, 25.0) < willoughby_rmw(100.0, 25.0)  # stronger -> tighter
    assert willoughby_rmw(100.0, 35.0) > willoughby_rmw(100.0, 25.0)  # poleward -> broader
    assert willoughby_rmw(100.0, -25.0) == pytest.approx(willoughby_rmw(100.0, 25.0))


def test_fill_missing_rmw_default_is_willoughby_and_needs_latitude():
    df = pd.DataFrame(
        {"max_wind_speed_kt": [100.0, 100.0], "latitude": [25.0, 25.0],
         "radius_max_wind_nm": [np.nan, 12.0]}
    )  # fmt: skip
    out = fill_missing_rmw(df)["radius_max_wind_nm"]
    assert out.iloc[0] == pytest.approx(17.2, abs=0.1)
    assert out.iloc[1] == 12.0
    with pytest.raises(KeyError, match="latitude"):
        fill_missing_rmw(df.drop(columns="latitude"))
    with pytest.raises(ValueError, match="method"):
        fill_missing_rmw(df, method="magic")


def test_fill_missing_rmw_keeps_observed_values():
    df = pd.DataFrame({"max_wind_speed_kt": [100.0], "radius_max_wind_nm": [12.0]})
    assert fill_missing_rmw(df, method="step")["radius_max_wind_nm"].iloc[0] == 12.0


def test_fill_missing_rmw_replaces_non_positive():
    df = pd.DataFrame({"max_wind_speed_kt": [100.0], "radius_max_wind_nm": [0.0]})
    assert fill_missing_rmw(df, method="step")["radius_max_wind_nm"].iloc[0] == 25.0


def test_fill_missing_rmw_scales_extratropical():
    df = pd.DataFrame(
        {
            "max_wind_speed_kt": [30.0, 30.0],
            "radius_max_wind_nm": [np.nan, np.nan],
            "storm_type": ["TD", "EX"],
        }
    )
    out = fill_missing_rmw(df, method="step")["radius_max_wind_nm"].tolist()
    assert out == pytest.approx([80.0, 120.0])


def test_fill_missing_rmw_adds_column_when_absent():
    df = pd.DataFrame({"max_wind_speed_kt": [50.0]})
    assert fill_missing_rmw(df, method="step")["radius_max_wind_nm"].iloc[0] == 60.0


def test_fill_missing_rmw_does_not_mutate_input():
    df = pd.DataFrame({"max_wind_speed_kt": [50.0], "radius_max_wind_nm": [np.nan]})
    before = df.copy(deep=True)
    fill_missing_rmw(df, method="step")
    pd.testing.assert_frame_equal(df, before)


def test_fill_missing_rmw_requires_wind_column():
    with pytest.raises(KeyError):
        fill_missing_rmw(pd.DataFrame({"radius_max_wind_nm": [10.0]}))


# -- kinematics -------------------------------------------------------------
def test_add_translation_velocity_constant_speed(tiny_track):
    out = add_translation_velocity(tiny_track)
    # One degree of latitude per six hours is 60 nm / 6 h = 10 kt.
    assert np.allclose(out["translation_speed_kt"], 10.0, atol=0.05)


def test_add_translation_velocity_forward_fills_last_point(tiny_track):
    out = add_translation_velocity(tiny_track)
    assert out["translation_speed_kt"].iloc[-1] == out["translation_speed_kt"].iloc[-2]


def test_add_heading_due_north(tiny_track):
    out = add_heading(tiny_track)
    assert np.allclose(out["heading_deg"], 0.0, atol=1e-9)


def test_kinematics_do_not_mutate_input(tiny_track):
    before = tiny_track.copy(deep=True)
    add_translation_velocity(tiny_track)
    add_heading(tiny_track)
    pd.testing.assert_frame_equal(tiny_track, before)


def test_kinematics_handle_empty_frame():
    empty = pd.DataFrame({"time": [], "latitude": [], "longitude": []})
    assert "translation_speed_kt" in add_translation_velocity(empty).columns
    assert "heading_deg" in add_heading(empty).columns


# -- prepare_track ----------------------------------------------------------
def test_prepare_track_columns(tiny_track):
    out = prepare_track(tiny_track, freq="1h")
    assert PROCESSED_COLUMNS.issubset(out.columns)
    assert out["time"].is_monotonic_increasing
    assert not out[list(PROCESSED_COLUMNS - {"time"})].isna().any().any()


def test_prepare_track_does_not_mutate_input(tiny_track):
    before = tiny_track.copy(deep=True)
    prepare_track(tiny_track)
    pd.testing.assert_frame_equal(tiny_track, before)


def test_prepare_track_dedupes_before_interpolating(tiny_track):
    duplicated = pd.concat([tiny_track, tiny_track], ignore_index=True)
    out = prepare_track(duplicated, freq="1h")
    assert out["time"].is_unique


def test_prepare_track_on_michael(michael_track):
    assert PROCESSED_COLUMNS.issubset(michael_track.columns)
    assert len(michael_track) == 103  # 6 Oct 18Z to the last HU fix, 11 Oct 00Z, hourly
    assert michael_track["radius_max_wind_nm"].gt(0).all()
    assert michael_track["translation_speed_kt"].between(0, 60).all()


def test_load_best_track_offline(michael_path):
    from natcat.tracks import load_best_track

    track = load_best_track(2018, "al", "14", download=False)
    assert len(track) == 1225  # 102 h on the default 5-minute grid
    assert track["time"].max() == pd.Timestamp("2018-10-11 00:00")  # last HU fix
    assert len(load_best_track(2018, "al", "14", freq="1h", download=False)) == 103
    full = load_best_track(2018, "al", "14", truncate_after_hurricane=False, download=False)
    assert len(full) == 2593 and full["storm_type"].iloc[-1] == "EX"
    assert track["storm_id"].iloc[0] == "AL142018"


def test_load_best_track_missing_without_download(tmp_path):
    from natcat.tracks import load_best_track

    with pytest.raises(FileNotFoundError):
        load_best_track(1899, "al", "99", data_dir=tmp_path, download=False)


def test_prepare_track_default_step_is_five_minutes(tiny_track):
    out = prepare_track(tiny_track)
    steps = out["time"].diff().dropna().unique()
    assert len(steps) == 1
    assert pd.Timedelta(steps[0]) == pd.Timedelta("5min")


# -- truncate_after_hurricane ----------------------------------------------
def test_truncate_uses_storm_type_and_keeps_reintensification():
    df = pd.DataFrame(
        {
            "max_wind_speed_kt": [30.0, 70.0, 50.0, 75.0, 45.0, 35.0],
            "storm_type": ["TS", "HU", "TS", "HU", "TS", "EX"],
        }
    )
    out = truncate_after_hurricane(df)
    assert out["storm_type"].tolist() == ["TS", "HU", "TS", "HU"]


def test_truncate_falls_back_to_wind_speed():
    df = pd.DataFrame({"max_wind_speed_kt": [40.0, 64.0, 90.0, 63.9, 20.0]})
    assert truncate_after_hurricane(df)["max_wind_speed_kt"].tolist() == [40.0, 64.0, 90.0]


def test_truncate_keeps_storms_that_never_reach_hurricane_strength():
    df = pd.DataFrame({"max_wind_speed_kt": [30.0, 45.0, 60.0], "storm_type": ["TD", "TS", "TS"]})
    out = truncate_after_hurricane(df)
    pd.testing.assert_frame_equal(out, df)
    assert out is not df


def test_truncate_requires_a_classification_column():
    with pytest.raises(KeyError):
        truncate_after_hurricane(pd.DataFrame({"latitude": [1.0]}))


def test_prepare_track_can_skip_truncation(tiny_track):
    weakening = tiny_track.copy()
    weakening["storm_type"] = ["TS", "HU", "HU", "TS"]
    cut = prepare_track(weakening, freq="1h")
    full = prepare_track(weakening, freq="1h", truncate_after_hurricane=False)
    assert cut["time"].max() == pd.Timestamp("2020-01-01 12:00")
    assert full["time"].max() == pd.Timestamp("2020-01-01 18:00")
