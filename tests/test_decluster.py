import numpy as np
import pandas as pd
import pytest

from ssm.decluster import decluster_catalog, gardner_knopoff, gk74_windows


def test_windows_monotone_in_mag():
    L, T = gk74_windows(np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
    assert np.all(np.diff(L) > 0)
    assert np.all(np.diff(T) > 0)


def test_isolated_events_all_mainshocks():
    times = np.array(["2000-01-01", "2005-01-01", "2010-01-01", "2015-01-01"],
                     dtype="datetime64[s]")
    mags = np.array([5.0, 5.0, 5.0, 5.0])
    lons = np.array([-70.0, -75.0, -68.0, -73.0])
    lats = np.array([-30.0, -40.0, -25.0, -45.0])
    res = gardner_knopoff(times, mags, lons, lats)
    assert res.is_mainshock.all()
    assert (res.cluster_id > 0).all()
    assert len(np.unique(res.cluster_id)) == 4


def test_aftershock_absorbed():
    times = np.array(["2010-01-01", "2010-01-02"], dtype="datetime64[s]")
    mags = np.array([7.0, 4.5])
    lons = np.array([-71.0, -71.05])
    lats = np.array([-33.0, -33.0])
    res = gardner_knopoff(times, mags, lons, lats)
    assert res.is_mainshock.tolist() == [True, False]
    assert res.cluster_id[0] == res.cluster_id[1]


def test_foreshock_absorbed_with_symmetric_window():
    times = np.array(["2009-12-31", "2010-01-01"], dtype="datetime64[s]")
    mags = np.array([4.5, 7.0])
    lons = np.array([-71.05, -71.0])
    lats = np.array([-33.0, -33.0])
    res = gardner_knopoff(times, mags, lons, lats, fs_time_prop=1.0)
    assert res.is_mainshock.tolist() == [False, True]


def test_distant_foreshock_kept_with_default_fs():
    # 1 year before a M7 event, 0.1*~918d ~= 92d backward window -> outside
    times = np.array(["2009-01-01", "2010-01-01"], dtype="datetime64[s]")
    mags = np.array([4.5, 7.0])
    lons = np.array([-71.05, -71.0])
    lats = np.array([-33.0, -33.0])
    res = gardner_knopoff(times, mags, lons, lats, fs_time_prop=0.1)
    assert res.is_mainshock.tolist() == [True, True]


def test_catalog_adds_columns_and_classifies():
    df = pd.DataFrame({
        "lon":   [-71.0, -71.05],
        "lat":   [-33.0, -33.0],
        "depth": [ 25.0,  25.0],
        "time":  ["2010-01-01", "2010-01-02"],
        "mag":   [  7.0,   4.5],
    })
    out = decluster_catalog(df)
    assert {"is_mainshock", "cluster_id"}.issubset(out.columns)
    assert out["is_mainshock"].tolist() == [True, False]
    assert "_t" not in out.columns


def test_catalog_min_mag_filter_skips_event():
    df = pd.DataFrame({
        "lon":   [-71.0, -71.05],
        "lat":   [-33.0, -33.0],
        "depth": [ 25.0,  25.0],
        "time":  ["2010-01-01", "2010-01-02"],
        "mag":   [  7.0,   4.5],
    })
    out = decluster_catalog(df, min_mag=5.0)
    assert out["is_mainshock"].tolist() == [True, True]
    assert out.iloc[1]["cluster_id"] == 0


def test_catalog_missing_column_raises():
    df = pd.DataFrame({"lon": [1.0], "lat": [2.0], "mag": [3.0]})
    with pytest.raises(ValueError, match="missing columns"):
        decluster_catalog(df)


def test_catalog_preserves_extra_columns():
    df = pd.DataFrame({
        "lon": [-71.0], "lat": [-33.0], "depth": [25.0],
        "time": ["2010-01-01"], "mag": [5.0],
        "id": ["evt001"], "class": ["intra_slab"],
    })
    out = decluster_catalog(df)
    assert {"id", "class"}.issubset(out.columns)