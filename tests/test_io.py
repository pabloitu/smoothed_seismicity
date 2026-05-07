import pandas as pd
import pytest

from ssm.io import load_catalog


@pytest.fixture
def canonical_csv(tmp_path):
    p = tmp_path / "cat.csv"
    pd.DataFrame({
        "lon":   [-71.0, -72.0, -75.0, -68.0],
        "lat":   [-33.0, -34.0, -36.0, -30.0],
        "depth": [ 25.0,  50.0,  80.0,  15.0],
        "time":  ["2010-01-01", "2015-06-15", "2020-03-20", "2022-11-01"],
        "mag":   [  4.5,   5.2,   6.0,   4.8],
    }).to_csv(p, index=False)
    return p


@pytest.fixture
def renamed_csv(tmp_path):
    p = tmp_path / "cat.csv"
    pd.DataFrame({
        "LON": [-71.0, -72.0],
        "LAT": [-33.0, -34.0],
        "EVDP": [25.0, 50.0],
        "ORIGTIME": ["2010-01-01", "2015-06-15"],
        "MAG": [4.5, 5.2],
    }).to_csv(p, index=False)
    return p


def test_loads_canonical(canonical_csv):
    df = load_catalog(canonical_csv)
    assert len(df) == 4
    assert {"lon", "lat", "depth", "time", "mag"}.issubset(df.columns)


def test_renames_via_columns_arg(renamed_csv):
    df = load_catalog(renamed_csv, columns={
        "LON": "lon", "LAT": "lat", "EVDP": "depth",
        "ORIGTIME": "time", "MAG": "mag",
    })
    assert {"lon", "lat", "depth", "time", "mag"}.issubset(df.columns)
    assert len(df) == 2


def test_bbox_filters_events(canonical_csv):
    df = load_catalog(canonical_csv, bbox=(-73.0, -70.0, -35.0, -32.0))
    assert len(df) == 2
    assert df["lon"].between(-73.0, -70.0).all()
    assert df["lat"].between(-35.0, -32.0).all()


def test_max_events_truncates_to_earliest(canonical_csv):
    df = load_catalog(canonical_csv, max_events=2)
    assert len(df) == 2
    assert df["time"].tolist() == ["2010-01-01", "2015-06-15"]


def test_missing_required_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"lon": [1.0], "lat": [2.0], "mag": [3.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_catalog(p)


def test_extra_columns_pass_through(tmp_path):
    p = tmp_path / "cat.csv"
    pd.DataFrame({
        "lon": [-71.0], "lat": [-33.0], "depth": [25.0],
        "time": ["2010-01-01"], "mag": [4.5],
        "id": ["evt001"], "class": ["intra_slab"], "is_mainshock": [True],
    }).to_csv(p, index=False)
    df = load_catalog(p)
    assert {"id", "class", "is_mainshock"}.issubset(df.columns)


def test_columns_remap_then_bbox(renamed_csv):
    df = load_catalog(
        renamed_csv,
        columns={"LON": "lon", "LAT": "lat", "EVDP": "depth",
                 "ORIGTIME": "time", "MAG": "mag"},
        bbox=(-71.5, -70.0, -33.5, -32.0),
    )
    assert len(df) == 1
    assert df.iloc[0]["lon"] == -71.0