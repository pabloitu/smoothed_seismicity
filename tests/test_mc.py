import pandas as pd
import numpy as np
import pytest

from ssm.mc import assign_mc_to_events, write_mc_table, read_mc_table


@pytest.fixture
def mc_table():
    return pd.DataFrame({
        "window_index": [0, 1, 2],
        "start_iso":    ["1980-01-01T00:00:00", "2000-01-01T00:00:00", "2010-01-01T00:00:00"],
        "end_iso":      ["2000-01-01T00:00:00", "2010-01-01T00:00:00", "2020-01-01T00:00:00"],
        "n_events":     [100, 200, 300],
        "mc":           [5.2, 4.8, 4.5],
        "b":            [1.0, 1.0, 1.0],
        "years":        [20.0, 10.0, 10.0],
        "mcs_tested":   [[], [], []],
        "p_values":     [[], [], []],
    })


def test_assign_tags_each_window(mc_table):
    events = pd.DataFrame({
        "lon": [-71.0, -71.0, -71.0, -71.0],
        "lat": [-33.0, -33.0, -33.0, -33.0],
        "depth": [25.0, 25.0, 25.0, 25.0],
        "time": ["1990-06-01", "2005-06-01", "2015-06-01", "1850-01-01"],
        "mag": [5.5, 5.0, 4.7, 6.0],
    })
    out = assign_mc_to_events(events, mc_table)
    assert out.loc[0, "mc_window"] == 5.2
    assert out.loc[1, "mc_window"] == 4.8
    assert out.loc[2, "mc_window"] == 4.5
    assert pd.isna(out.loc[3, "mc_window"])
    assert out.loc[3, "mc_window_index"] == -1


def test_assign_preserves_other_columns(mc_table):
    events = pd.DataFrame({
        "lon": [-71.0], "lat": [-33.0], "depth": [25.0],
        "time": ["1990-06-01"], "mag": [5.5], "id": ["evt1"],
    })
    out = assign_mc_to_events(events, mc_table)
    assert "id" in out.columns
    assert out.loc[0, "id"] == "evt1"


def test_table_roundtrip(tmp_path, mc_table):
    p = tmp_path / "MC.txt"
    write_mc_table(mc_table, p)
    df = read_mc_table(p)
    assert list(df.columns) == ["start_iso", "end_iso", "N", "Mc", "b", "Years"]
    assert np.allclose(df["Mc"].to_numpy(), [5.2, 4.8, 4.5])
    assert np.allclose(df["Years"].to_numpy(), [20.0, 10.0, 10.0])


def test_table_handles_nan_mc(tmp_path):
    table = pd.DataFrame({
        "window_index": [0, 1],
        "start_iso": ["1900-01-01T00:00:00", "2000-01-01T00:00:00"],
        "end_iso":   ["2000-01-01T00:00:00", "2020-01-01T00:00:00"],
        "n_events":  [10, 200],
        "mc":        [np.nan, 4.5],
        "b":         [np.nan, 1.0],
        "years":     [100.0, 20.0],
        "mcs_tested": [[], []],
        "p_values":   [[], []],
    })
    p = tmp_path / "MC.txt"
    write_mc_table(table, p)
    df = read_mc_table(p)
    assert pd.isna(df.loc[0, "Mc"])
    assert df.loc[1, "Mc"] == 4.5


