import numpy as np
import pandas as pd
import pytest

from ssm.smoothing import adaptive_radii, compute_ssm, fixed_radii


@pytest.fixture
def events_5():
    return pd.DataFrame({
        "lon":   [-71.0, -71.1, -71.2, -71.3, -71.4],
        "lat":   [-33.0, -33.1, -33.2, -33.3, -33.4],
        "depth": [25.0] * 5,
        "time":  ["2010-01-01"] * 5,
        "mag":   [5.0] * 5,
    })


@pytest.fixture
def dense_grid():
    lons, lats = np.meshgrid(np.arange(-72.0, -70.0, 0.1),
                             np.arange(-34.0, -32.0, 0.1))
    return pd.DataFrame({"lon": lons.ravel(), "lat": lats.ravel()})


def test_adaptive_radii_shape_and_positive(events_5):
    r = adaptive_radii(events_5, n=2, min_km=0.0)
    assert r.shape == (5,)
    assert (r > 0).all()


def test_adaptive_radii_min_km_floor(events_5):
    r = adaptive_radii(events_5, n=1, min_km=1000.0)
    assert (r == 1000.0).all()


def test_adaptive_radii_fallback_when_n_too_large(events_5):
    r = adaptive_radii(events_5, n=100, min_km=0.0)
    assert r.shape == (5,)


def test_adaptive_radii_raises_for_single_event():
    df = pd.DataFrame({"lon": [0.0], "lat": [0.0], "depth": [0.0],
                       "time": ["2010-01-01"], "mag": [5.0]})
    with pytest.raises(ValueError, match=">=2"):
        adaptive_radii(df, n=1)


def test_fixed_radii_constant():
    r = fixed_radii(10, sigma_km=20.0)
    assert r.shape == (10,)
    assert (r == 20.0).all()


def test_fixed_radii_floor():
    r = fixed_radii(5, sigma_km=2.0, min_km=10.0)
    assert (r == 10.0).all()


def test_compute_ssm_conservation(dense_grid):
    """sum(rates) == sum(weights) when all events fully covered."""
    events = pd.DataFrame({
        "lon": [-71.0], "lat": [-33.0], "depth": [25.0],
        "time": ["2010-01-01"], "mag": [5.0],
    })
    rates = compute_ssm(events, dense_grid,
                        weights=np.array([2.5]),
                        radii=np.array([10.0]),
                        max_dist_km=500.0, power=1.5)
    assert np.isclose(rates.sum(), 2.5, rtol=1e-6)


def test_compute_ssm_far_event_zero_contribution():
    events = pd.DataFrame({
        "lon": [0.0], "lat": [0.0], "depth": [0.0],
        "time": ["2010-01-01"], "mag": [5.0],
    })
    grid = pd.DataFrame({"lon": [180.0, 179.0], "lat": [0.0, 0.0]})
    rates = compute_ssm(events, grid,
                        weights=np.array([1.0]),
                        radii=np.array([10.0]),
                        max_dist_km=100.0, power=1.5)
    assert (rates == 0).all()


def test_compute_ssm_peak_at_event_location(dense_grid):
    events = pd.DataFrame({
        "lon": [-71.0], "lat": [-33.0], "depth": [25.0],
        "time": ["2010-01-01"], "mag": [5.0],
    })
    rates = compute_ssm(events, dense_grid,
                        weights=np.array([1.0]),
                        radii=np.array([10.0]))
    d = np.hypot(dense_grid["lon"] - (-71.0), dense_grid["lat"] - (-33.0))
    closest = int(d.idxmin())
    assert rates[closest] == rates.max()


def test_compute_ssm_additive(dense_grid):
    e1 = pd.DataFrame({"lon": [-71.0], "lat": [-33.0], "depth": [25.0],
                       "time": ["2010-01-01"], "mag": [5.0]})
    e2 = pd.DataFrame({"lon": [-71.5], "lat": [-33.5], "depth": [25.0],
                       "time": ["2010-01-01"], "mag": [5.0]})
    eb = pd.concat([e1, e2], ignore_index=True)

    r1 = compute_ssm(e1, dense_grid, weights=np.array([1.0]), radii=np.array([10.0]))
    r2 = compute_ssm(e2, dense_grid, weights=np.array([1.0]), radii=np.array([10.0]))
    rb = compute_ssm(eb, dense_grid,
                     weights=np.array([1.0, 1.0]),
                     radii=np.array([10.0, 10.0]))
    assert np.allclose(rb, r1 + r2)


def test_compute_ssm_skips_non_finite_weights(dense_grid):
    """NaN weights should be silently skipped, not propagate."""
    events = pd.DataFrame({
        "lon": [-71.0, -71.5], "lat": [-33.0, -33.5],
        "depth": [25.0, 25.0],
        "time": ["2010-01-01", "2010-01-01"], "mag": [5.0, 5.0],
    })
    rates = compute_ssm(events, dense_grid,
                        weights=np.array([1.0, np.nan]),
                        radii=np.array([10.0, 10.0]))
    assert np.isfinite(rates).all()
    assert np.isclose(rates.sum(), 1.0, rtol=1e-6)


def test_compute_ssm_length_mismatch_raises():
    events = pd.DataFrame({
        "lon": [0.0, 1.0], "lat": [0.0, 1.0], "depth": [0.0, 0.0],
        "time": ["2010-01-01", "2010-01-01"], "mag": [5.0, 5.0],
    })
    grid = pd.DataFrame({"lon": [0.5], "lat": [0.5]})
    with pytest.raises(ValueError, match="length mismatch"):
        compute_ssm(events, grid,
                    weights=np.array([1.0]),
                    radii=np.array([10.0, 10.0]))

