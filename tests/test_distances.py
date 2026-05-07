import numpy as np

from ssm.distances import gc_dist, R_EARTH_KM


def test_zero_for_identical_points():
    assert gc_dist(0.0, 0.0, 0.0, 0.0) == 0.0


def test_quarter_circle():
    # Equator to north pole = pi/2 radians
    assert np.isclose(gc_dist(0.0, 0.0, 0.0, 90.0),
                      np.pi * R_EARTH_KM / 2.0, rtol=1e-6)


def test_santiago_to_valparaiso():
    d = gc_dist(-70.65, -33.45, -71.62, -33.05)
    assert 100.0 < d < 120.0


def test_pairwise_via_broadcasting():
    lon = np.array([0.0, 0.0, 90.0])
    lat = np.array([0.0, 90.0, 0.0])
    D = gc_dist(lon[:, None], lat[:, None], lon[None, :], lat[None, :])
    assert D.shape == (3, 3)
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(D, D.T)