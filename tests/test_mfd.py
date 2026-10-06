import numpy as np
import pytest

from ssm.mfd import truncated_gr

a, b, mmin, mmax = 2.737, 1.09, 3.5, 5.5


def run(mmax=mmax, dm=0.1, n=5):
    return truncated_gr(np.ones(n), a, b, mmin, mmax, dm)


def test_edges_end_at_mmax():
    _, e = run()
    assert e[0] == pytest.approx(mmin)
    assert e[-1] == pytest.approx(mmax)
    assert len(e) == 21


def test_top_bin_is_not_empty():
    r, e = run()
    assert (r[:, -1] > 0).all()
    exp = 10 ** (a - b * 5.4) - 10 ** (a - b * 5.5)
    assert r[:, -1].sum() == pytest.approx(exp, rel=1e-9)


def test_total_matches_truncated_gr():
    r, _ = run()
    exp = 10 ** (a - b * mmin) * (1 - 10 ** (-b * (mmax - mmin)))
    assert r.sum() == pytest.approx(exp, rel=1e-9)


def test_cumulative_rates():
    r, e = run()
    for m in (4.5, 5.0):
        k = int(np.searchsorted(e, m - 1e-6))
        top = 10 ** (a - b * mmax)
        assert r[:, k:].sum() == pytest.approx(10 ** (a - b * m) - top, rel=1e-9)


def test_uneven_last_bin():
    r, e = run(mmax=5.24)
    assert e[-1] == pytest.approx(5.24)
    assert (r > 0).all()
    exp = 10 ** (a - b * mmin) * (1 - 10 ** (-b * (5.24 - mmin)))
    assert r.sum() == pytest.approx(exp, rel=1e-9)


def test_spatial_shape_kept():
    w = np.array([1.0, 3.0])
    r, _ = truncated_gr(w, a, b, mmin, mmax, 0.1)
    assert r[1].sum() / r[0].sum() == pytest.approx(3.0)
