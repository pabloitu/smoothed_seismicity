import numpy as np
import pandas as pd
import pytest
from openquake.hazardlib.scalerel import WC1994

from ssm.mfd import truncated_gr
from ssm.pointsource import build_sources, check_consistency

hdd = [(0.2, 3.0), (0.6, 5.0), (0.2, 9.0)]
npd = [(0.6, 240.0, 55.0, -130.0), (0.4, 320.0, 85.0, -35.0)]


def make(usd=2.0, lsd=14.0):
    grid = pd.DataFrame({"lon": [0.0, 0.01, 0.02], "lat": [43.0] * 3})
    rates, edges = truncated_gr(np.array([1.0, 2.0, 0.0]), 3.0, 1.2, 3.5, 5.2, 0.1)
    src = build_sources(grid, rates, edges, None, usd, lsd, trt="Active Shallow Crust",
                        msr=WC1994(), npd=npd, aspect=1.0, hdd=hdd)
    return rates, src


def test_shared_pmf_and_planes():
    rates, src = make()
    assert len(src) == 2
    assert len(src[0].hypocenter_distribution.data) == 3
    assert len(src[0].nodal_plane_distribution.data) == 2
    check_consistency(rates, src)


def test_hypocentre_outside_band():
    with pytest.raises(ValueError):
        make(usd=4.0)


def test_last_bin_kept_for_mmax():
    rates, src = make()
    assert len(src[0].mfd.occurrence_rates) == rates.shape[1]
