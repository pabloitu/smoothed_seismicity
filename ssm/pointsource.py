from pathlib import Path

import numpy as np

from openquake.hazardlib.geo import Point
from openquake.hazardlib.mfd import EvenlyDiscretizedMFD
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.source import PointSource
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.sourcewriter import write_source_model


def build_sources(grid, rates, edges, depths, usd, lsd, *,
                  trt, msr, npd, aspect,
                  mesh_spacing=5.0, tom=None):
    """
    Build OpenQuake PointSource objects from per-cell rates.

    All depth quantities are arrays of shape `(n_cells,)`. No half-thickness
    rule is imposed: compute USD/LSD however you want in the run script
    (e.g. `depth ± thickness(depth)` for intraslab; constants for crustal).

    Parameters
    ----------
    grid : DataFrame with 'lon', 'lat'.
    rates : ndarray, shape (n_cells, n_bins).
    edges : ndarray, shape (n_bins + 1,).
    depths, usd, lsd : array_like, shape (n_cells,)
        Hypocentral depth and upper/lower seismogenic depth per cell, km.
    trt : str
    msr : MSR instance
    npd : PMF
    aspect : float
    mesh_spacing : float, default 5.0 km.
    tom : TemporalOccurrenceModel, default PoissonTOM(1.0).

    Returns
    -------
    list[PointSource]
    """
    n = len(grid)
    if rates.shape[0] != n:
        raise ValueError("rates rows must match grid length")
    if rates.shape[1] != len(edges) - 1:
        raise ValueError("rates cols must match len(edges)-1")
    if len(depths) != n or len(usd) != n or len(lsd) != n:
        raise ValueError("depths/usd/lsd lengths must match grid")

    if tom is None:
        tom = PoissonTOM(1.0)

    dM = float(np.median(np.diff(edges)))
    min_mag = float(edges[0]) + dM / 2.0

    lon = grid["lon"].to_numpy(float)
    lat = grid["lat"].to_numpy(float)
    d = np.asarray(depths, float)
    u = np.asarray(usd, float)
    L = np.asarray(lsd, float)

    sources = []
    n_zero = 0
    n_bad = 0
    for i in range(n):
        row = rates[i]
        if not np.any(row > 0):
            n_zero += 1
            continue
        if not (np.isfinite(d[i]) and np.isfinite(u[i]) and np.isfinite(L[i])):
            n_bad += 1
            continue
        sid = f"ps_{len(sources):06d}"
        src = PointSource(
            source_id=sid,
            name=sid,
            tectonic_region_type=trt,
            mfd=EvenlyDiscretizedMFD(
                min_mag=min_mag, bin_width=dM,
                occurrence_rates=row.tolist(),
            ),
            rupture_mesh_spacing=mesh_spacing,
            magnitude_scaling_relationship=msr,
            rupture_aspect_ratio=aspect,
            temporal_occurrence_model=tom,
            upper_seismogenic_depth=float(u[i]),
            lower_seismogenic_depth=float(L[i]),
            location=Point(float(lon[i]), float(lat[i])),
            nodal_plane_distribution=npd,
            hypocenter_distribution=PMF([(1.0, float(d[i]))]),
        )
        sources.append(src)

    print(f"built {len(sources)} sources "
          f"(skipped {n_zero} zero-rate, {n_bad} bad-depth)")
    return sources


def check_consistency(rates, sources, *, rtol=1e-10):
    """Compare per-bin grid totals to the sum across source MFDs."""
    n_bins = rates.shape[1]
    src_sum = np.zeros(n_bins, float)
    for s in sources:
        r = np.asarray(s.mfd.occurrence_rates, float)
        if r.size != n_bins:
            raise ValueError(f"source has {r.size} bins, expected {n_bins}")
        src_sum += r
    grid_sum = rates.sum(axis=0)

    g_tot = float(grid_sum.sum())
    s_tot = float(src_sum.sum())
    rel = (s_tot - g_tot) / g_tot if g_tot else float("nan")
    denom = np.where(grid_sum > 0, grid_sum, 1.0)
    max_rel = float((np.abs(src_sum - grid_sum) / denom).max())

    print(f"consistency: grid={g_tot:.4e}, sources={s_tot:.4e}, "
          f"rel={rel:.2e}, max per-bin rel={max_rel:.2e}")
    if not np.allclose(src_sum, grid_sum, rtol=rtol):
        print(f"  WARNING: mismatch beyond rtol={rtol}")


def write_nrml(sources, out_path, *, name, investigation_time=1.0):
    """Write a list of sources as a NRML source-model XML."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_source_model(
        dest=str(out),
        sources_or_groups=sources,
        name=name,
        investigation_time=investigation_time,
    )
    print(f"wrote {out} ({len(sources)} sources)")