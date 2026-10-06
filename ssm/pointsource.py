from pathlib import Path

import numpy as np

from openquake.hazardlib.geo import NodalPlane, Point
from openquake.hazardlib.mfd import EvenlyDiscretizedMFD
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.source import PointSource
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.sourcewriter import write_source_model


def build_sources(grid, rates, edges, depths, usd, lsd, *,
                  trt, msr, npd, aspect,
                  mesh_spacing=5.0, tom=None, hdd=None):
    """
    Build OpenQuake PointSource objects from per-cell rates.

    Parameters
    ----------
    grid : DataFrame with 'lon', 'lat'.
    rates : ndarray, shape (n_cells, n_bins).
    edges : ndarray, shape (n_bins + 1,).
    depths : array_like, shape (n_cells,) or None
        Hypocentral depth per cell, km. Ignored when `hdd` is given.
    usd, lsd : float or array_like, shape (n_cells,)
        Upper and lower seismogenic depth, km.
    trt : str
    msr : MSR instance
    npd : PMF or list of (weight, strike, dip, rake)
    aspect : float
    mesh_spacing : float, default 5.0 km.
    tom : TemporalOccurrenceModel, default PoissonTOM(1.0).
    hdd : list of (weight, depth), optional
        Hypocentral depth distribution shared by every source.

    Returns
    -------
    list[PointSource]
    """
    n = len(grid)
    if rates.shape[0] != n:
        raise ValueError("rates rows must match grid length")
    if rates.shape[1] != len(edges) - 1:
        raise ValueError("rates cols must match len(edges)-1")

    u = np.broadcast_to(np.asarray(usd, float), (n,))
    L = np.broadcast_to(np.asarray(lsd, float), (n,))
    if hdd is None:
        d = np.asarray(depths, float)
        if d.shape != (n,):
            raise ValueError("depths length must match grid")
        hz = None
    else:
        w = np.array([x[0] for x in hdd], float)
        dep = np.array([x[1] for x in hdd], float)
        pmf = PMF(list(zip((w / w.sum()).tolist(), dep.tolist())))
        hz = dep
    if not isinstance(npd, PMF):
        npd = PMF([(float(w), NodalPlane(float(st), float(dp), float(rk)))
                   for w, st, dp, rk in npd])

    if tom is None:
        tom = PoissonTOM(1.0)

    dM = float(np.median(np.diff(edges)))
    min_mag = float(edges[0]) + dM / 2.0

    lon = grid["lon"].to_numpy(float)
    lat = grid["lat"].to_numpy(float)

    sources = []
    n_zero = 0
    n_bad = 0
    for i in range(n):
        row = rates[i]
        if not np.any(row > 0):
            n_zero += 1
            continue
        if hz is None:
            if not (np.isfinite(d[i]) and np.isfinite(u[i]) and np.isfinite(L[i])):
                n_bad += 1
                continue
            h = PMF([(1.0, float(d[i]))])
            lo, hi = d[i], d[i]
        else:
            h = pmf
            lo, hi = hz.min(), hz.max()
        if lo < u[i] or hi > L[i]:
            raise ValueError(f"hypocentre outside {u[i]}-{L[i]} km in cell {i}")
        last = int(np.nonzero(row > 0)[0][-1])
        sid = f"ps_{len(sources):06d}"
        src = PointSource(
            source_id=sid,
            name=sid,
            tectonic_region_type=trt,
            mfd=EvenlyDiscretizedMFD(
                min_mag=min_mag, bin_width=dM,
                occurrence_rates=row[:last + 1].tolist(),
            ),
            rupture_mesh_spacing=mesh_spacing,
            magnitude_scaling_relationship=msr,
            rupture_aspect_ratio=aspect,
            temporal_occurrence_model=tom,
            upper_seismogenic_depth=float(u[i]),
            lower_seismogenic_depth=float(L[i]),
            location=Point(float(lon[i]), float(lat[i])),
            nodal_plane_distribution=npd,
            hypocenter_distribution=h,
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
        if r.size > n_bins:
            raise ValueError(f"source has {r.size} bins, expected at most {n_bins}")
        src_sum[:r.size] += r
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