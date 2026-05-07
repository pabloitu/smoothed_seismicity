from dataclasses import dataclass

import numpy as np
import pandas as pd

from .distances import gc_dist


@dataclass
class GKResult:
    is_mainshock: np.ndarray
    cluster_id: np.ndarray


def gk74_windows(mag):
    """Gardner & Knopoff (1974) windows in km / days, Teng & Baker (2019) form."""
    m = np.asarray(mag, dtype=float)
    L = 10.0 ** (0.1238 * m + 0.983)
    T_hi = 10.0 ** (0.032 * m + 2.7389)
    T_lo = 10.0 ** (0.5409 * m - 0.547)
    T = np.where(m >= 6.5, T_hi, T_lo)
    return L, T


def gardner_knopoff(times, mags, lons, lats, fs_time_prop=0.1):
    """
    Gardner & Knopoff (1974) Type-1 declustering.

    Iterates events in descending magnitude order. For each unassigned
    seed, absorbs still-unassigned events whose space-time offset from
    the seed falls inside the seed's window. The window extends backward
    by `fs_time_prop * T` days and forward by `T` days.

    Parameters
    ----------
    times : np.ndarray of np.datetime64
    mags, lons, lats : np.ndarray of float, same length as times
    fs_time_prop : float, default 0.1
        Foreshock-window proportion. Use 1.0 for symmetric windows.

    Returns
    -------
    GKResult with `is_mainshock` (bool array) and `cluster_id` (int array).
    """
    n = len(mags)
    if not (len(times) == len(lons) == len(lats) == n):
        raise ValueError("input arrays must have the same length")

    mags = np.asarray(mags, dtype=float)
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    L_km, T_days = gk74_windows(mags)

    cid = np.zeros(n, dtype=int)
    is_main = np.ones(n, dtype=bool)
    next_id = 1

    order = np.lexsort((times, -mags))
    t0 = times.min()
    t = (times - t0).astype("timedelta64[s]").astype(float) / 86400.0

    for i in order:
        if cid[i] != 0:
            continue
        tw = T_days[i]
        dt = t - t[i]
        in_time = (dt >= -tw * fs_time_prop) & (dt <= tw)
        cands = np.where(in_time & (cid == 0))[0]
        d = gc_dist(lons[i], lats[i], lons[cands], lats[cands])
        members = cands[d <= L_km[i]]
        cid[members] = next_id
        is_main[members] = False
        is_main[i] = True
        next_id += 1

    return GKResult(is_mainshock=is_main, cluster_id=cid)


def decluster_catalog(df, fs_time_prop=1.0, from_time=None, min_mag=None):
    """
    Run Gardner-Knopoff declustering and return mainshocks only.

    Parameters
    ----------
    df : DataFrame with canonical columns 'time', 'mag', 'lon', 'lat'.
    fs_time_prop : float, default 1.0
        1.0 = symmetric foreshock/aftershock window.
    from_time : str or pd.Timestamp, optional
        Skip events before this time (kept in the output as mainshocks).
    min_mag : float, optional
        Skip events below this magnitude (kept in the output as mainshocks).

    Returns
    -------
    DataFrame
        Subset of `df` containing only mainshocks. No extra columns added.
    """
    needed = {"time", "mag", "lon", "lat"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"missing columns: {sorted(miss)}")

    out = df.copy()
    out["t"] = pd.to_datetime(out["time"], utc=True, errors="coerce")

    mask = pd.notna(out["mag"]) & pd.notna(out["t"])
    if from_time is not None:
        mask &= out["t"] >= pd.Timestamp(from_time, tz="UTC")
    if min_mag is not None:
        mask &= out["mag"].astype(float) >= min_mag

    is_main = np.ones(len(out), dtype=bool)
    sub = out.loc[mask]
    if len(sub) > 0:
        res = gardner_knopoff(
            times=sub["t"].dt.tz_convert(None).values.astype("datetime64[s]"),
            mags=sub["mag"].to_numpy(float),
            lons=sub["lon"].to_numpy(float),
            lats=sub["lat"].to_numpy(float),
            fs_time_prop=fs_time_prop,
        )
        idx = np.where(mask.to_numpy())[0]
        is_main[idx] = res.is_mainshock

    n_total = len(out)
    out = out.loc[is_main].drop(columns=["t"]).reset_index(drop=True)
    print(f"declustered: {len(out)}/{n_total} mainshocks")
    return out