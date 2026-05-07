import numpy as np
import pandas as pd

from .distances import gc_dist


def weights(events, *, mc_min=None, b_completeness=1.0, t_years=None):
    """
    Per-event weights for the SSM kernel sum.

    Two modes, picked automatically from the catalog columns:

    * If `events` has 'mc_window' and 'tc_years' (Mc-tagged catalog),
      the weight is the time-varying-completeness correction

          w_i = (1 / tc_i) * 10 ** (b_completeness * (mc_window_i - mc_min))

      `mc_min` defaults to the minimum non-NaN mc_window if not given.
      Pass `b_completeness=None` to disable the magnitude factor and
      keep only 1/tc.

    * Otherwise, all events get `1 / t_years`. `t_years` is inferred
      from the time range of the catalog if not given.

    Returns
    -------
    np.ndarray of shape (n_events,)
    """
    if {"mc_window", "tc_years"}.issubset(events.columns):
        tc = events["tc_years"].to_numpy(float)
        mc = events["mc_window"].to_numpy(float)
        if np.any(tc <= 0):
            raise ValueError("non-positive tc_years")
        w = 1.0 / tc
        if b_completeness is not None:
            if mc_min is None:
                if not np.any(np.isfinite(mc)):
                    raise ValueError("'mc_window' is all NaN; pass mc_min explicitly")
                mc_min = float(np.nanmin(mc))
            mc_clean = np.where(np.isnan(mc), mc_min, mc)
            w = w * 10.0 ** (b_completeness * (mc_clean - mc_min))
        print(f"weights: mc-corrected, mc_min={mc_min}, median={np.median(w):.3e}")
        return w

    if t_years is None:
        t = pd.to_datetime(events["time"], utc=True, errors="coerce")
        t_years = (t.max() - t.min()).total_seconds() / (365.25 * 86400)
    if t_years <= 0:
        raise ValueError("non-positive t_years")
    w = np.full(len(events), 1.0 / t_years)
    print(f"weights: uniform, T={t_years:.2f} yr")
    return w


def adaptive_radii(events, *, n=25, min_km=5.0):
    """Per-event kernel radius = great-circle distance to the n-th nearest event, floored at `min_km`."""
    n_ev = len(events)
    if n_ev < 2:
        raise ValueError("need >= 2 events")
    lon = events["lon"].to_numpy(float)
    lat = events["lat"].to_numpy(float)
    d = gc_dist(lon[:, None], lat[:, None], lon[None, :], lat[None, :])
    d = np.sort(d, axis=1)
    k = min(n, n_ev - 1)
    r = np.maximum(d[:, k], min_km)
    print(f"adaptive radii (n={n}): median={np.median(r):.1f} km, "
          f"range [{r.min():.1f}, {r.max():.1f}]")
    return r


def fixed_radii(n_events, *, sigma_km=25.0, min_km=5.0):
    """Constant kernel radius `sigma_km` per event, floored at `min_km`."""
    r = np.full(int(n_events), max(float(sigma_km), float(min_km)))
    print(f"fixed radii: {len(r)} events, sigma={sigma_km} km")
    return r


def compute_ssm(events, grid, *, weights, radii,
                max_dist_km=500.0, power=1.5):
    """
    Smoothed seismicity sum on a regular grid.

    For each event i with weight w_i and kernel radius d_i, the
    contribution to grid cell j is

        K_ij = 1 / (r_ij**2 + d_i**2) ** power

    normalized over reachable cells (sum_j K_ij == 1 within the
    `max_dist_km` window) and scaled by w_i. The function returns the
    per-cell sum over all events.

    Parameters
    ----------
    events : DataFrame
        Must have 'lon' and 'lat'.
    grid : DataFrame
        Must have 'lon' and 'lat'.
    weights : array-like, length n_events
        Annual rate represented by each event (e.g. 1/T or Mc-corrected).
    radii : array-like, length n_events
        Kernel radius (km) per event. Use `adaptive_radii` or `fixed_radii`.
    max_dist_km : float
        Distance cutoff used per-event for normalization.
    power : float
        Exponent of the inverse-distance kernel.

    Returns
    -------
    np.ndarray of shape (n_cells,)
        Sum of all event contributions per cell.
    """
    if not {"lon", "lat"}.issubset(grid.columns):
        raise ValueError("grid must have 'lon','lat'")
    if not (len(events) == len(weights) == len(radii)):
        raise ValueError(
            f"length mismatch: events={len(events)}, "
            f"weights={len(weights)}, radii={len(radii)}"
        )

    lon_g = grid["lon"].to_numpy(float)
    lat_g = grid["lat"].to_numpy(float)
    lon_e = events["lon"].to_numpy(float)
    lat_e = events["lat"].to_numpy(float)
    w_arr = np.asarray(weights, float)
    r_arr = np.asarray(radii, float)

    n_e = len(events)
    rates = np.zeros(len(grid), dtype=float)
    eps2 = 1e-12

    for i in range(n_e):
        w = w_arr[i]
        d = r_arr[i]
        if not (np.isfinite(w) and np.isfinite(d)) or w == 0.0:
            continue

        r = gc_dist(lon_e[i], lat_e[i], lon_g, lat_g)
        m = r <= max_dist_km
        if not np.any(m):
            continue

        denom = np.maximum(r[m] ** 2 + d ** 2, eps2)
        k = 1.0 / denom ** power
        s = k.sum()
        if s <= 0 or not np.isfinite(s):
            continue

        rates[m] += k * (w / s)

        if (i + 1) % 100 == 0 or i == n_e - 1:
            print(f"  {i + 1}/{n_e}")

    print(f"ssm: total rate {rates.sum():.3f}/yr, "
          f"non-zero cells {(rates > 0).sum()}/{len(rates)}")
    return rates


def optimize_kernel(events, grid, *, split_date, kernel="adaptive",
                    candidates=None, min_km=5.0,
                    max_dist_km=500.0, power=1.5):
    """
    Pick the best spatial kernel by maximising the CSEP S-test log-likelihood
    on a chronological train/test split.

    Splits `events` at `split_date`, fits the SSM on the training subset
    for each candidate parameter, rescales the forecast so its total
    expected count equals the observed test count, and computes the
    per-cell Poisson log-likelihood (count-factorial term omitted since
    it's constant across candidates).

    Parameters
    ----------
    events : DataFrame with 'lon', 'lat', 'time'. Should be declustered
        and time-filtered already.
    grid : DataFrame with 'lon', 'lat'.
    split_date : str or pd.Timestamp
        Training is `time < split_date`; testing is `time >= split_date`.
    kernel : {"adaptive", "fixed"}
        Adaptive uses `candidates` as `n_neighbors` (int);
        fixed uses `candidates` as `sigma_km` (float).
    candidates : sequence, optional
        Defaults: [5, 10, 15, 25, 50, 100] for adaptive,
        [10, 25, 50, 100] km for fixed.
    min_km, max_dist_km, power : float

    Returns
    -------
    dict
        {"best", "scores": [(c, logL), ...], "kernel", "n_train", "n_test"}
    """
    from scipy.spatial import cKDTree

    if kernel not in ("adaptive", "fixed"):
        raise ValueError(f"unknown kernel: {kernel}")
    if candidates is None:
        candidates = [5, 10, 15, 25, 50, 100] if kernel == "adaptive" \
            else [10.0, 25.0, 50.0, 100.0]

    t = pd.to_datetime(events["time"], utc=True, errors="coerce")
    m = t < pd.Timestamp(split_date, tz="UTC")
    train = events[m].reset_index(drop=True)
    test = events[~m].reset_index(drop=True)
    if len(train) < 10 or len(test) < 10:
        raise ValueError(f"too few events: train={len(train)}, test={len(test)}")

    tree = cKDTree(np.column_stack([grid["lon"].to_numpy(float),
                                    grid["lat"].to_numpy(float)]))
    _, idx = tree.query(np.column_stack([test["lon"].to_numpy(float),
                                         test["lat"].to_numpy(float)]),
                        k=1)
    obs = np.bincount(idx, minlength=len(grid))
    n_obs = int(obs.sum())

    w = weights(train)
    eps = 1e-300

    scores = []
    for c in candidates:
        if kernel == "adaptive":
            r = adaptive_radii(train, n=int(c), min_km=min_km)
        else:
            r = fixed_radii(len(train), sigma_km=float(c), min_km=min_km)
        rates = compute_ssm(train, grid, weights=w, radii=r,
                            max_dist_km=max_dist_km, power=power)
        s = rates.sum()
        if s <= 0:
            scores.append((c, -np.inf))
            continue
        p = rates / s
        ll = float(-n_obs + np.sum(obs * np.log(n_obs * p + eps)))
        scores.append((c, ll))

    best = max(scores, key=lambda x: x[1])[0]

    print(f"kernel scan ({kernel}): split={split_date}, "
          f"train={len(train)}, test={len(test)}")
    for c, ll in scores:
        flag = " *" if c == best else ""
        if kernel == "adaptive":
            print(f"  n={c:>3}      logL = {ll:.2f}{flag}")
        else:
            print(f"  sigma={c:>5} km  logL = {ll:.2f}{flag}")

    return {"best": best, "scores": scores, "kernel": kernel,
            "n_train": len(train), "n_test": len(test)}