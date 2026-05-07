import numpy as np
import pandas as pd


def fit_ab(events, *, mmin, t_years=None, delta_m=0.1):
    """
    Fit Gutenberg-Richter (a, b) by maximum likelihood with binning correction.

    Uses Aki/Utsu's MLE for `b` with the Tinti & Mulargia correction for
    binned magnitudes; `a` is set so 10**(a - b*mmin) equals the observed
    annual rate of events with M >= mmin.

    Parameters
    ----------
    events : DataFrame with 'mag' and (if `t_years` is None) 'time'.
    mmin : float
        Lower magnitude cutoff. Events below are dropped from the fit.
    t_years : float, optional
        Catalog duration. Inferred from 'time' if None.
    delta_m : float, default 0.1

    Returns
    -------
    a, b : float
        log10 N(M >= m) = a - b * m, where N is the annual rate.
    """
    if t_years is None:
        t = pd.to_datetime(events["time"], utc=True, errors="coerce")
        t_years = (t.max() - t.min()).total_seconds() / (365.25 * 86400)

    mags = events["mag"].to_numpy(float)
    mags = mags[(~np.isnan(mags)) & (mags >= mmin)]
    if mags.size < 2:
        raise ValueError("need at least 2 events at or above mmin")

    b = np.log10(np.e) / (mags.mean() - (mmin - delta_m / 2))
    a = np.log10(len(mags) / t_years) + b * mmin
    return float(a), float(b)


def truncated_gr(rates_raw, a, b, mmin, mmax, delta_m):
    """
    Distribute a global GR(a, b) rate across grid cells using `rates_raw`
    as the spatial shape, then split into truncated-GR magnitude bins.

    Parameters
    ----------
    rates_raw : np.ndarray of shape (n_cells,)
        Spatial shape from `compute_ssm` (gets renormalized).
    a, b : float
        log10 lambda(M >= m) = a - b * m.
    mmin : float
        Lowest forecast magnitude.
    mmax : float
        Truncation magnitude (no rate above this).
    delta_m : float
        Magnitude bin width.

    Returns
    -------
    rates_bins : np.ndarray of shape (n_cells, n_bins)
    edges : np.ndarray of shape (n_bins + 1,)
    """
    rates_raw = np.asarray(rates_raw, float)
    if rates_raw.ndim != 1:
        raise ValueError("rates_raw must be 1D")
    total = rates_raw.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("sum(rates_raw) must be > 0")

    lam_global = 10.0 ** (a - b * mmin)
    rates_mmin = (rates_raw / total) * lam_global

    edges = np.arange(mmin, mmax + 1e-6, delta_m)
    if edges[-1] < mmax - 1e-5:
        edges = np.append(edges, mmax)
    n_bins = len(edges) - 1

    rates = np.zeros((rates_raw.size, n_bins), float)
    for i in range(n_bins):
        m_lo = edges[i]
        m_hi = edges[i + 1]
        f_lo = 10.0 ** (b * (mmin - m_lo))
        f_hi = 0.0 if m_hi >= mmax else 10.0 ** (b * (mmin - m_hi))
        rates[:, i] = np.maximum(rates_mmin * (f_lo - f_hi), 0.0)

    print(f"truncated GR: a={a:.3f}, b={b:.3f}, mmin={mmin}, mmax={mmax}, "
          f"global lambda={lam_global:.3f}/yr, after truncation={rates.sum():.3f}/yr")
    return rates, edges


def count_check(events, rates_bins, edges, *, t_years=None):
    """Compare observed event count in [edges[0], edges[-1]] to T * sum(rates_bins)."""
    if t_years is None:
        t = pd.to_datetime(events["time"], utc=True, errors="coerce")
        t_years = (t.max() - t.min()).total_seconds() / (365.25 * 86400)

    mags = events["mag"].to_numpy(float)
    mags = mags[~np.isnan(mags)]
    mmin, mmax = float(edges[0]), float(edges[-1])

    obs = int(((mags >= mmin) & (mags <= mmax)).sum())
    exp = float(t_years * rates_bins.sum())
    ratio = exp / obs if obs > 0 else float("nan")

    print(f"count check: T={t_years:.1f} yr, observed={obs}, "
          f"expected={exp:.1f}, ratio={ratio:.3f}")
    return {"observed": obs, "expected": exp, "t_years": t_years}