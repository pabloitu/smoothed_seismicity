import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_CATALOG_COLS = ("lon", "lat", "depth", "time", "mag")


def load_grid(path, bbox=None, columns=None):
    """
    Load a regular grid CSV.

    Parameters
    ----------
    path : Path or str
        CSV with at least 'lon' and 'lat'. May also contain 'depth'.
    bbox : tuple, optional
        (lon_min, lon_max, lat_min, lat_max). Points outside are dropped.
    columns : dict, optional
        Rename mapping {csv_name: canonical_name}. Canonical names are
        'lon' and 'lat'. Example: {"X": "lon", "Y": "lat"}.

    Returns
    -------
    pandas.DataFrame
    """
    df = pd.read_csv(path)
    if columns:
        df = df.rename(columns=columns)
    if not {"lon", "lat"}.issubset(df.columns):
        raise ValueError(f"load_grid: expected 'lon','lat' in {list(df.columns)}")
    if bbox is not None:
        lo_min, lo_max, la_min, la_max = bbox
        df = df[
            (df["lon"] >= lo_min) & (df["lon"] <= lo_max)
            & (df["lat"] >= la_min) & (df["lat"] <= la_max)
        ].reset_index(drop=True)
    print(f"load_grid: {len(df)} points from {path}")
    return df


def load_catalog(path, bbox=None, columns=None,
                 from_time=None, to_time=None, max_events=None):
    """
    Load an earthquake catalog CSV.

    Parameters
    ----------
    path : Path or str
    bbox : tuple, optional
        (lon_min, lon_max, lat_min, lat_max). Events outside are dropped.
    columns : dict, optional
        Rename mapping {csv_name: canonical_name} applied before validation.
        Canonical names are 'lon', 'lat', 'depth', 'time', 'mag'.
    from_time, to_time : str or pd.Timestamp, optional
        Half-open time filter: keep events with from_time <= time < to_time.
    max_events : int, optional
        Keep the earliest N events (sorted by 'time').

    Returns
    -------
    pandas.DataFrame
    """
    df = pd.read_csv(path)
    n0 = len(df)
    if columns:
        df = df.rename(columns=columns)
    missing = set(REQUIRED_CATALOG_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"load_catalog: missing required columns {sorted(missing)}")

    mask = pd.Series(True, index=df.index)
    if from_time is not None or to_time is not None:
        t = pd.to_datetime(df["time"], utc=True, errors="coerce")
        if from_time is not None:
            mask &= t >= pd.Timestamp(from_time, tz="UTC")
        if to_time is not None:
            mask &= t < pd.Timestamp(to_time, tz="UTC")
    if bbox is not None:
        lo_min, lo_max, la_min, la_max = bbox
        mask &= (
            (df["lon"] >= lo_min) & (df["lon"] <= lo_max)
            & (df["lat"] >= la_min) & (df["lat"] <= la_max)
        )
    df = df[mask]

    if max_events is not None and len(df) > max_events:
        df = df.sort_values("time").head(max_events)
    df = df.reset_index(drop=True)
    print(f"load_catalog: {len(df)}/{n0} events from {path}")
    return df

def load_slab(path):
    """
    Load a slab xyz file (no header: lon, lat, depth).

    Longitudes in [0, 360] are mapped to [-180, 180]. Depths are stored
    negative-down in the file and returned positive-down.

    Returns
    -------
    lon, lat, depth_km : np.ndarray
        Same length, may contain NaN where the slab is undefined.
    """
    df = pd.read_csv(path, header=None, names=["lon", "lat", "depth"])
    lon = df["lon"].to_numpy(float)
    lon = np.where(lon > 180, lon - 360, lon)
    lat = df["lat"].to_numpy(float)
    z = -df["depth"].to_numpy(float)
    print(f"load_slab: {len(df)} points from {path}")
    return lon, lat, z


def load_ssm_csv(path):
    """
    Read a stage-1 SSM CSV produced by `write_ssm_csv`.

    Returns
    -------
    df : DataFrame with lon, lat, depth and the original rate columns.
    rates : np.ndarray of shape (n_cells, n_bins).
    edges : np.ndarray of shape (n_bins + 1,).
    dM : float, magnitude bin width.
    """
    df = pd.read_csv(path)
    pat = re.compile(r"rate_M([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)")
    cols = []
    for c in df.columns:
        m = pat.fullmatch(c)
        if m:
            cols.append((c, float(m.group(1)), float(m.group(2))))
    if not cols:
        raise ValueError(f"load_ssm_csv: no rate_M*_* columns in {path}")
    cols.sort(key=lambda x: x[1])
    names = [c for c, _, _ in cols]
    lows = np.array([lo for _, lo, _ in cols])
    highs = np.array([hi for _, _, hi in cols])
    edges = np.concatenate([lows[:1], highs])
    widths = highs - lows
    dM = round(float(np.median(widths)), 6)
    if not np.allclose(widths, dM, atol=1e-6):
        raise ValueError(f"load_ssm_csv: inconsistent bin widths {np.unique(widths)}")
    rates = df[names].to_numpy(float)
    print(f"load_ssm_csv: {len(df)} cells, {len(names)} bins, dM={dM}")
    return df, rates, edges, dM


def write_ssm_csv(grid, rates, edges, out_path):
    """
    Write lon, lat and per-bin rate columns to CSV.

    Bin column names are 'rate_M{lo:.1f}_{hi:.1f}'.
    """
    out = Path(out_path)
    if not {"lon", "lat"}.issubset(grid.columns):
        raise ValueError("write_ssm_csv: grid must have 'lon','lat'")
    if rates.shape[0] != len(grid):
        raise ValueError("write_ssm_csv: rates length mismatches grid")

    df = grid[["lon", "lat"]].copy()
    for i in range(rates.shape[1]):
        df[f"rate_M{edges[i]:.1f}_{edges[i+1]:.1f}"] = rates[:, i]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"write_ssm_csv: {out}")

def write_meta(out_path, **kwargs):
    """Write a JSON sidecar with arbitrary fields. Arrays/Paths are coerced."""
    out = Path(out_path)

    def conv(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x

    payload = {k: conv(v) for k, v in kwargs.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"write_meta: {out}")


def safe_log10(values, *, fill=np.nan):
    """
    log10 with non-positive and non-finite inputs replaced by `fill`.

    Use before writing a log raster: rates of 0 produce -inf from
    np.log10, which is not the raster nodata sentinel.
    """
    v = np.asarray(values, float)
    out = np.full_like(v, fill, dtype=float)
    m = (v > 0) & np.isfinite(v)
    out[m] = np.log10(v[m])
    return out


def write_raster(grid, values, out_path):
    """
    Rasterize lon/lat point values to a GeoTIFF (EPSG:4326).

    Each grid row becomes a pixel center; spacing is inferred from the
    unique lons/lats. Cells with no data become NaN; any non-finite
    input value (inf, -inf, NaN) is also written as NaN.
    """
    import rasterio
    from rasterio.transform import from_origin

    out = Path(out_path)
    if not {"lon", "lat"}.issubset(grid.columns):
        raise ValueError("write_raster: grid must have 'lon','lat'")

    lon = grid["lon"].to_numpy(float)
    lat = grid["lat"].to_numpy(float)
    val = np.asarray(values, float)
    val = np.where(np.isfinite(val), val, np.nan)
    if lon.size != val.size:
        raise ValueError("write_raster: values length mismatches grid")

    ulon = np.sort(np.unique(lon))
    ulat = np.sort(np.unique(lat))
    if ulon.size < 2 or ulat.size < 2:
        raise ValueError("write_raster: need >= 2 unique lons/lats")
    dx = float(np.median(np.diff(ulon)))
    dy = float(np.median(np.diff(ulat)))

    nc = int(round((ulon.max() - ulon.min()) / dx)) + 1
    nr = int(round((ulat.max() - ulat.min()) / dy)) + 1
    arr = np.full((nr, nc), np.nan, float)
    for x, y, v in zip(lon, lat, val):
        c = int(round((x - ulon.min()) / dx))
        r = int(round((ulat.max() - y) / dy))
        if 0 <= r < nr and 0 <= c < nc:
            arr[r, c] = v

    out.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(ulon.min() - dx / 2, ulat.max() + dy / 2, dx, dy)
    with rasterio.open(
        out, "w", driver="GTiff", height=nr, width=nc, count=1,
        dtype=arr.dtype, crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(arr, 1)
    print(f"write_raster: {out}")