import numpy as np


def from_slab(lon, lat, slab_lon, slab_lat, slab_depth, *, offset_km=0.0):
    """
    Nearest-slab-point depth for each query (lon, lat), plus an offset.

    Slab points with non-finite depth are dropped before the nearest-
    neighbour search.

    Parameters
    ----------
    lon, lat : array-like
        Query points (e.g. SSM grid centres).
    slab_lon, slab_lat, slab_depth : array-like, same length
        Slab grid; `slab_depth` is positive-down (km).
    offset_km : float
        Added on top of the slab depth.

    Returns
    -------
    np.ndarray of shape (len(lon),) with depths in km.
    """
    from scipy.spatial import cKDTree

    sl_lon = np.asarray(slab_lon, float)
    sl_lat = np.asarray(slab_lat, float)
    sl_z = np.asarray(slab_depth, float)
    finite = np.isfinite(sl_z)
    if not np.any(finite):
        raise ValueError("no finite slab depths")

    tree = cKDTree(np.column_stack([sl_lon[finite], sl_lat[finite]]))
    pts = np.column_stack([np.asarray(lon, float), np.asarray(lat, float)])
    _, idx = tree.query(pts, k=1)
    z = sl_z[finite][idx] + offset_km
    print(f"slab depths: {len(z)} points, range [{z.min():.1f}, {z.max():.1f}] km")
    return z