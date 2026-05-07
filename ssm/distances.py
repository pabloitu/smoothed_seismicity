import numpy as np

R_EARTH_KM = 6371.0


def gc_dist(lon1, lat1, lon2, lat2):
    """Great-circle distance in km. Inputs in degrees, broadcasting-friendly."""
    lo1 = np.deg2rad(lon1)
    la1 = np.deg2rad(lat1)
    lo2 = np.deg2rad(lon2)
    la2 = np.deg2rad(lat2)
    dlon = lo2 - lo1
    dlat = la2 - la1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return 2.0 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))