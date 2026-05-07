from pathlib import Path

from ssm import io, decluster, smoothing, mfd

# inputs
catalog_path = Path("../data/catalog_intraslab.csv")
grid_path = Path("../data/grid.csv")
outdir = Path("../outputs/intraslab")

bbox = (-80.0, -60.0, -62.0, -14.0)
catalog_columns = {"longitude": "lon", "latitude": "lat", "time_iso": "time"}
grid_columns = {"lon": "lon", "lat": "lat"}

# smoothing parameters
power = 1.5
min_km = 5.0
max_dist_km = 300.0

# GR parameters
mmin = 4.9
mmax = 8.0
delta_m = 0.1

# load
events = io.load_catalog(catalog_path,
                         columns=catalog_columns,
                         bbox=bbox,
                         from_time="1976-01-01")
grid = io.load_grid(grid_path, columns=grid_columns, bbox=bbox)

# decluster
events = decluster.decluster_catalog(events, fs_time_prop=0.5,
                                     from_time="1980-01-01")

# weights and kernel radii
w = smoothing.weights(events)  # uniform: 1/T
# completeness-corrected: needs mc_window and tc_years on the catalog
# w = smoothing.weights(events, mc_min=4.9, b_completeness=1)

# adaptive smoothed-seismicity (Helmstetter et al., 2007)
res = smoothing.optimize_kernel(events, grid,
                                candidates=[2, 3, 5, 7, 10, 15],
                                split_date="2010-01-01",
                                kernel="adaptive")
n_neighbors = res["best"]
r = smoothing.adaptive_radii(events, n=n_neighbors, min_km=min_km)

# fixed smoothed-seismicity (Frankel 1995)
# res = smoothing.optimize_kernel(events, grid, split_date="2010-01-01",
#                                 kernel="fixed",
#                                 candidates=[10, 25, 50, 100, 200])
# sigma_km = res["best"]
# r = smoothing.fixed_radii(len(events), sigma_km=sigma_km, min_km=min_km)

# spatial SSM
rates_raw = smoothing.compute_ssm(events, grid, weights=w, radii=r,
                                  max_dist_km=max_dist_km, power=power)

# truncated GR
a, b = mfd.fit_ab(events, mmin=mmin, delta_m=delta_m)

rates_bins, edges = mfd.truncated_gr(
    rates_raw, a=a, b=b, mmin=mmin, mmax=mmax, delta_m=delta_m,
)
mfd.count_check(events, rates_bins, edges)

# outputs
outdir.mkdir(parents=True, exist_ok=True)
io.write_ssm_csv(grid, rates_bins, edges, outdir / "intraslab_ssm.csv")
io.write_meta(outdir / "intraslab_ssm_meta.json",
              a=a, b=b, mmin=mmin, mmax=mmax, delta_m=delta_m, edges=edges)

total_rate = rates_bins.sum(axis=1)
io.write_raster(grid, total_rate, outdir / "intraslab_rate.tif")
io.write_raster(grid, io.safe_log10(total_rate),
                outdir / "intraslab_rate_log10.tif")