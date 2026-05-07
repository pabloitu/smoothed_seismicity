# Smoothed seismicity model


## Inputs

The catalog needs `lon`, `lat`, `depth`, `time`, `mag` (rename via the
`columns=` mapping). `time` is parsed as UTC.
The grid CSV needs `lon` and `lat`; anything else is ignored.

If depth is wanted for the intraslab source model, use the Slab2.0 `.xyz` file. Longitudes in [0, 360] and negative-down depths are handled automatically.

## Create SSM

```
cd runs
python run_ssm_intraslab.py
python run_ssm_crustal.py
```

Edit the constants at the top: paths, bbox, `mmin`/`mmax`, kernel range. The
script loads and bbox-filters the catalog, declusters with GK74, builds per-event weights (uniform `1/T`; switches
to completeness correction if the catalog has `mc_window` and `tc_years`
columns -- Check Helmstetter et al., 2007), picks the kernel by CSEP S-test on a train/test split, computes
the smoothed sum on the grid, fits `(a, b)` from the catalog with
Aki/Utsu (replace with own a,b values when necessary), distributes the truncated GR rate per cell per
magnitude bin, and writes the SSM .csv plus JSON metadata with the
parameters and two rasters for visualization (linear and log10 total rate per cell).


## Create OQ source model

```
python run_oq_intraslab.py
python run_oq_crustal.py
```

For intraslab, depth comes from the nearest slab point plus `slab_offset_km`
(default 7.5). USD/LSD are `depth ± thickness/2` with a 
thickness varying with depth.

For crustal, depth, upper-seismogenic-depth (USD), and LSD are constants set at the top of the script
(default 15, 0, 30 km).

## Choosing the kernel

`smoothing.optimize_kernel(events, grid, split_date=..., kernel="adaptive")`
splits the catalog at `split_date`, fits the SSM on the training subset for
each candidate, rescales the forecast to the observed test count (CSEP
S-test), and reports the spatial Poisson log-likelihood. The candidate
maximising logL wins.

`kernel="adaptive"` ranges over `n_neighbors`; `kernel="fixed"` ranges over
`sigma_km`. The (a, b) prefactor cancels between candidates, which is why
this is purely about spatial pattern.


## References

- Gardner, J.K. & Knopoff, L. (1974). *Is the sequence of earthquakes in
  southern California, with aftershocks removed, Poissonian?* BSSA 64.
- Aki, K. (1965). *Maximum likelihood estimate of b in the formula
  log N = a − bM and its confidence limits.* Bull. ERI 43.
- Helmstetter, A., Kagan, Y.Y. & Jackson, D.D. (2007). *High-resolution
  time-independent grid-based forecast for M ≥ 5 earthquakes in California.*
  SRL 78. (Adaptive n-th nearest-neighbour kernel.)
- Frankel, A. (1995). *Mapping seismic hazard in the central and eastern
  United States.* SRL 66. (Fixed-σ kernel.)
- Schorlemmer, D., Gerstenberger, M.C., Wiemer, S., Jackson, D.D. &
  Rhoades, D.A. (2007). *Earthquake likelihood model testing.* SRL 78.
  (CSEP L/N/S tests.)
- Hayes, G.P. et al. (2018). *Slab2: a comprehensive subduction zone geometry
  model.* Science 362.

  