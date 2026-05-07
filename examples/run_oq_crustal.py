from pathlib import Path

import numpy as np
from openquake.hazardlib.geo.nodalplane import NodalPlane
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.scalerel import WC1994
from openquake.hazardlib.tom import PoissonTOM

from ssm import io, pointsource

# inputs
ssm_csv = Path("../outputs/crustal/crustal_ssm.csv")
out_path = Path("../outputs/crustal/crustal_sources.xml")

# source-model choices
trt = "Active Shallow Crust"
msr = WC1994()
aspect = 1.0
mesh_spacing = 5.0
investigation_time = 1.0
model_name = "Chile crustal SSM"

# constant depth geometry
hypo_km = 15.0
usd_km = 0.0
lsd_km = 30.0

npd = PMF([
    (0.5, NodalPlane(0.0, 60.0, 90.0)),
    (0.5, NodalPlane(180.0, 60.0, 90.0)),
])

# load SSM
grid, rates, edges, dM = io.load_ssm_csv(ssm_csv)

# per-cell depths (constant)
hypo = np.full(len(grid), hypo_km)
usd = np.full(len(grid), usd_km)
lsd = np.full(len(grid), lsd_km)

# build, check, write
sources = pointsource.build_sources(
    grid, rates, edges, hypo, usd, lsd,
    trt=trt, msr=msr, npd=npd, aspect=aspect,
    mesh_spacing=mesh_spacing,
    tom=PoissonTOM(investigation_time),
)
pointsource.check_consistency(rates, sources)
pointsource.write_nrml(sources, out_path,
                       name=model_name,
                       investigation_time=investigation_time)