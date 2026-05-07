from pathlib import Path

import numpy as np
from openquake.hazardlib.geo.nodalplane import NodalPlane
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.scalerel.strasser2010 import StrasserIntraslab
from openquake.hazardlib.tom import PoissonTOM

from ssm import io, depths, pointsource

# inputs
ssm_csv = Path("../outputs/intraslab/intraslab_ssm.csv")
slab_path = Path("../data/slab2.0_sam_depth.xyz")
out_path = Path("../outputs/intraslab/intraslab_sources.xml")

# source-model choices
trt = "Subduction IntraSlab"
msr = StrasserIntraslab()
aspect = 1.5
mesh_spacing = 5.0
slab_offset_km = 7.5
investigation_time = 1.0
model_name = "Chile intraslab SSM"

npd = PMF([
    (0.5, NodalPlane(0.0, 60.0, 90.0)),
    (0.5, NodalPlane(180.0, 60.0, 90.0)),
])

# load SSM and slab
grid, rates, edges, dM = io.load_ssm_csv(ssm_csv)
slab_lon, slab_lat, slab_z = io.load_slab(slab_path)

# depth from slab
hypo = depths.from_slab(grid["lon"], grid["lat"],
                        slab_lon, slab_lat, slab_z,
                        offset_km=slab_offset_km)

# slab thickness by depth band
thickness = np.where(hypo < 50.0, 15.0,
              np.where(hypo < 90.0, 20.0,
                       30.0))
usd = hypo - thickness / 2
lsd = hypo + thickness / 2

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