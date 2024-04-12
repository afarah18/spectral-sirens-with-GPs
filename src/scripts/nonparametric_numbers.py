import arviz as az
import numpy as np
import paths
from data_generation import H0_FID

id = az.InferenceData.from_netcdf(paths.data / "bias/mcmc_nonparametric_16.nc4")
h0samps = id.posterior['H0'][0]

with open(paths.output / "nonparh0percent.txt", "w") as f:
    print(f"{np.std(h0samps)/np.mean(h0samps)*100:.1f}", file=f)
lower = np.mean(h0samps)-np.percentile(h0samps,5)
upper = np.percentile(h0samps,95)-np.mean(h0samps)
with open(paths.output / "nonparh0CI.txt", "w") as f:
    print(f"${np.mean(h0samps):.1f}"+"^{+"+f"{lower:.1f}"+"}"+"_{-"+f"{upper:.1f}"+"}$", file=f)

nonpar_offset = np.abs(h0samps.mean()-H0_FID)/h0samps.std()
with open(paths.output / "nonparh0offset.txt","w") as f:
    print(f"{nonpar_offset:.1f}",file=f)