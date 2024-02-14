import arviz as az
import numpy as np
import paths

id = az.InferenceData.from_netcdf(paths.data / "mcmc_nonparametric.nc4")
h0samps = id.posterior['H0'][0]
print(len(h0samps))

with open(paths.output / "nonparh0percent.txt", "w") as f:
    print(f"{np.std(h0samps)/np.mean(h0samps):.1f}", file=f)
with open(paths.output / "nonparh0CI.txt", "w") as f:
    print(f"{np.mean(h0samps):.1f}"+"^{+"+f"{np.percentile(h0samps,95)}"+"}"+"_{-"+f"{np.percentile(h0samps,95)}"+"}", file=f)