import matplotlib.pyplot as plt
import numpy as np
# import arviz as az

import paths
import gwpop

# load in inference results
# mcmc = az.InferenceData.from_netcdf(paths.data / "mcmc_nonparametric.nc4")
# samples = mcmc.posterior

# plot true underlying disribution
# import it from gwpop

plt.plot(1,2)
plt.savefig(paths.figures / "O5_GP.pdf")