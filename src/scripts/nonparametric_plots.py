import matplotlib.pyplot as plt
import numpy as np
import arviz as az

from nonparametric_inference import TEST_M1S
import paths

# load in inference results
mcmc = az.InferenceData.from_netcdf(paths.data / "mcmc_nonparametric.nc4")
samples = mcmc.posterior
r = np.exp(samples['log_rate'][0])
rt = np.exp(samples['log_rate_test'][0])

# plot them
for i in range(100):
    gp_norm = np.trapz(rt[i],TEST_M1S)
    plt.plot(TEST_M1S, rt[i]/gp_norm, lw=0.2, c='b',alpha=0.1)

plt.plot([],linewidth=0.5,alpha=0.5, label='Gaussian process fit')
plt.xlabel("$m_{1, \\mathrm{source}}$")
plt.legend()
plt.xlim(right=110,left=0)

# plot true underlying disribution
# import it from gwpop

plt.savefig(paths.figures / "O5_GP.pdf")