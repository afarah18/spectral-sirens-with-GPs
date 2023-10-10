import matplotlib.pyplot as plt
import numpy as np
import arviz as az

from nonparametric_inference import TEST_M1S
import paths

# load in inference results
mcmc = az.InferenceData.from_netcdf(paths.data / "mcmc_nonparametric.nc4")
samples = mcmc.posterior
r = np.exp(samples['log_rate'])
rt = np.exp(samples['log_rate_test'])


# plot them
for i in range(200):
    gp_norm = np.trapz(rt[i],TEST_M1S)
    if i==1:
        plt.plot(TEST_M1S, rt[i]/gp_norm, lw=0.2, c='b',alpha=0.1,label="gaussian process fit")
    else:
        plt.plot(TEST_M1S, rt[i]/gp_norm, lw=0.2, c='b',alpha=0.1)

plt.xlabel("$m_{1, \\text{source}}$")
plt.legend()
plt.xlim(right=110,left=0)

# plot true underlying disribution
# import it from gwpop

plt.savefig(paths.figures / "O5_GP.pdf")