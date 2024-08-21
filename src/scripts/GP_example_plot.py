import matplotlib.pyplot as plt
import numpy as np
import jax
import arviz as az
from tinygp import kernels, GaussianProcess
from tinygp.solvers import QuasisepSolver

from priors import TEST_M1S

import paths

plt.style.use(paths.scripts / "matplotlibrc")

color_GP = "#1f78b4"
logtestm1s = np.log(TEST_M1S)
fig, axes = plt.subplots(ncols=2,figsize=(7.5,4*.75),facecolor='none',sharex=True,sharey=True)

# posterior
axes[1].set_title("Population Posterior")
path_fit = paths.data / "bias/mcmc_nonparametric_29.nc4"

id = az.InferenceData.from_netcdf(path_fit)
samples = id.posterior
r = samples['log_rate_test'][0]

n = min(25,len(r))
for i in range(n):
    axes[1].plot(logtestm1s, r[i], lw=0.7, c=color_GP,alpha=0.5)

# prior
axes[0].set_title("Population Prior")
jax_rng = jax.random.PRNGKey(42)
# use fixed values of mean, sigma and length scale for simplicity 
# these are fit for in the actual inference so this is not techincally the full prior, but will suffice as an illustration
sigma=2.3 
rho = 0.2
mean = samples['mean'].mean().values
kernel = sigma**2 * kernels.quasisep.Matern52(rho)
gp = GaussianProcess(kernel,logtestm1s,mean=mean,diag=0.001,
                        solver=QuasisepSolver,assume_sorted=True)
log_rate_prior = gp.sample(jax_rng,shape=(50,))
for i in range(25):
     axes[0].plot(logtestm1s, log_rate_prior[i], lw=0.7, c=color_GP,alpha=0.5)

for i in (0,1):
     axes[i].set_xlim(1,5.5)
     axes[i].set_xlabel("$\log(m_1/$M$_{\odot})$")
axes[0].set_ylabel(r"$\log \left( \frac{{\rm d} N}{{\rm d} \log{m_1}} / {\rm Gpc}^{-3}{\rm yr}^{-1} \right)$")

fig.savefig(paths.figures / "GP_example.pdf")
plt.clf()