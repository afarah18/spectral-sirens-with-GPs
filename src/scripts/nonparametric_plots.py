import matplotlib.pyplot as plt
import numpy as np
import arviz as az

from nonparametric_inference import TEST_M1S, H0_FID
import paths

plt.style.use(paths.scripts / "matplotlibrc")

# load in inference results
mcmc = az.InferenceData.from_netcdf(paths.data / "mcmc_nonparametric.nc4")
samples = mcmc.posterior
r = np.exp(samples['log_rate'][0])
rt = np.exp(samples['log_rate_test'][0])

# plot them
fig, axes = plt.subplots(ncols=2,figsize=(6.5,3))
for i in range(10):
    gp_norm = np.trapz(rt[i],TEST_M1S)
    axes[0].plot(TEST_M1S, rt[i]/gp_norm, lw=0.2, c='b',alpha=0.1)

axes[0].plot([],linewidth=0.5,alpha=0.5, label='Gaussian process fit')

# plot true underlying disribution
# TODO: import it from gwpop
m1s_true = np.load(paths.data / "gw_data/m1s_true_feature_full.npy")
axes[0].hist(m1s_true,bins=50,histtype='step',color='black',label = 'underlying data',density=True)

axes[0].set_xlabel("$m_{1, \\mathrm{source}}$ [$M_{\\odot}$]")
axes[0].legend()
axes[0].set_xlim(right=110,left=0)

# H0 posterior
az.plot_density(mcmc,var_names='H0',ax=axes[1])
axes[1].axvline(H0_FID,color='k')
axes[1].set_xlabel('$H_0$ [km/s/Mpc]')
axes[1].set_ylabel('posterior_density')
fig.savefig(paths.figures / "O5_GP.pdf")