import matplotlib.pyplot as plt
import numpy as np
import arviz as az
from scipy.stats import gaussian_kde

from nonparametric_inference import TEST_M1S
from data_generation import H0_FID
from gwpop import powerlaw_peak

import paths

plt.style.use(paths.scripts / "matplotlibrc")

def two_panel(path):
    id = az.InferenceData.from_netcdf(path)
    samples = id.posterior
    try:
        r = np.nan_to_num(np.exp(samples['log_rate_test'][0]))
    except KeyError :
        r = np.nan_to_num(np.exp(samples['log_rate'][0]))

    fig, axes = plt.subplots(ncols=2,figsize=(20,8),facecolor='none',gridspec_kw={'width_ratios': [1.2, 1]})
    for i in range(20):
        PLP_norm = np.trapz(r[i],TEST_M1S)*1.15
        axes[0].plot(TEST_M1S, r[i]/PLP_norm, lw=0.2, c="blue",alpha=0.1)
        
    axes[0].plot(TEST_M1S, 7*powerlaw_peak(TEST_M1S,alpha=-2.7,f_peak=0.05,mMax=78.0,mMin=10.0,mu_m1=30.0,sig_m1=7.0),
                c='k',lw=3,label='Underlying distribution')
    axes[0].set_yscale('log')
    axes[0].set_ylim(1e-5,1e-1)
    axes[0].set_xlim(1,80)

    prior=np.linspace(55,120,num=200)
    ho_kde=gaussian_kde(samples['H0'])
    axes[1].plot(prior,ho_kde(prior))

    axes[1].axvline(H0_FID,color='k',ls='--',label="True value")
    axes[1].set_xlabel('$H_0$ [km/s/Mpc]')
    axes[1].set_ylabel('posterior density')
    axes[1].legend(framealpha=0)
    plt.tight_layout()
    fig.savefig(paths.figures / "O5_GP.pdf")
    plt.clf()

two_panel(paths.data / "mcmc_nonparametric.nc4")