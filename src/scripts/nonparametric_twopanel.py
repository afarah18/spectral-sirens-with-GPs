import matplotlib.pyplot as plt
import numpy as np
import arviz as az
from scipy.stats import gaussian_kde

from nonparametric_inference import TEST_M1S
from data_generation import H0_FID
from gwpop import powerlaw_peak

import paths
NSAMPS = 500

plt.style.use(paths.scripts / "matplotlibrc")

def two_panel(path, hyperparam='H0'):
    id = az.InferenceData.from_netcdf(path)
    samples = id.posterior
    try:
        r = np.nan_to_num(np.exp(samples['log_rate_test'][0]))
    except KeyError :
        r = np.nan_to_num(np.exp(samples['log_rate'][0]))

    fig, axes = plt.subplots(ncols=2,figsize=(7.5,4*.75),facecolor='none',gridspec_kw={'width_ratios': [1.2, 1]})
    logtestm1s = np.log(TEST_M1S)
    rate = 0.
    for i in range(NSAMPS):
        rate += np.trapz(r[i][2:200],logtestm1s[2:200])
        axes[0].plot(TEST_M1S, r[i], lw=0.2, c="blue",alpha=0.05)
    rate /= NSAMPS
    axes[0].plot([], lw=0.2, c="blue",alpha=0.5,label='Gaussian process fit')
    
    norm = np.trapz(powerlaw_peak(TEST_M1S,alpha=-2.7,f_peak=0.05,mMax=78.0,mMin=10.0,mu_m1=30.0,sig_m1=7.0),TEST_M1S)
    axes[0].plot(TEST_M1S, rate*powerlaw_peak(TEST_M1S,alpha=-2.7,f_peak=0.05,mMax=78.0,mMin=10.0,mu_m1=30.0,sig_m1=7.0)/norm*TEST_M1S,
                c='k',label='True distribution')
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].set_ylim(1e-3,100)
    axes[0].set_xlim(5.,100)
    axes[0].set_xlabel("$m_1 \,$[M$_{\odot}$]")
    axes[0].set_ylabel(r"$\frac{d N}{d m_1}\,$[M$_{\odot}^{-1}$Gpc$^{-3}$yr$^{-1}$]")
    axes[0].legend(framealpha=0)

    try:
        prior=np.linspace(samples[hyperparam][0].min(),samples[hyperparam][0].max(),num=200)
        kde=gaussian_kde(samples[hyperparam])
    except np.linalg.LinAlgError:
        prior = H0_FID
        kde = lambda x: 1.
    axes[1].plot(prior,kde(prior))
    axes[1].set_xlabel(hyperparam)
    if hyperparam == 'H0':
        axes[1].axvline(H0_FID,color='k',label="True value")
        axes[1].set_xlabel('$H_0$ [km/s/Mpc]')
        # axes[1].legend(framealpha=0, loc="upper right")
    axes[1].set_ylabel('posterior density')
    plt.tight_layout()
    fig.savefig(paths.figures / "O5_GP_pm.pdf")
    plt.clf()

two_panel(paths.data / "mcmc_nonparametric.nc4")