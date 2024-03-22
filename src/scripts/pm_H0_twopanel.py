import matplotlib.pyplot as plt
import numpy as np
import arviz as az
from scipy.stats import gaussian_kde

from priors import TEST_M1S
from nonparametric_inference import NSAMPS
from data_generation import H0_FID,TRUEVALS
from gwpop import powerlaw_peak

import paths

plt.style.use(paths.scripts / "matplotlibrc")

def two_panel(path, path_PLP, path_BPL, hyperparam='H0'):
    id = az.InferenceData.from_netcdf(path)
    samples = id.posterior
    r = np.nan_to_num(np.exp(samples['log_rate_test'][0]))

    id_PLP = az.InferenceData.from_netcdf(path_PLP)
    samples_PLP = id_PLP.posterior.sel(chain=0).reset_coords("chain",drop=True)
    r_PLP = np.nan_to_num(np.exp(samples_PLP['log_rate']))
    
    id_BPL = az.InferenceData.from_netcdf(path_BPL)
    samples_BPL = id_BPL.posterior.sel(chain=0).reset_coords("chain",drop=True)
    r_BPL = np.nan_to_num(np.exp(samples_BPL['log_rate']))

    fig, axes = plt.subplots(ncols=2,figsize=(7.5,4*.75),facecolor='none',gridspec_kw={'width_ratios': [1.2, 1]})
    for i in range(NSAMPS//2):
        axes[0].plot(TEST_M1S, r[i]/TEST_M1S, lw=0.1, c="blue",alpha=0.03)
        axes[0].plot(TEST_M1S, r_PLP[i],lw=0.1, c="green",alpha=0.03)
        axes[0].plot(TEST_M1S, r_BPL[i],lw=0.1, c="orange",alpha=0.03)
    
    axes[0].plot(TEST_M1S, samples_PLP['rate'].mean(dim='draw').values*\
                 powerlaw_peak(TEST_M1S,f_peak=TRUEVALS['f_peak'],
                               alpha=TRUEVALS['alpha'],mMax=TRUEVALS['mmax'],mMin=TRUEVALS['mmin'],
                               mu_m1=TRUEVALS['mu_m1'],sig_m1=TRUEVALS['sig_m1']),
                 c='k')
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].set_ylim(5e-4,5)
    axes[0].set_xlim(5.,100)
    axes[0].set_xlabel("$m_1 \,$[M$_{\odot}$]")
    axes[0].set_ylabel(r"$\frac{d N}{d m_1}\,$[M$_{\odot}^{-1}$Gpc$^{-3}$yr$^{-1}$]")
    # axes[0].legend(framealpha=0)

    if hyperparam=='H0':
        prior=np.linspace(50,100,num=100)
        axes[1].axvline(H0_FID,color='k',label="Truth")
        axes[1].set_xlabel('$H_0$ [km/s/Mpc]')
    else:
        prior=np.linspace(samples[hyperparam][0].min(),samples[hyperparam][0].max(),num=200)
        axes[1].set_xlabel(hyperparam)
    try:
        kde=gaussian_kde(samples[hyperparam])
        kde_PLP = gaussian_kde(samples_PLP[hyperparam])
        kde_BPL = gaussian_kde(samples_BPL[hyperparam])
    except np.linalg.LinAlgError:
        prior = H0_FID
        kde = lambda x: 1.
    axes[1].plot(prior,kde(prior),c='blue',label='Gaussian process')
    axes[1].plot(prior,kde_PLP(prior),c='green',label=r'\textsc{Power Law + Peak}')
    axes[1].plot(prior,kde_BPL(prior),c='orange',label='Broken power law')
    axes[1].set_ylabel('posterior density')
    fig.legend(ncol=4,framealpha=0,loc="outside upper center")
    # plt.tight_layout()
    fig.savefig(paths.figures / "O5_pm.pdf")
    plt.clf()

two_panel(paths.data / "mcmc_nonparametric.nc4", 
          paths.data / "mcmc_parametric_PLP.nc4", paths.data / "mcmc_parametric_BPL.nc4")