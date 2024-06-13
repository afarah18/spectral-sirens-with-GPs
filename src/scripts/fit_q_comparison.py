import matplotlib.pyplot as plt
import numpy as np
import arviz as az
from scipy.stats import gaussian_kde

from pm_H0_twopanel import color_PLP
from data_generation import H0_FID, OM0_FID
from priors import TEST_M1S
from gwpop import powerlaw, truncnorm
import paths

# when we did these fits, we normalized PLP slightly differently, so let's use that version
def powerlaw_peak(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak):
    tmp_min = 0.01
    tmp_max = 150.
    dmMax = 2
    dmMin = 1

    # Define power-law and peak
    p_m1_pl = powerlaw(m1,low=tmp_min,high=tmp_max,alpha=alpha)
    p_m1_peak = truncnorm(m1, mu=mu_m1, sigma=sig_m1, high=tmp_max, low=tmp_min) #

    # Compute low- and high-mass filters
    low_filter = np.exp(-(m1-mMin)**2/(2.*dmMin**2))
    low_filter = np.where(m1<mMin,low_filter,1.)
    high_filter = np.exp(-(m1-mMax)**2/(2.*dmMax**2))
    high_filter = np.where(m1>mMax,high_filter,1.)

    # Apply filters to combined power-law and peak
    return (f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl)*low_filter*high_filter

plt.style.use(paths.scripts / "matplotlibrc")
color_fitq = color_fitq = "#CD32CD"
TRUEVALS = dict(
    H0=H0_FID, OM0=OM0_FID,
    alpha=-3.4,f_peak=1.4e-8*5, mmax=87,mmin=8.75, mu_m1=34.,sig_m1=3.6,
    zp=1.9,alpha_z=2.7,beta_z=5.6-2.7
)

path_q = paths.data / "samples_powerlaw_peak_smooth_Ndet_500_Nsamples_200_Nfoundinj_5672336_Ninj_50000000_Vz_zmax_15_m1z_power_law_alpha_-0.3_mmin_1.0_mmax_100.0_bq_alpha_m.nc4"
path_noq = paths.data / "samples_powerlaw_peak_smooth_Ndet_500_Nsamples_200_Nfoundinj_5672336_Ninj_50000000_Vz_zmax_15_m1z_power_law_alpha_-0.3_mmin_1.0_mmax_100.0_bq_fixed_alpha_m.nc4"

id = az.InferenceData.from_netcdf(path_noq)
samples_noq = id.posterior.sel(chain=0).reset_coords("chain",drop=True)
id = az.InferenceData.from_netcdf(path_q)
samples_q = id.posterior.sel(chain=0).reset_coords("chain",drop=True)
num_samps = len(samples_q['alpha'])
r_noq = np.zeros((num_samps,len(TEST_M1S)))
r_q = np.copy(r_noq)
for i in range(num_samps):
    r_noq[i] = powerlaw_peak(TEST_M1S,f_peak=samples_noq['f_peak'][i].values*5, # compensate for slightly different definitions of f_peak
                                alpha=samples_noq['alpha'][i].values,mMax=samples_noq['mmax'][i].values,mMin=samples_noq['mmin'][i].values,
                                mu_m1=samples_noq['mu_m1'][i].values,sig_m1=samples_noq['sig_m1'][i].values)
    r_q[i] = powerlaw_peak(TEST_M1S,f_peak=samples_noq['f_peak'][i].values*5, # compensate for slightly different definitions of f_peak
                                alpha=samples_q['alpha'][i].values,mMax=samples_q['mmax'][i].values,mMin=samples_q['mmin'][i].values,
                                mu_m1=samples_q['mu_m1'][i].values,sig_m1=samples_q['sig_m1'][i].values)

fig, axes = plt.subplots(ncols=2,figsize=(7.5,4*.75),facecolor='none',gridspec_kw={'width_ratios': [1.2, 1]})
axes[0].plot(TEST_M1S,np.percentile(r_q/TEST_M1S,(5,95),axis=0).T,lw=0.2, c=color_fitq,alpha=0.7)
axes[0].fill_between(TEST_M1S,*np.percentile(r_q/TEST_M1S,(5,95),axis=0), color=color_fitq,alpha=0.2)
axes[0].plot(TEST_M1S,np.percentile(r_noq/TEST_M1S,(5,95),axis=0).T,lw=2, c=color_PLP,ls="--")
# axes[0].fill_between(TEST_M1S,*np.percentile(r_q/TEST_M1S,(5,95),axis=0), color=color_PLP,alpha=0.5)


axes[0].plot(TEST_M1S, powerlaw_peak(TEST_M1S,f_peak=TRUEVALS['f_peak'],
                            alpha=TRUEVALS['alpha'],mMax=TRUEVALS['mmax'],mMin=TRUEVALS['mmin'],
                            mu_m1=TRUEVALS['mu_m1'],sig_m1=TRUEVALS['sig_m1'])/TEST_M1S,
                c='k')
axes[0].set_yscale('log')
axes[0].set_xscale('log')
axes[0].set_ylim(1e-15,1e-7)
axes[0].set_xlim(5,110)
axes[0].set_xlabel("$m_1 \,$[M$_{\odot}$]")
axes[0].set_ylabel(r"$\frac{{\rm d} N}{{\rm d} m_1}\,$[M$_{\odot}^{-1}$Gpc$^{-3}$yr$^{-1}$]")

prior=np.linspace(50,100,num=100)
axes[1].axvline(H0_FID,color='k',label="Truth")
axes[1].set_xlabel('$H_0$ [km/s/Mpc]')
try:
    kde_noq=gaussian_kde(samples_noq["h0"]*100)
    kde_q=gaussian_kde(samples_q["h0"]*100)
except np.linalg.LinAlgError:
    prior = H0_FID
    kde = lambda x: 1. 
axes[1].plot(prior,kde_noq(prior),c=color_PLP,lw=1,label='Do not fit for secondary mass')
axes[1].plot(prior,kde_q(prior),c=color_fitq,lw=1,label='Fit for secondary mass')
axes[1].set_ylabel('posterior density')
fig.legend(ncol=4,framealpha=0,loc="outside upper center")
plt.tight_layout()
plt.savefig(paths.figures / "q_comparison.pdf")
plt.show()