import numpyro
import arviz as az
import numpy as np
from tqdm import trange
import jax

# custom
import paths
from mock_posteriors import gen_snr_scaled_PE
from data_generation import (true_vals_PLP, SNR_THRESH, N_SAMPLES_PER_EVENT,N_SOURCES,
                              H0_FID, OM0_FID, osnr_interp, reference_distance)
from parametric_inference import NSAMPS
from priors import PLP, BPL

jax.config.update("jax_enable_x64", True)

# options
N_CATALOGS=50
N_SOURCES = N_SOURCES//2 # cut catalogs in half for this study, for computational feasibility
plot = True

# random number generators
jax_rng = jax.random.PRNGKey(42)
np_rng = np.random.default_rng(516)

 # load injection set
m1zinj_det = np.load(paths.data / "gw_data/m1zinj_det.npy")
dLinj_det = np.load(paths.data / "gw_data/dLinj_det.npy")
log_pinj_det = np.load(paths.data / "gw_data/log_pinj_det.npy")

if plot:
    import matplotlib.pyplot as plt
bias_PLP = np.zeros(N_CATALOGS)
bias_BPL = np.zeros(N_CATALOGS)
for i in trange(N_CATALOGS):
    ## draw events, find them, and generate mock PE
    m1s_true, zt, m1z_true, dL_true = true_vals_PLP(rng=np_rng)
    m1z_PE, m2z_PE, dL_PE, log_PE_prior = gen_snr_scaled_PE(np_rng,m1s_true,m1s_true,dL_true/1000,osnr_interp,
                                                            reference_distance,N_SAMPLES_PER_EVENT,H0_FID,OM0_FID,
                                                            mc_sigma=3.0e-2,eta_sigma=5.0e-3,theta_sigma=5.0e-2, snr_thresh=SNR_THRESH,
                                                            return_og=False)

    dL_PE *= 1000 # unit matching

    # Inference - power law peak
    nuts_settings = dict(target_accept_prob=0.9, max_tree_depth=10,dense_mass=False)
    nuts_kernel = numpyro.infer.NUTS(PLP,**nuts_settings)
    kwargs = dict(m1det=m1z_PE,dL=dL_PE, m1det_inj=m1zinj_det,dL_inj=dLinj_det,
                    log_pinj=log_pinj_det, log_PE_prior=log_PE_prior,
                    remove_low_Neff=False)
    mcmc = numpyro.infer.MCMC(nuts_kernel,num_warmup=NSAMPS//4*3,num_samples=NSAMPS,
                              num_chains=1,progress_bar=False)   
    mcmc.run(jax_rng,**kwargs)

    id = az.from_numpyro(mcmc)
    id.to_netcdf(paths.data / f"bias/mcmc_parametric_PLP_{i}.nc4")
    bias_PLP[i] = (id.posterior['H0'][0].mean() - H0_FID)/id.posterior['H0'][0].std()
    if plot:
        plt.hist(id.posterior['H0'][0],density=True, bins=50,histtype='step',color='green',alpha=0.5,lw=0.5)
    # Inference - broken power law
    nuts_settings = dict(target_accept_prob=0.9, max_tree_depth=10,dense_mass=False)
    nuts_kernel = numpyro.infer.NUTS(BPL,**nuts_settings)
    kwargs = dict(m1det=m1z_PE,dL=dL_PE, m1det_inj=m1zinj_det,dL_inj=dLinj_det,
                    log_pinj=log_pinj_det, log_PE_prior=log_PE_prior,
                    remove_low_Neff=False)
    mcmc = numpyro.infer.MCMC(nuts_kernel,num_warmup=NSAMPS//4*3,num_samples=NSAMPS,
                              num_chains=1,progress_bar=False)   
    mcmc.run(jax_rng,**kwargs)

    id = az.from_numpyro(mcmc)
    id.to_netcdf(paths.data / f"bias/mcmc_parametric_BPL_{i}.nc4")
    bias_BPL[i] = (id.posterior['H0'][0].mean() - H0_FID)/id.posterior['H0'][0].std()
    if plot:
        plt.hist(id.posterior['H0'][0],density=True, bins=50,histtype='step',color='orange',alpha=0.5,lw=0.5)

# calcualte summary statistics and save
percent_bias_PLP = np.sum(bias_PLP>1)/N_CATALOGS * 100
percent_bias_BPL = np.sum(bias_BPL>1)/N_CATALOGS * 100
with open(paths.output / "PLP_bias_percent.txt", "w") as f:
    print(f"{percent_bias_PLP:.1f}", file=f)
with open(paths.output / "BPL_bias_percent.txt", "w") as f:
    print(f"{percent_bias_BPL:.1f}", file=f)

if plot:
    plt.xlabel("H0")
    plt.axvline(H0_FID,c='k')
    plt.savefig(paths.output / "bias.pdf")