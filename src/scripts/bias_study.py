import numpyro
import arviz as az
import numpy as np
from tqdm import trange
import jax
import os

# custom
import paths
from mock_posteriors import gen_snr_scaled_PE
from data_generation import (true_vals_PLP, SNR_THRESH, N_SAMPLES_PER_EVENT,N_SOURCES,
                              H0_FID, OM0_FID)
from GWMockCat.vt_utils import interpolate_optimal_snr_grid 
from parametric_inference import NSAMPS
from priors import PLP, BPL, hyper_prior, get_ell_frechet_params, get_sigma_gamma_params
import gwcosmo

jax.config.update("jax_enable_x64", True)

# options
N_CATALOGS=50
N_SOURCES = N_SOURCES
plot = True

# random number generators
jax_rng = jax.random.PRNGKey(42) # these numbers are arbitrary, I think I just copied them from a tutorial
np_rng = np.random.default_rng(516)

# load injection set
m1zinj_det = np.load(paths.data / "gw_data/m1zinj_det.npy")
dLinj_det = np.load(paths.data / "gw_data/dLinj_det.npy")
log_pinj_det = np.load(paths.data / "gw_data/log_pinj_det.npy")

# load SNR interpolator
osnr_interp, reference_distance = interpolate_optimal_snr_grid(
    fname=paths.data / "optimal_snr_aplus_design_O5.h5")

try:
    os.mkdir(paths.data / "bias")
except FileExistsError:
    pass
if plot:
    import matplotlib.pyplot as plt
bias_PLP = np.zeros(N_CATALOGS)
bias_BPL = np.zeros(N_CATALOGS)
for i in trange(N_CATALOGS):
    ## draw events, find them, and generate mock PE
    m1s_true, m2s_true, zt, m1z_true, m2z_true, dL_true = true_vals_PLP(rng=np_rng)
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

    id_PLP = az.from_numpyro(mcmc)
    id_PLP.to_netcdf(paths.data / f"bias/mcmc_parametric_PLP_{i}.nc4")
    bias_PLP[i] = np.abs(id_PLP.posterior['H0'][0].mean() - H0_FID)/id_PLP.posterior['H0'][0].std()
    if plot:
        plt.hist(id_PLP.posterior['H0'][0],density=True, bins=50,histtype='step',color='green',alpha=0.5,lw=0.5)
    
    # Inference - broken power law
    nuts_settings = dict(target_accept_prob=0.9, max_tree_depth=10,dense_mass=False)
    nuts_kernel = numpyro.infer.NUTS(BPL,**nuts_settings)
    mcmc = numpyro.infer.MCMC(nuts_kernel,num_warmup=NSAMPS//4*3,num_samples=NSAMPS,
                              num_chains=1,progress_bar=False)   
    mcmc.run(jax_rng,**kwargs)

    id_BPL = az.from_numpyro(mcmc)
    id_BPL.to_netcdf(paths.data / f"bias/mcmc_parametric_BPL_{i}.nc4")
    bias_BPL[i] = np.abs(id_BPL.posterior['H0'][0].mean() - H0_FID)/id_BPL.posterior['H0'][0].std()
    if plot:
        plt.hist(id_BPL.posterior['H0'][0],density=True, bins=50,histtype='step',color='orange',alpha=0.5,lw=0.5)
    
    # inference - GP
    # this is expensive so we will by default only do it one time.
    # arbitrarily choose an index to do it on so that its reproducible every time.
    # I like 16 bc 4^2 = 2^4 = 16, so why not use that
    if i==16:       
        # parametric summary stats that we only need for this catalog
        with open(paths.output / "PLPh0offset.txt","w") as f:
            print(f"{bias_PLP[i]:.1f}",file=f)
        with open(paths.output / "PLPh0percent.txt","w") as f:
            print(f"{np.std(id_PLP.posterior['H0'][0])/np.mean(id_PLP.posterior['H0'][0])*100:.0f}",file=f)
        with open(paths.output / "BPLh0offset.txt","w") as f:
            print(f"{bias_BPL[i]:.1f}",file=f)
        
        # Penalized complexity priors on the hyper-hyper parameters
        scale, concentration, L = get_ell_frechet_params(np.log(m1z_PE).mean(axis=1),return_L=True)
        conc, lam_sigma = get_sigma_gamma_params(U=2.)
        
        nuts_kernel = numpyro.infer.NUTS(hyper_prior,**nuts_settings)
        kwargs = dict(m1det=m1z_PE,dL=dL_PE, m1det_inj=m1zinj_det,dL_inj=dLinj_det,
                        log_pinj=log_pinj_det, log_PE_prior=log_PE_prior,
                        PC_params=dict(conc=conc,concentration=concentration,scale=scale,lam_sigma=lam_sigma),
                        remove_low_Neff=False)
        mcmc = numpyro.infer.MCMC(nuts_kernel,num_warmup=NSAMPS,num_samples=NSAMPS,
                                num_chains=1,progress_bar=True)   
        mcmc.run(jax_rng,**kwargs)
        id = az.from_numpyro(mcmc)
        id.to_netcdf(paths.data / f"bias/mcmc_nonparametric_{i}.nc4")
        
        # GP-specific summary stats
        h0samps = id.posterior['H0'][0]
        with open(paths.output / "nonparh0percent.txt", "w") as f:
            print(f"{np.std(h0samps)/np.mean(h0samps)*100:.0f}", file=f)
        lower = np.mean(h0samps)-np.percentile(h0samps,5)
        upper = np.percentile(h0samps,95)-np.mean(h0samps)
        with open(paths.output / "nonparh0CI.txt", "w") as f:
            print(f"${np.mean(h0samps):.1f}"+"^{+"+f"{lower:.1f}"+"}"+"_{-"+f"{upper:.1f}"+"}$", file=f)
        nonpar_offset = np.abs(h0samps.mean()-H0_FID)/h0samps.std()
        with open(paths.output / "nonparh0offset.txt","w") as f:
            print(f"{nonpar_offset:.1f}",file=f)

# calcualte summary statistics and save
percent_bias_PLP = np.sum(bias_PLP>1)/N_CATALOGS * 100
percent_bias_BPL = np.sum(bias_BPL>1)/N_CATALOGS * 100
with open(paths.output / "PLP_bias_percent.txt", "w") as f:
    print(f"{percent_bias_PLP:.0f}", file=f)
with open(paths.output / "BPL_bias_percent.txt", "w") as f:
    print(f"{percent_bias_BPL:.0f}", file=f)
np.savetxt(paths.data / "bias/bias_PLP.txt",bias_PLP)
np.savetxt(paths.data / "bias/bias_BPL.txt",bias_BPL)

if plot:
    plt.xlabel("H0")
    plt.axvline(H0_FID,c='k')
    plt.savefig(paths.output / "bias.pdf")