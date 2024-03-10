# custom
import jgwcosmo # for inference
import jgwpop
import paths
Clight = jgwcosmo.Clight
from priors import hyper_prior, get_ell_frechet_params, get_sigma_gamma_params

# inference
import jax
import numpyro
import arviz as az

# data
import numpy as np

jax.config.update("jax_enable_x64", True)

NSAMPS=5

# random number generators
jax_rng = jax.random.PRNGKey(425)

# options
remove_low_Neff=False

if  __name__ == "__main__":
    # load data
    m1z_PE = np.load(paths.data / "gw_data/m1z_PE.npy")
    dL_PE = np.load(paths.data / "gw_data/dL_PE.npy")
    log_PE_prior = np.load(paths.data / "gw_data/log_PE_prior.npy")

    # load injection set
    m1zinj_det = np.load(paths.data / "gw_data/m1zinj_det.npy")
    dLinj_det = np.load(paths.data / "gw_data/dLinj_det.npy")
    log_pinj_det = np.load(paths.data / "gw_data/log_pinj_det.npy")

    # Penalized complexity priors on the hyper-hyper parameters
    scale, concentration, L = get_ell_frechet_params(np.log(m1z_PE).mean(axis=1),return_L=True)
    conc, lam_sigma = get_sigma_gamma_params(U=2.)
    
    # Inference
    nuts_settings = dict(target_accept_prob=0.9, max_tree_depth=10,dense_mass=False)
    nuts_kernel = numpyro.infer.NUTS(hyper_prior,**nuts_settings)
    kwargs = dict(m1det=m1z_PE,dL=dL_PE, m1det_inj=m1zinj_det,dL_inj=dLinj_det,
                    log_pinj=log_pinj_det, log_PE_prior=log_PE_prior,
                    remove_low_Neff=remove_low_Neff)
    mcmc = numpyro.infer.MCMC(nuts_kernel,num_warmup=NSAMPS,num_samples=NSAMPS,
                              num_chains=1,progress_bar=True)   
    mcmc.run(jax_rng,**kwargs)

    # save results
    id = az.from_numpyro(mcmc)
    id.to_netcdf(paths.data / "mcmc_nonparametric.nc4")
