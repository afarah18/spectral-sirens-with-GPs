# custom
import jgwcosmo # for inference
import paths
Clight = jgwcosmo.Clight
from priors import PLP, BPL
from nonparametric_inference import NSAMPS
from data_generation import H0_FID

# inference
import jax
import numpyro
import arviz as az

# data
import numpy as np

jax.config.update("jax_enable_x64", True)

# random number generators
jax_rng = jax.random.PRNGKey(42)

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

    # Inference - power law peak
    nuts_settings = dict(target_accept_prob=0.9, max_tree_depth=10,dense_mass=False)
    nuts_kernel = numpyro.infer.NUTS(PLP,**nuts_settings)
    kwargs = dict(m1det=m1z_PE,dL=dL_PE, m1det_inj=m1zinj_det,dL_inj=dLinj_det,
                    log_pinj=log_pinj_det, log_PE_prior=log_PE_prior,
                    remove_low_Neff=remove_low_Neff)
    mcmc = numpyro.infer.MCMC(nuts_kernel,num_warmup=NSAMPS,num_samples=NSAMPS,
                              num_chains=1,progress_bar=True)   
    mcmc.run(jax_rng,**kwargs)

    # save results
    id = az.from_numpyro(mcmc)
    id.to_netcdf(paths.data / "mcmc_parametric_PLP_fitz.nc4")
    offset = np.abs(id.posterior['H0'][0].mean()-H0_FID)/id.posterior['H0'][0].std()
    with open(paths.output / "PLPh0offset.txt","w") as f:
        print(f"{offset:.1f}",file=f)
    with open(paths.output / "PLPh0percent.txt","w") as f:
        print(f"{np.std(id.posterior['H0'][0])/np.mean(id.posterior['H0'][0])*100:.1f}",file=f)

    # Inference - broken power law
    nuts_settings = dict(target_accept_prob=0.9, max_tree_depth=10,dense_mass=False)
    nuts_kernel = numpyro.infer.NUTS(BPL,**nuts_settings)
    kwargs = dict(m1det=m1z_PE,dL=dL_PE, m1det_inj=m1zinj_det,dL_inj=dLinj_det,
                    log_pinj=log_pinj_det, log_PE_prior=log_PE_prior,
                    remove_low_Neff=remove_low_Neff)
    mcmc = numpyro.infer.MCMC(nuts_kernel,num_warmup=NSAMPS,num_samples=NSAMPS,
                              num_chains=1,progress_bar=True)   
    mcmc.run(jax_rng,**kwargs)

    # save results
    id = az.from_numpyro(mcmc)
    id.to_netcdf(paths.data / "mcmc_parametric_BPL_fitz.nc4")
    offset = np.abs(id.posterior['H0'][0].mean()-H0_FID)/id.posterior['H0'][0].std()
    with open(paths.output / "BPLh0offset.txt","w") as f:
        print(f"{offset:.1f}",file=f)