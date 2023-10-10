# custom
import jgwcosmo # for inference
import jgwpop
import paths
Clight = jgwcosmo.Clight
from data_generation import NUM_INJ, OM0_FID, H0_FID, ZMAX, ZMIN, N_SAMPLES_PER_EVENT

# inference
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import arviz as az

# data
import numpy as np

jax.config.update("jax_enable_x64", True)

# Constants
TEST_Z = jnp.linspace(ZMIN,ZMAX,num=100)
TEST_M1S = jnp.linspace(0,250.,num=500)

H0_PRIOR_MIN=30.
H0_PRIOR_MAX=120.

# random number generators
jax_rng = jax.random.PRNGKey(42)

# options
remove_low_Neff=False

def model(m1det,dL,m1det_inj,dL_inj,log_pinj,log_PE_prior=0.,remove_low_Neff=False):
    mean = numpyro.sample("mean",dist.Normal(0,3))
    sigma = numpyro.deterministic("sigma",2.)
    rho = numpyro.deterministic("rho",3.)

    H0 = numpyro.sample("H0",dist.Uniform(H0_PRIOR_MIN,H0_PRIOR_MAX))
    Om0=OM0_FID

    # construct mass dist in the source frame
    log_rate = 

    # convert event data to source frame
    z = jgwcosmo.z_at_dl_approx(dL,H0,Om0,zmin=ZMIN,zmax=ZMAX+8.)
    m1source = m1det / (1 + z)
    log_jac = - jnp.log1p(z) - jnp.log(jnp.abs(jgwcosmo.dDLdz_approx(z,H0,Om0))) 

    # convert injections to source frame 
    z_injs = jgwcosmo.z_at_dl_approx(dL_inj,H0,Om0,zmin=ZMIN,zmax=ZMAX+8.)
    m1_injs = m1det_inj / (1 + z_injs)

    # evaluate z dist on data
    # make z dist taper to zero outside of injection bounds
    z_taper = - jnp.log(1.+(z/z_injs.min())**(-15.))
    p_z = jgwpop.unif_comoving(z,H0,Om0)

    # total population is mass dist times redshift distribution 
    # times jacobians for the transformation
    # this is the only line that is different for the hierarchical problem
    single_event_logL = jax.scipy.special.logsumexp(log_rate + jnp.log(p_z) + z_taper
                                                    + log_jac - log_PE_prior,
        axis=1) - jnp.log(N_SAMPLES_PER_EVENT)
    numpyro.factor("logp",jnp.sum(single_event_logL))
    
    # evaluate the population for injections 
    log_rate_injs = 
    log_jac_injs = 2*jnp.log1p(z_injs) + jnp.log(jnp.abs(jgwcosmo.dDLdz_approx(z_injs,H0,Om0))) # try negative
    p_z_injs = jgwcosmo.diff_comoving_volume_approx(z_injs,H0,Om0)
    z_taper_injs =  - jnp.log(1.+(z_injs/z_injs.min())**(-15.))
    # MC integral for the expected number of events
    log_weights = log_rate_injs + jnp.log(p_z_injs) + z_taper_injs - (log_pinj + log_jac_injs)
    Nexp = jnp.sum(jnp.exp(log_weights))/NUM_INJ
    numpyro.factor("Nexp",-1*Nexp)
    numpyro.deterministic("nexp",Nexp)

    # check for convergence with the effecive number of injections
    Neff = jnp.sum(jnp.exp(log_weights))**2/jnp.sum(jnp.exp(log_weights)**2)
    numpyro.deterministic('neff', Neff)

    if remove_low_Neff:
        numpyro.factor("Neff_inj_penalty",jnp.log(1./(1.+(Neff/(4.*len(single_event_logL)))**(-30.))))

if  __name__ == "__main__":
    # load data
    m1z_PE = np.save(paths.data / "gw_data/m1z_PE.npy")
    dL_PE = np.save(paths.data / "gw_data/dL_PE.npy")
    log_PE_prior = np.save(paths.data / "gw_data/log_PE_prior.npy")

    # load injection set
    m1zinj_det = np.save("path.data" / "gw_data/m1zinj_det.npy")
    dLinj_det = np.save("path.data" / "gw_data/dLinj_det.npy")
    log_pinj_det = np.save("path.data" / "gw_data/log_pinj_det.npy")

    # Inference
    nuts_settings = dict(target_accept_prob=0.9, max_tree_depth=10,dense_mass=False)
    nuts_kernel = numpyro.infer.NUTS(model,**nuts_settings)
    kwargs = dict(m1det=m1z_PE,dL=dL_PE, m1det_inj=m1zinj_det,dL_inj=dLinj_det,
                    log_pinj=log_pinj_det, log_PE_prior=log_PE_prior,
                    remove_low_Neff=remove_low_Neff)
    mcmc = numpyro.infer.MCMC(nuts_kernel,num_warmup=100,num_samples=100,
                              num_chains=1,progress_bar=True)   
    mcmc.run(jax_rng,**kwargs)

    # save results
    id = az.from_numpyro(mcmc)
    id.to_netcdf(paths.data / "mcmc_parametric.nc4")
