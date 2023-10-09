# custom
import jgwcosmo # for inference
import jgwpop
import gwcosmo # for data generation
import gwpop
import paths
Clight = jgwcosmo.Clight
from utils import inverse_transform_sample
from mock_posteriors import gen_snr_scaled_PE

# inference
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from tinygp import kernels, GaussianProcess
from tinygp.solvers import QuasisepSolver
import arviz as az

# data generation
import numpy as np
from GWMockCat.vt_utils import draw_thetas, interpolate_optimal_snr_grid 


jax.config.update("jax_enable_x64", True)

# Constants
H0_FID = 67.66
DH_FID = (Clight/1.0e3) / H0_FID
OM0_FID = 0.3096
ZMAX = 10.
ZMIN = 1e-6 # can't be zero since we will eventually take the log of it
TEST_Z = jnp.linspace(ZMIN,ZMAX,num=100)
TEST_M1S = jnp.linspace(0,250.,num=500)

H0_PRIOR_MIN=30.
H0_PRIOR_MAX=120.

N_SOURCES = 10000
N_SAMPLES_PER_EVENT = 100
SIGMA_M = 0.5
SIGMA_DL = 1000
SNR_THRESH=8.
NUM_INJ=N_SOURCES*50

# random number generators
jax_rng = jax.random.PRNGKey(42)
np_rng = np.random.default_rng(4242)

# options
remove_low_Neff=False

def true_vals_feature_full(rng, num_ridges=3, mmin=5, mmax=100,
                        n_sources=N_SOURCES,zmax=ZMAX):
    assert N_SOURCES % num_ridges == 0 , "num_ridges must be a divisor of N_SOURCES"
    num_sources_per_ridge = int(N_SOURCES/num_ridges)
    ridge_width = ((mmax - mmin)/num_ridges )/2
    ridge_leftedge = np.linspace(mmin,mmax-ridge_width,num=num_ridges)
    mass_samples = []
    for ridge in range(num_ridges):
        mass_samples.append(rng.uniform(
            low=ridge_leftedge[ridge],
            high=ridge_leftedge[ridge]+ridge_width,
            size=num_sources_per_ridge
            )
        )
    m1s_true = np.array(mass_samples).flatten()
    zt = inverse_transform_sample(gwpop.unif_comoving_rate, [1e-3,ZMAX], rng,
                              N=N_SOURCES, H0=H0_FID,Om0=OM0_FID)
    m1z_true = m1s_true * (1+zt)
    dL_true = gwcosmo.dL_approx(zt,H0_FID,OM0_FID)
    return m1s_true, zt, m1z_true, dL_true

def make_injections(rng, alpha, mmax_inj, mmin_inj, zmax_inj=ZMAX,num_inj=NUM_INJ):
    m1zinj = inverse_transform_sample(gwpop.powerlaw, [mmin_inj,mmax_inj],rng, N=num_inj, 
                                 alpha=-alpha, high=mmax_inj, low=mmin_inj)
    p_inj = gwpop.powerlaw(m1zinj,-alpha,mmax_inj,mmin_inj )
    log_pinj = -alpha * np.log(m1zinj) + np.log(
        (-alpha + 1)/(mmax_inj**(-alpha + 1)-mmin_inj**(-alpha + 1)))
    dLinj = gwcosmo.dL_approx(inverse_transform_sample(
        gwpop.unif_comoving_rate, [ZMIN,zmax_inj], rng, N=num_inj, H0=H0_FID, Om0=OM0_FID),
          H0=H0_FID, Om0=OM0_FID)
    z_integrand = np.linspace(ZMIN,zmax_inj,num=500)
    pdl = np.abs(gwpop.unif_comoving_rate(gwcosmo.z_at_dl_approx(dLinj,H0=H0_FID, Om0=OM0_FID,zmin=ZMIN,zmax=50),H0=H0_FID,Om0=OM0_FID # p(z)
        )/np.trapz(y=gwcosmo.diff_comoving_volume_approx(z_integrand,H0=H0_FID, Om0=OM0_FID),x=z_integrand # normalization
        )/gwcosmo.dDLdz_approx(gwcosmo.z_at_dl_approx(dLinj,H0_FID,OM0_FID,zmin=ZMIN,zmax=50), H0=H0_FID, Om0=OM0_FID) # convert to p(dl)
    )
    log_pinj += np.log(pdl)

    return m1zinj, dLinj, log_pinj

def model(m1det,dL,m1det_inj,dL_inj,log_pinj,log_PE_prior=0.,remove_low_Neff=False):
    mean = numpyro.sample("mean",dist.Normal(0,3))
    sigma = numpyro.deterministic("sigma",2.)
    rho = numpyro.deterministic("rho",3.)

    H0 = numpyro.sample("H0",dist.Uniform(H0_PRIOR_MIN,H0_PRIOR_MAX))
    Om0=OM0_FID

    # construct GP in the source frame
    kernel = sigma**2 * kernels.quasisep.Matern52(rho) # can change kernel type
    gp = GaussianProcess(kernel,TEST_M1S,mean=mean,diag=0.001,
                         solver=QuasisepSolver,assume_sorted=True)
    log_rate_test = numpyro.sample("log_rate_test",gp.numpyro_dist())

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

    # interpolate GP
    log_rate = numpyro.deterministic("log_rate", jnp.interp(m1source,
                    TEST_M1S,log_rate_test, left=-jnp.inf, right=-jnp.inf)
                    )
    # total population is interpolated GP times redshift distribution 
    # times jacobians for the transformation
    # this is the only line that is different for the hierarchical problem
    single_event_logL = jax.scipy.special.logsumexp(log_rate + jnp.log(p_z) + z_taper
                                                    + log_jac - log_PE_prior,
        axis=1) - jnp.log(N_SAMPLES_PER_EVENT)
    numpyro.factor("logp",jnp.sum(single_event_logL))
    
    # evaluate the population for injections 
    log_rate_injs = jnp.interp(m1_injs, TEST_M1S, log_rate_test, 
                               left=-jnp.inf, right=-jnp.inf)
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
    # generate data
    m1s_true, zt, m1z_true, dL_true = true_vals_feature_full(rng=np_rng, num_ridges=2)
    # generate injection set
    m1zinj, dLinj, log_pinj = make_injections(np_rng,alpha=0.5,mmax_inj=350., mmin_inj=1.5)
    # select injections and data based off of an SNR threshold
    osnr_interp, reference_distance = interpolate_optimal_snr_grid(
        fname=paths.data+"/optimal_snr_aplus_design_O5.h5") # TODO: put this on zenodo as a static dataset
    ## find injections
    thetas_inj = draw_thetas(len(dLinj),rng=np_rng)
    snr_true_inj = osnr_interp(m1zinj, m1zinj, grid=False)/dLinj * thetas_inj * 1000.
    snr_obs_inj = snr_true_inj + 1. * np_rng.normal(size=len(dLinj))
    m1zinj_det = m1zinj[snr_obs_inj>SNR_THRESH]
    dLinj_det = dLinj[snr_obs_inj>SNR_THRESH]
    log_pinj_det = log_pinj[snr_obs_inj>SNR_THRESH]
    ## find events and generate mock PE
    m1z_PE, dL_PE, log_PE_prior = gen_snr_scaled_PE(np_rng,m1s_true,dL_true/1000,osnr_interp,
                                                            reference_distance,N_SAMPLES_PER_EVENT,H0_FID,OM0_FID)
    dL_PE *= 1000 # unit matching

    # Inference
    nuts_settings = dict(target_accept_prob=0.9, max_tree_depth=10,dense_mass=False)
    nuts_kernel = numpyro.infer.NUTS(model,**nuts_settings)
    kwargs = dict(m1det=m1z_PE,dL=dL_PE, m1det_inj=m1zinj_det,dL_inj=dLinj_det,
                    log_pinj=log_pinj_det, log_PE_prior=log_PE_prior,
                    remove_low_Neff=remove_low_Neff)
    mcmc = numpyro.infer.MCMC(nuts_kernel,num_warmup=1000,num_samples=1000,
                              num_chains=1,progress_bar=True)   
    mcmc.run(jax_rng,**kwargs)

    # save results
    id = az.from_numpyro(mcmc)
    id.to_netcdf(paths.data+"/mcmc_nonparametric.nc4")
