""" All modeling done here. 
Penalized complexity priors for inferring GP parameters, plus the GP-based population model itself """
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    validate_sample, promote_shapes, is_prng_key
)

import jax
from jax import core, lax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from tinygp import kernels, GaussianProcess
from tinygp.solvers import QuasisepSolver
from jax.scipy.special import gammaln

# custom
from data_generation import NUM_INJ, H0_FID, OM0_FID, ZMAX, ZMIN, N_SAMPLES_PER_EVENT, TRUEVALS
import jgwcosmo
import jgwpop
import paths

## CONSTANTS
TEST_M1S = jnp.linspace(0.1,250.,num=500)

H0_PRIOR_MIN=30.
H0_PRIOR_MAX=120.
# make a dummy variable so that the PDF links to this module
with open(paths.output / "priors_placeholder.txt","w") as f:
        print(" ",file=f)

## LENGTH SCALE PRIOR
## Applying: https://dansblog.netlify.app/posts/2022-09-07-priors5/priors5.html#rescuing-the-pc-prior-on-ell-or-what-i-recommend-you-do
## According to https://en.wikipedia.org/wiki/Weibull_distribution, the Frechet distribution is just the Weibull distribution with negative k.
##

class Frechet(Distribution):
    arg_constraints = {
        "scale": constraints.positive,
        "concentration": constraints.real,
    }
    support = constraints.positive
    reparametrized_params = ["scale", "concentration"]

    def __init__(self, scale, concentration, *, validate_args=None):
        self.concentration, self.scale = promote_shapes(concentration, scale)
        batch_shape = lax.broadcast_shapes(jnp.shape(concentration), jnp.shape(scale))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return jax.random.weibull_min(
            key,
            scale=self.scale,
            concentration=-self.concentration,
            shape=sample_shape + self.batch_shape,
        )

    @validate_sample
    def log_prob(self, value):
        """https://en.wikipedia.org/wiki/Weibull_distribution#Related_distributions"""
        k = self.concentration
        ll = -jnp.power(value / self.scale, -k)
        ll += jnp.log(k)
        ll -= (k + 1.0) * jnp.log(value)
        ll += k * jnp.log(self.scale)
        return ll

    def cdf(self, value):
        return jnp.exp(-((value / self.scale) ** -self.concentration))

    @property
    def mean(self):
        return self.scale * jnp.exp(gammaln(1.0 - 1.0 / self.concentration))

    @property
    def median(self):
        return self.scale/jnp.log(2)**(1/self.concentration)

    @property
    def variance(self):
        var = jnp.where(self.concentration >2.,
                        self.scale**2 * (
                            jnp.exp(gammaln(1.0 - 2.0 / self.concentration))
                            - jnp.exp(gammaln(1.0 - 1.0 / self.concentration)) ** 2
                        ),
                        jnp.inf)
        return var

def get_ell_frechet_params(data,dims=1.,alpha=0.05, return_L=False):
    concentration = dims/2.
    # we choose the lower bound for the length scale to be the characteristic
    # distance between datapoints. we could also choose it to be the minimum
    # distance between datapoints, but this choice makes inference faster
    L = jnp.mean(jnp.diff(jnp.sort(data)))
    lam = -jnp.log(alpha) * (L**concentration)
    scale = lam**(2./dims)
    if return_L:
        return scale, concentration, L
    else:
        return scale, concentration

# VARIANCE PRIOR
# Applying: https://dansblog.netlify.app/posts/2022-09-07-priors5/priors5.html#a-first-crack-at-a-pc-prior
# According to https://en.wikipedia.org/wiki/Gamma_distribution the prior on sigma is basically a Gamma distribution
# with concentration k=1 and scale parameter theta=1/lambda (lambda is defined in the blog post)
# numpyro uses a different parameterization for the Gamma distribution, so we
# have to convert k and theta to those params (shape, rate)

def get_sigma_gamma_params(U,alpha=0.05):
    k=1.
    lam = - jnp.log(alpha)/U # I think the blog post is missing this minus sign
    #theta = 1./lam
    rate = lam
    return k, rate

# HYPER PRIOR / POPULATION MODEL
def hyper_prior(m1det,dL,m2det,m1det_inj,q_inj,dL_inj,log_pinj,log_PE_prior=0.,PC_params=dict(),remove_low_Neff=False,fit_Om0=False):
    """ Non-parametric population inference """
    mean = numpyro.sample("mean",dist.Normal(0,3))
    sigma = numpyro.deterministic("sigma",2.5)
    # sigma = numpyro.sample("sigma",dist.Gamma(concentration=PC_params["conc"], rate=PC_params["lam_sigma"]))
    # rho = numpyro.deterministic("rho",0.5)
    rho = numpyro.sample("rho",Frechet(concentration=PC_params["concentration"],scale=PC_params["scale"]))

    H0 = numpyro.sample("H0",dist.Uniform(H0_PRIOR_MIN,H0_PRIOR_MAX))
    if fit_Om0:
        Om0 = numpyro.sample("Om0",dist.Uniform(0.01,0.99))
    else:
        Om0=OM0_FID

    # redshift dist
    alpha_z = TRUEVALS['alpha_z']#numpyro.sample("alpha_z",dist.Uniform(0,2)) #
    zp = TRUEVALS['zp']#numpyro.sample("zp",dist.Uniform(0,5)) #
    beta_z =  TRUEVALS['beta_z'] # numpyro.deterministic('beta',TRUEVALS['beta_z'])# numpyro.sample("beta_z",dist.Uniform(0,5)) #

    # construct GP in the source frame
    logtestm1s = jnp.log(TEST_M1S)
    kernel = sigma**2 * kernels.quasisep.Matern52(rho) # can change kernel type
    gp = GaussianProcess(kernel,logtestm1s,mean=mean,diag=0.001,
                         solver=QuasisepSolver,assume_sorted=True)
    log_rate_test = numpyro.sample("log_rate_test",gp.numpyro_dist())

    # convert event data to source frame
    z = jgwcosmo.z_at_dl_approx(dL,H0,Om0,zmin=ZMIN,zmax=ZMAX+8.)
    logm1source = jnp.log(m1det) - jnp.log1p(z)
    q = m2det/m1det
    
    # convert injections to source frame 
    z_injs = jgwcosmo.z_at_dl_approx(dL_inj,H0,Om0,zmin=ZMIN,zmax=ZMAX+8.)
    logm1_injs = jnp.log(m1det_inj) - jnp.log1p(z_injs)

    # evaluate z dist on data
    logcosmo_data = jgwpop.log_unif_comoving_rate(z,H0,Om0)
    logRzs_data = jgwpop.log_shouts_murmurs(z,zp,alpha_z,beta_z)
    # make z dist taper to zero outside of injection bounds
    z_taper_data = - jnp.log(1.+(z/z_injs.min())**(-30.)) #- jnp.log(1.+(z_injs.max()/z)**(-30.))
    log_Jacobian_dL_z_data = - jnp.log(jgwcosmo.dDLdz_approx(z,H0,Om0))
    log_pz_data = logcosmo_data + logRzs_data + z_taper_data + log_Jacobian_dL_z_data

    # evaluate z dist on injections
    logcosmo_injs = jgwpop.log_unif_comoving_rate(z_injs,H0,Om0)
    logRzs_injs = jgwpop.log_shouts_murmurs(z_injs,zp,alpha_z,beta_z)
    # make z dist taper to zero outside of injection bounds
    z_taper_injs =  - jnp.log(1.+(z_injs/z_injs.min())**(-30.)) #- jnp.log(1.+(z_injs.max()/z_injs)**(-30.))
    log_Jacobian_dL_z_injs = - jnp.log(jgwcosmo.dDLdz_approx(z_injs,H0,Om0))
    log_pz_injs = logcosmo_injs + logRzs_injs + z_taper_injs + log_Jacobian_dL_z_injs

    # evaluate mass dist on data
    log_rate_m1s_data = jnp.interp(logm1source,logtestm1s,log_rate_test-logtestm1s,left=-jnp.inf,right=-jnp.inf)
    log_Jacobian_m1z_m1s_data = - jnp.log1p(z)
    log_pq_data = 0. # jnp.log(jgwpop.truncnorm(q,mu=0.9,sigma=0.05,high=1,low=0))#jgwpop.logpowerlaw(q,mMin=0.0001,mMax=1.,alpha=bq)#,mMin=mmin,mMax=m1_injs,alpha=bq,f_peak=0.,mu_m1=25.,sig_m1=5.)
    logJacobian_m2z_q_data = - jnp.log(m1det)
    log_rate_m1_data = log_rate_m1s_data + log_pq_data + log_Jacobian_m1z_m1s_data  + logJacobian_m2z_q_data

    # evaluate mass dist on injections
    log_rate_m1s_injs = jnp.interp(logm1_injs,logtestm1s,log_rate_test-logtestm1s,left=-jnp.inf,right=-jnp.inf)
    log_Jacobian_m1z_m1s_injs = - jnp.log1p(z_injs)
    log_pq_injs = 0. # jnp.log(jgwpop.truncnorm(q_inj,mu=0.9,sigma=0.05,high=1,low=0))#jgwpop.logpowerlaw(q,mMin=0.0001,mMax=1.,alpha=bq)#,mMin=mmin,mMax=m1_injs,alpha=bq,f_peak=0.,mu_m1=25.,sig_m1=5.)
    # pdraw is in terms of q already, so no need for an extra jacobian
    log_rate_m1_injs = log_rate_m1s_injs + log_pq_injs + log_Jacobian_m1z_m1s_injs 

    # event part of likelihood
    single_event_logL = jax.scipy.special.logsumexp(log_rate_m1_data + log_pz_data - log_PE_prior,
        axis=1) - jnp.log(N_SAMPLES_PER_EVENT)
    numpyro.factor("logp",jnp.sum(single_event_logL))

    # injection part of likelihood
    log_weights = log_rate_m1_injs + log_pz_injs - log_pinj
    Nexp = jnp.sum(jnp.exp(log_weights))/NUM_INJ
    numpyro.factor("Nexp",-1*Nexp)
    numpyro.deterministic("nexp",Nexp)

    # check for convergence with the effecive number of injections
    Neff = jnp.sum(jnp.exp(log_weights))**2/jnp.sum(jnp.exp(log_weights)**2)
    numpyro.deterministic('neff', Neff)

    if remove_low_Neff:
        numpyro.factor("Neff_inj_penalty",jnp.log(1./(1.+(Neff/(4.*len(single_event_logL)))**(-30.))))

def PLP(m1det,dL,m2det,m1det_inj,q_inj,dL_inj,log_pinj,log_PE_prior=0.,remove_low_Neff=False,fit_Om0=False):
    """Correct parametric population inference"""
    # cosmology
    H0 = numpyro.sample("H0",dist.Uniform(H0_PRIOR_MIN,H0_PRIOR_MAX))
    # H0 = numpyro.deterministic("H0",H0_FID)
    if fit_Om0:
        Om0 = numpyro.sample("Om0",dist.Uniform(0.01,0.99))
    else:
        Om0=OM0_FID

    rate = numpyro.sample('rate',dist.TruncatedNormal(loc=10.,scale=30.,low=0.01))

    # mass dist
    mmin = numpyro.sample("mmin",dist.Uniform(1.0,20.0))
    mmax = numpyro.sample("mmax",dist.Uniform(30.0,100.0))
    alpha = numpyro.sample("alpha",dist.Normal(0,5))
    mu_m1 = numpyro.sample("mu_m1",dist.Uniform(20,60))# TRUEVALS['mu_m1']
    sig_m1 = numpyro.sample("sig_m1",dist.Uniform(1,10))# TRUEVALS['sig_m1']
    f_peak = TRUEVALS['f_peak'] #numpyro.sample("f_peak",dist.Uniform(0,1))
    # bq = 0# numpyro.sample("bq",dist.Normal(0,5))

    # Fixed
    alpha_z = TRUEVALS['alpha_z']# numpyro.sample("alpha_z",dist.Uniform(0,2)) #
    zp = TRUEVALS['zp']#numpyro.sample("zp",dist.Uniform(0,5)) #
    beta = TRUEVALS['beta_z'] #numpyro.sample("beta_z",dist.Uniform(0,5)) # numpyro.deterministic('beta',TRUEVALS['beta_z'])# 

    # convert event data to source frame
    z = jgwcosmo.z_at_dl_approx(dL,H0,Om0,zmin=ZMIN,zmax=ZMAX+8.)
    m1source = m1det / (1 + z)
    q = m2det/m1det

    # convert injections to source frame 
    z_injs = jgwcosmo.z_at_dl_approx(dL_inj,H0,Om0,zmin=ZMIN,zmax=ZMAX+8.)
    m1_injs = m1det_inj / (1 + z_injs)

    # evaluate z dist on data
    logcosmo_data = jgwpop.log_unif_comoving_rate(z,H0=H0,Om0=Om0)
    logRzs_data = jgwpop.log_shouts_murmurs(z=z,zp=zp,alpha=alpha_z,beta=beta)
    z_taper_data = 0. #- jnp.log(1.+(z/z_injs.min())**(-15.))
    logJacobian_dL_z_data = - jnp.log(jnp.abs(jgwcosmo.dDLdz_approx(z,H0,Om0))) 
    log_pz_data = logcosmo_data + logRzs_data + z_taper_data + logJacobian_dL_z_data

    # evaluate z dist on injs
    logcosmo_injs = jgwpop.log_unif_comoving_rate(z_injs,H0=H0,Om0=Om0)
    logRzs_injs = jgwpop.log_shouts_murmurs(z=z_injs,zp=zp,alpha=alpha_z,beta=beta)
    z_taper_injs = 0. #- jnp.log(1.+(z_injs/z_injs.min())**(-15.))
    logJacobian_dL_z_injs = - jnp.log(jnp.abs(jgwcosmo.dDLdz_approx(z_injs,H0,Om0))) 
    log_pz_injs = logcosmo_injs + logRzs_injs + z_taper_injs + logJacobian_dL_z_injs

    # evaluate mass dist on data - final mass dist is in terms of m1z and m2z
    log_pm1_data = jgwpop.logpowerlaw_peak(m1source,mmin,mmax,alpha,sig_m1,mu_m1,f_peak)
    log_pq_data = 0. # jnp.log(jgwpop.truncnorm(q,mu=0.9,sigma=0.05,high=1,low=0))# jgwpop.logpowerlaw(q,mMin=0.0001,mMax=1.,alpha=bq)#,mMin=mmin,mMax=m1_injs,alpha=bq,f_peak=0.,mu_m1=25.,sig_m1=5.) 
    logJacobian_m1z_m1_data = - 1.0*jnp.log1p(z)
    logJacobian_m2z_q_data = - 1.0*jnp.log(m1det)
    log_pm_data = log_pm1_data + log_pq_data + logJacobian_m1z_m1_data + logJacobian_m2z_q_data

    # evaluate mass dist on injs - final mass dist is in terms of m1s and q
    log_pm1_injs = jgwpop.logpowerlaw_peak(m1_injs,mmin,mmax,alpha,sig_m1,mu_m1,f_peak)
    log_pq_injs =  0. # jnp.log(jgwpop.truncnorm(q_inj,mu=0.9,sigma=0.05,high=1,low=0)) # jgwpop.logpowerlaw(q_inj,mMin=0.0001,mMax=1.,alpha=bq)#,mMin=mmin,mMax=m1_injs,alpha=bq,f_peak=0.,mu_m1=25.,sig_m1=5.)
    logJacobian_m1z_m1_injs = - 1.0*jnp.log1p(z_injs)
    # pdraw is in terms of q already, so no need for an extra jacobian
    log_pm_injs = log_pm1_injs + log_pq_injs + logJacobian_m1z_m1_injs

    # event part of likelihood
    single_event_logL = jax.scipy.special.logsumexp(log_pm_data + log_pz_data +jnp.log(rate) - log_PE_prior,
                                                    axis=1) - jnp.log(N_SAMPLES_PER_EVENT)
    numpyro.factor("logp",jnp.sum(single_event_logL))

    # injection part of likelihood
    log_weights = log_pm_injs + log_pz_injs + jnp.log(rate) - log_pinj
    Nexp = jnp.sum(jnp.exp(log_weights))/NUM_INJ
    numpyro.factor("Nexp",-1*Nexp)
    numpyro.deterministic("nexp",Nexp)
    

    # check for convergence with the effecive number of injections
    Neff = jnp.sum(jnp.exp(log_weights))**2/jnp.sum(jnp.exp(log_weights)**2)
    numpyro.deterministic('neff', Neff)
    if remove_low_Neff:
        numpyro.factor("Neff_inj_penalty",jnp.log(1./(1.+(Neff/(4.*len(single_event_logL)))**(-30.))))    
    # for plotting
    numpyro.deterministic("log_rate", jnp.log(rate) + jgwpop.logpowerlaw_peak(TEST_M1S,mmin,mmax,alpha,sig_m1,mu_m1,f_peak))
    
def BPL(m1det,dL,m2det,m1det_inj,q_inj,dL_inj,log_pinj,log_PE_prior=0.,remove_low_Neff=False,fit_Om0=False):
    # cosmology
    H0 = numpyro.sample("H0",dist.Uniform(H0_PRIOR_MIN,H0_PRIOR_MAX))
    # H0 = numpyro.deterministic("H0",H0_FID)
    if fit_Om0:
        Om0 = numpyro.sample("Om0",dist.Uniform(0.01,0.99))
    else:
        Om0=OM0_FID

    rate = numpyro.sample('rate',dist.TruncatedNormal(loc=10.,scale=30.,low=0.01))

    # mass dist
    mmin = numpyro.sample("mmin",dist.Uniform(1.0,20.0))
    mmax = numpyro.sample("mmax",dist.Uniform(30.0,100.0)) # 225
    alpha1 = numpyro.sample("alpha1",dist.Normal(0,5))
    alpha2 = numpyro.sample("alpha2",dist.Normal(0,5))
    b_m1 = numpyro.sample("b_m1",dist.Uniform(20,100))
    
    #Fixed
    # bq = numpyro.sample("bq",dist.Normal(0,5))
    alpha_z = TRUEVALS['alpha_z']#numpyro.sample("alpha_z",dist.Uniform(0,2)) #
    zp = TRUEVALS['zp']#numpyro.sample("zp",dist.Uniform(0,5)) #
    beta = TRUEVALS['beta_z'] # TRUEVALS['beta_z'] # numpyro.deterministic('beta',TRUEVALS['beta_z'])# 

    # convert event data to source frame
    z = jgwcosmo.z_at_dl_approx(dL,H0,Om0,zmin=ZMIN,zmax=ZMAX+8.)
    m1source = m1det / (1 + z)
    q = m2det/m1det

    # convert injections to source frame 
    z_injs = jgwcosmo.z_at_dl_approx(dL_inj,H0,Om0,zmin=ZMIN,zmax=ZMAX+8.)
    m1_injs = m1det_inj / (1 + z_injs)

    # evaluate z dist on data
    logcosmo_data = jgwpop.log_unif_comoving_rate(z,H0,Om0)
    logRzs_data = jgwpop.log_shouts_murmurs(z,zp,alpha_z,beta)
    logJacobian_dL_z_data = - jnp.log(jnp.abs(jgwcosmo.dDLdz_approx(z,H0,Om0))) 
    log_pz_data = logcosmo_data + logRzs_data + logJacobian_dL_z_data

    # evaluate z dist on injs
    logcosmo_injs = jgwpop.log_unif_comoving_rate(z_injs,H0,Om0)
    logRzs_injs = jgwpop.log_shouts_murmurs(z_injs,zp,alpha_z,beta)
    logJacobian_dL_z_injs = - jnp.log(jnp.abs(jgwcosmo.dDLdz_approx(z_injs,H0,Om0))) 
    log_pz_injs = logcosmo_injs + logRzs_injs + logJacobian_dL_z_injs

    # evaluate mass dist on data
    # log_pm1_data = jgwpop.logpowerlaw_peak(m1source,mmin,mmax,alpha,sig_m1,mu_m1,f_peak)
    log_pm1_data = jgwpop.logbroken_powerlaw(m1source,mmin,mmax,alpha1,alpha2,b_m1)
    log_pq_data = 0. #jnp.log(jgwpop.truncnorm(q,mu=0.9,sigma=0.05,high=1,low=0))#jgwpop.logpowerlaw(q,mMin=0.0001,mMax=1.,alpha=bq)#,mMin=mmin,mMax=m1_injs,alpha=bq,f_peak=0.,mu_m1=25.,sig_m1=5.)
    logJacobian_m1z_m1_data = - 1.0*jnp.log1p(z)
    logJacobian_m2z_q_data = - 1.0*jnp.log(m1det)
    log_pm_data = log_pm1_data + log_pq_data + logJacobian_m1z_m1_data + logJacobian_m2z_q_data

    # evaluate mass dist on injs
    # log_pm1_injs = jgwpop.logpowerlaw_peak(m1_injs,mmin,mmax,alpha,sig_m1,mu_m1,f_peak)
    log_pm1_injs = jgwpop.logbroken_powerlaw(m1_injs,mmin,mmax,alpha1,alpha2,b_m1)
    log_pq_injs = 0. # jnp.log(jgwpop.truncnorm(q_inj,mu=0.9,sigma=0.05,high=1,low=0))#jgwpop.logpowerlaw(q_inj,mMin=0.0001,mMax=1.,alpha=bq)#,mMin=mmin,mMax=m1_injs,alpha=bq,f_peak=0.,mu_m1=25.,sig_m1=5.)
    logJacobian_m1z_m1_injs = - 1.0*jnp.log1p(z_injs)
    # pdraw is in terms of q already, so no need for an extra jacobian
    log_pm_injs = log_pm1_injs + log_pq_injs + logJacobian_m1z_m1_injs
    
    # event part of likelihood
    single_event_logL = jax.scipy.special.logsumexp(log_pm_data + log_pz_data +jnp.log(rate) - log_PE_prior,
                                                    axis=1) - jnp.log(N_SAMPLES_PER_EVENT)
    numpyro.factor("logp",jnp.sum(single_event_logL))

    # injection part of likelihood
    log_weights = log_pm_injs  + log_pz_injs +jnp.log(rate) - log_pinj
    Nexp = jnp.sum(jnp.exp(log_weights))/NUM_INJ
    numpyro.factor("Nexp",-1*Nexp)
    numpyro.deterministic("nexp",Nexp)

    # for plotting
    numpyro.deterministic("log_rate",jnp.log(rate) + jgwpop.logbroken_powerlaw(TEST_M1S,mmin,mmax,alpha1,alpha2,b_m1) )

# TESTING
if  __name__ == "__main__":
    from scipy.stats import invweibull
    import numpy as np
    import matplotlib.pyplot as plt
    rng_key = jax.random.PRNGKey(2)
    data = jax.random.normal(key=rng_key,shape=(200,)) * 10. + 50.
    scale, k, L = get_ell_frechet_params(data,dims=1.,alpha=0.01,return_L=True)

    scipy_invweibul = invweibull.rvs(k, scale=scale,size=5000)
    b = np.logspace(np.log10(scipy_invweibul.min()),np.log10(scipy_invweibul.max()),num=50)
    frechet = Frechet(concentration=k, scale=scale).sample(rng_key,sample_shape=(5000,))
    analytic = jnp.exp(Frechet(concentration=k, scale=scale).log_prob(b))
    analytic /= jnp.trapz(analytic,b)
    analytic_CDF = Frechet(concentration=k, scale=scale).cdf(b)

    plt.hist(scipy_invweibul,bins=b,density=True,histtype='step',
             label='scipy Inverse Weibul')
    plt.hist(frechet,bins=b,density=True,histtype='step',
             label='User-defined Frechet')
    plt.plot(b,analytic,label='analytic Frechet PDF')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("Density")
    plt.xlabel("length scale")
    plt.axvline(L)
    plt.show()
    print("If I specified lambda and k correctly, this should be equal to"\
      f" {0.01}: {np.sum(frechet<L)/len(frechet)}")

    plt.plot(b, analytic_CDF)
    plt.hist(frechet,bins=b,cumulative=True, density=True,histtype='step')
    plt.xscale("log")
    plt.ylabel("CDF")
    plt.xlabel("length scale")
    plt.show()
