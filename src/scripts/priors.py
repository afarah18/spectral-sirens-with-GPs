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
from data_generation import NUM_INJ, OM0_FID, ZMAX, ZMIN, N_SAMPLES_PER_EVENT
import jgwcosmo
import jgwpop

## CONSTANTS
TEST_M1S = jnp.linspace(0,250.,num=500)

H0_PRIOR_MIN=30.
H0_PRIOR_MAX=120.

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
def hyper_prior(m1det,dL,m1det_inj,dL_inj,log_pinj,log_PE_prior=0.,remove_low_Neff=False,fit_Om0=False):
    mean = numpyro.sample("mean",dist.Normal(0,3))
    sigma = numpyro.deterministic("sigma",2.)
    rho = numpyro.deterministic("rho",3.)

    H0 = numpyro.sample("H0",dist.Uniform(H0_PRIOR_MIN,H0_PRIOR_MAX))
    if fit_Om0:
        Om0 = numpyro.sample("Om0",dist.Uniform(0.01,0.99))
    else:
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
    p_z = jgwpop.unif_comoving_rate(z,H0,Om0)

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
    # some tests
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