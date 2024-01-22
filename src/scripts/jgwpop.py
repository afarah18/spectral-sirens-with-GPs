import jax.numpy as jnp
from jutils import powerlaw, truncnorm
from jgwcosmo import diff_comoving_volume_approx, log_diff_comoving_volume_approx

""" Mass distribution """
def powerlaw_peak(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak):
    tmp_min = 2.
    tmp_max = 150.
    dmMax = 2
    dmMin = 1

    # Define power-law and peak
    p_m1_pl = powerlaw(m1,low=tmp_min,high=tmp_max,alpha=alpha)
    p_m1_peak = jnp.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/jnp.sqrt(2.*jnp.pi*sig_m1**2)

    # Compute low- and high-mass filters
    low_filter = jnp.exp(-(m1-mMin)**2/(2.*dmMin**2))
    low_filter = jnp.where(m1<mMin,low_filter,1.)
    high_filter = jnp.exp(-(m1-mMax)**2/(2.*dmMax**2))
    high_filter = jnp.where(m1>mMax,high_filter,1.)

    # Apply filters to combined power-law and peak
    return (f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl)*low_filter*high_filter

def powerlaw_smooth(m,mMin,mMax,alpha):
    tmp_min = 2.
    tmp_max = 150.
    dmMax = 1
    dmMin = 1

    pm_pl = powerlaw(m,tmp_min,tmp_max,alpha)

    # Compute low- and high-mass filters
    low_filter = jnp.exp(-(m-mMin)**2/(2.*dmMin**2))
    low_filter = jnp.where(m<mMin,low_filter,1.)
    high_filter = jnp.exp(-(m-mMax)**2/(2.*dmMax**2))
    high_filter = jnp.where(m>mMax,high_filter,1.)

    return pm_pl*low_filter*high_filter

def logpowerlaw(m,mMin,mMax,alpha):
    lognorm = jnp.log(1. + alpha) - jnp.log(mMax**(alpha+1.) - mMin**(alpha+1.))
    return lognorm + alpha * jnp.log(m)

def logbroken_powerlaw(m1,mMin,mMax,alpha1,alpha2,b_m1):
    dmMax = 2
    dmMin = 1
    
    # Define power-law and peak
    # p_m1_pl1 = powerlaw(m1/b_m1,low=tmp_min,high=b_m1,alpha=alpha1)
    # p_m1_pl2 = powerlaw(m1/b_m1,low=b_m1,high=tmp_max,alpha=alpha2)
    # logpl=jnp.log(jnp.where(m1<b_m1,p_m1_pl1,p_m1_pl2))
    logpl = jnp.where(m1<b_m1,alpha1*(jnp.log(m1)-jnp.log(b_m1)), alpha2*(jnp.log(m1)-jnp.log(b_m1)))

    # Compute low- and high-mass filters
    loglow_filter = -(m1-mMin)**2/(2.*dmMin**2)
    loglow_filter = jnp.where(m1<mMin,loglow_filter,0.)
    loghigh_filter = -(m1-mMax)**2/(2.*dmMax**2)
    loghigh_filter = jnp.where(m1>mMax,loghigh_filter,0.)

    # Apply filters to combined power-law and peak
    return logpl + loghigh_filter + loglow_filter

def logpowerlaw_peak(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak):
    tmp_min = 2.
    tmp_max = 400.
    dmMax = 2
    dmMin = 1
    
    # Define power-law and peak
    p_m1_pl = powerlaw(m1, alpha=alpha, high=tmp_max, low=tmp_min)
    p_m1_peak = truncnorm(m1, mu=mu_m1, sigma=sig_m1, high=tmp_max, low=tmp_min) #jnp.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/jnp.sqrt(2.*jnp.pi*sig_m1**2)

    # Compute low- and high-mass filters
    loglow_filter = -(m1-mMin)**2/(2.*dmMin**2)
    loglow_filter = jnp.where(m1<mMin,loglow_filter,0.)
    loghigh_filter = -(m1-mMax)**2/(2.*dmMax**2)
    loghigh_filter = jnp.where(m1>mMax,loghigh_filter,0.)

    # Apply filters to combined power-law and peak
    return jnp.log(f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl) + loghigh_filter + loglow_filter

""" Redshift distribution """
def rconst(z):
    return 1.
def sfr(z):
    return 0.015*(1.+z)**2.7/(1.+((1.+z)/2.9)**5.6) #msun per yr per Mpc^3
def rsfr(z):
    return sfr(z)/sfr(0.)

def rate_z(z,zp,alpha,beta):
    c0 = 1. + (1. + zp)**(-alpha-beta)
    num = (1.+z)**alpha
    den = 1. + ((1.+z)/(1.+zp))**(alpha+beta)
    return c0 * num / den

def log_shouts_murmurs(z,zp,alpha,beta):
    logc0 = jnp.log1p((1. + zp)**(-alpha-beta))
    return logc0  + alpha*jnp.log1p(z) - jnp.log1p(jnp.power((1.+z)/(1.+zp),(alpha+beta)))

def unif_comoving_rate(z,H0,Om0):
    dvc_dz = diff_comoving_volume_approx(z,H0,Om0)
    dtsrc_dtdet = (1 + z)
    return dvc_dz / dtsrc_dtdet

def log_unif_comoving_rate(z,H0,Om0):
    log_dvc_dz = log_diff_comoving_volume_approx(z,H0,Om0)
    return log_dvc_dz - jnp.log1p(z)