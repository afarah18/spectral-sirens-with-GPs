import numpy as np
from utils import powerlaw, truncnorm
from gwcosmo import diff_comoving_volume_approx

""" Mass distribution """

def powerlaw_peak(m1,mMin,mMax,alpha,sig_m1,mu_m1,f_peak):
    tmp_min = 2.
    tmp_max = 400.
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

def powerlaw_smooth(m,mMin,mMax,alpha):
    tmp_min = 2.
    tmp_max = 150.
    dmMax = 1
    dmMin = 1

    pm_pl = powerlaw(m,tmp_min,tmp_max,alpha)

    # Compute low- and high-mass filters
    low_filter = np.exp(-(m-mMin)**2/(2.*dmMin**2))
    low_filter = np.where(m<mMin,low_filter,1.)
    high_filter = np.exp(-(m-mMax)**2/(2.*dmMax**2))
    high_filter = np.where(m>mMax,high_filter,1.)

    return pm_pl*low_filter*high_filter

""" Redshift distribution """
maximum_redshift_considered = 10
minimum_redshift_considered = 0
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

def unif_comoving_rate(z,H0,Om0):
    dvc_dz = diff_comoving_volume_approx(z,H0,Om0)
    dtsrc_dtdet = (1 + z)
    return dvc_dz / dtsrc_dtdet

def shouts_murmurs(z,H0,Om0,zp,alpha,beta):
    unif_comov = unif_comoving_rate(z, H0, Om0)
    c0 = 1 + (1 + zp)**(-alpha - beta)
    return unif_comov * c0  * (1 + z)**alpha / ( 1 + (np.power((1.+z)/(1.+zp),(alpha+beta)))) 
