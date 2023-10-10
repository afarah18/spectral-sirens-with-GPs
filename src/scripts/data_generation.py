import gwcosmo
import gwpop
import paths
Clight = gwcosmo.Clight
from utils import inverse_transform_sample
from mock_posteriors import gen_snr_scaled_PE

import os
import numpy as np
from GWMockCat.vt_utils import draw_thetas, interpolate_optimal_snr_grid 

# Constants
H0_FID = 67.66
DH_FID = (Clight/1.0e3) / H0_FID
OM0_FID = 0.3096
ZMAX = 10.
ZMIN = 1e-6 # can't be zero since we will eventually take the log of it

N_SOURCES = 10000
N_SAMPLES_PER_EVENT = 100
SIGMA_M = 0.5
SIGMA_DL = 1000
SNR_THRESH=8.
NUM_INJ=N_SOURCES*50

# random number generators
np_rng = np.random.default_rng(4242)

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


if  __name__ == "__main__":
    # generate data and save 
    m1s_true, zt, m1z_true, dL_true = true_vals_feature_full(rng=np_rng, num_ridges=2)
    os.mkdir(paths.data / "gw_data")
    np.save(paths.data / "gw_data/m1s_true_feature_full.npy", dL_true)
    np.save(paths.data / "gw_data/z_true_feature_full.npy", zt)
    np.save(paths.data / "gw_data/m1z_true_feature_full.npy",m1z_true)
    np.save(paths.data / "gw_data/dL_true_feature_full.npy",dL_true)
    # generate injection set
    m1zinj, dLinj, log_pinj = make_injections(np_rng,alpha=0.5,mmax_inj=350., mmin_inj=1.5)
    
    # select injections and data based off of an SNR threshold
    osnr_interp, reference_distance = interpolate_optimal_snr_grid(
        fname=paths.data / "optimal_snr_aplus_design_O5.h5") # TODO: put this on zenodo as a static dataset
    ## find injections
    thetas_inj = draw_thetas(len(dLinj),rng=np_rng)
    snr_true_inj = osnr_interp(m1zinj, m1zinj, grid=False)/dLinj * thetas_inj * 1000.
    snr_obs_inj = snr_true_inj + 1. * np_rng.normal(size=len(dLinj))
    m1zinj_det = m1zinj[snr_obs_inj>SNR_THRESH]
    dLinj_det = dLinj[snr_obs_inj>SNR_THRESH]
    log_pinj_det = log_pinj[snr_obs_inj>SNR_THRESH]
    np.save(paths.data / "gw_data/m1zinj_det.npy",m1zinj_det)
    np.save(paths.data / "gw_data/dLinj_det.npy",dLinj_det)
    np.save(paths.data / "gw_data/log_pinj_det.npy",log_pinj_det)
    ## find events and generate mock PE
    m1z_PE, dL_PE, log_PE_prior = gen_snr_scaled_PE(np_rng,m1s_true,dL_true/1000,osnr_interp,
                                                    reference_distance,N_SAMPLES_PER_EVENT,H0_FID,OM0_FID)
    dL_PE *= 1000 # unit matching
    np.save(paths.data / "gw_data/m1z_PE.npy",m1z_PE)
    np.save(paths.data / "gw_data/dL_PE.npy",dL_PE)
    np.save(paths.data / "gw_data/log_PE_prior.npy",log_PE_prior)

    