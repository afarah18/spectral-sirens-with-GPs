import gwcosmo
import gwpop
import paths
import os
Clight = gwcosmo.Clight
from utils import inverse_transform_sample
from mock_posteriors import gen_snr_scaled_PE

import numpy as np
from GWMockCat.vt_utils import draw_thetas, interpolate_optimal_snr_grid 

# Constants
H0_FID = 67.66
DH_FID = (Clight/1.0e3) / H0_FID
OM0_FID = 0.3096
ZMAX = 10.
ZMIN = 1e-6 # can't be zero since we will eventually take the log of it

TRUEVALS = dict(
    H0=H0_FID, OM0=OM0_FID,
    alpha=-2.7,f_peak=0.05,mmax=78.0,mmin=10.0, mu_m1=30.0,sig_m1=7.0,
    zp=2.4,alpha_z=1.,beta_z=3.4
)

N_SOURCES = int(15000*1.2)
N_SAMPLES_PER_EVENT = 100
SIGMA_M = 0.5
SIGMA_DL = 1000
SNR_THRESH=8.
NUM_INJ=N_SOURCES*50*30

perf_meas = False

# random number generators
np_rng = np.random.default_rng(516)

def true_vals_PLP(rng, n_sources=N_SOURCES):
    m1s_true = inverse_transform_sample(gwpop.powerlaw_peak,[1,400],rng,N=n_sources,
                                        alpha=TRUEVALS['alpha'],f_peak=TRUEVALS['f_peak'],
                                        mMax=TRUEVALS['mmax'],mMin=TRUEVALS['mmin'],
                                        mu_m1=TRUEVALS['mu_m1'],sig_m1=TRUEVALS['sig_m1'])
    zt = inverse_transform_sample(gwpop.shouts_murmurs, [1e-3,ZMAX], rng, 
                                N=n_sources,H0=H0_FID,Om0=OM0_FID,zp=TRUEVALS['zp'],
                                alpha=TRUEVALS['alpha_z'],beta=TRUEVALS['beta_z'])
    dL_true = gwcosmo.dL_approx(zt,H0_FID,OM0_FID)
    m1z_true = m1s_true * (1 + zt)

    q_true = rng.uniform(0.,1,size=n_sources)

    return m1s_true, zt, m1z_true, dL_true, q_true

def make_injections(rng, alpha, mmax_inj, mmin_inj, zmax_inj=ZMAX,num_inj=NUM_INJ):
    m1zinj = inverse_transform_sample(gwpop.powerlaw, [mmin_inj,mmax_inj],rng, N=num_inj, 
                                 alpha=-alpha, high=mmax_inj, low=mmin_inj)
    qinj = rng.uniform(0,1,size=num_inj)
    log_pinj = -alpha * np.log(m1zinj) + np.log(
        (-alpha + 1)/(mmax_inj**(-alpha + 1)-mmin_inj**(-alpha + 1)))
    dLinj = gwcosmo.dL_approx(inverse_transform_sample(
        gwpop.unif_comoving_rate, [ZMIN,zmax_inj], rng, N=num_inj, H0=H0_FID, Om0=OM0_FID),
          H0=H0_FID, Om0=OM0_FID)
    z_integrand = np.linspace(ZMIN,zmax_inj,num=500)
    pdl = np.abs(gwpop.unif_comoving_rate(gwcosmo.z_at_dl_approx(dLinj,H0=H0_FID, Om0=OM0_FID,zmin=ZMIN,zmax=50),H0=H0_FID,Om0=OM0_FID # p(z)
        )/np.trapz(y=gwpop.unif_comoving_rate(z_integrand,H0=H0_FID, Om0=OM0_FID),x=z_integrand # normalization
        )/gwcosmo.dDLdz_approx(gwcosmo.z_at_dl_approx(dLinj,H0_FID,OM0_FID,zmin=ZMIN,zmax=50), H0=H0_FID, Om0=OM0_FID) # convert to p(dl)
    )
    log_pinj += np.log(pdl)

    return m1zinj, dLinj, qinj, log_pinj

if  __name__ == "__main__":
    osnr_interp, reference_distance = interpolate_optimal_snr_grid(
        fname=paths.data / "optimal_snr_aplus_design_O5.h5")
    
    # generate data and save 
    m1s_true, zt, m1z_true, dL_true, q_true = true_vals_PLP(rng=np_rng)
    try:
        np.save(paths.data / "gw_data/m1s_true_PLP.npy", m1s_true)
    except FileNotFoundError:
        os.mkdir(paths.data / "gw_data")
        np.save(paths.data / "gw_data/m1s_true_PLP.npy", m1s_true)
    np.save(paths.data / "gw_data/q_true_PLP.npy",q_true)
    np.save(paths.data / "gw_data/z_true_PLP.npy", zt)
    np.save(paths.data / "gw_data/m1z_true_PLP.npy",m1z_true)
    np.save(paths.data / "gw_data/dL_true_PLP.npy",dL_true)
    # generate injection set
    m1zinj, dLinj, qinj, log_pinj = make_injections(np_rng,alpha=2.,mmax_inj=350., mmin_inj=1.5)
    
    # select injections and data based off of an SNR threshold
    ## find injections
    thetas_inj = draw_thetas(len(dLinj),rng=np_rng)
    snr_true_inj = osnr_interp(m1zinj, m1zinj*qinj, grid=False)/dLinj * thetas_inj * 1000.
    snr_obs_inj = snr_true_inj + 1. * np_rng.normal(size=len(dLinj))
    m1zinj_det = m1zinj[snr_obs_inj>SNR_THRESH]
    qinj_det = qinj[snr_obs_inj>SNR_THRESH]
    dLinj_det = dLinj[snr_obs_inj>SNR_THRESH]
    log_pinj_det = log_pinj[snr_obs_inj>SNR_THRESH]
    np.save(paths.data / "gw_data/m1zinj_det.npy",m1zinj_det)
    np.save(paths.data / "gw_data/qinj_det.npy",qinj_det)
    np.save(paths.data / "gw_data/dLinj_det.npy",dLinj_det)
    np.save(paths.data / "gw_data/log_pinj_det.npy",log_pinj_det)
    
    ## find events and generate mock PE
    m1z_PE, m2z_PE, dL_PE, log_PE_prior, det_dict = gen_snr_scaled_PE(np_rng,m1s_true,m1s_true*q_true,dL_true/1000,osnr_interp,
                                                            reference_distance,N_SAMPLES_PER_EVENT,H0_FID,OM0_FID,
                                                            # errors taken from Jose's "gwutils.py" for O5
                                                            mc_sigma=3.0e-2,eta_sigma=5.0e-3,theta_sigma=5.0e-2, snr_thresh=SNR_THRESH,
                                                            return_og=True)

    dL_PE *= 1000 # unit matching
    if perf_meas:
        # overwrite PE samples with the true values for each event, adding a dimension so that 
        # later array operations still work
        m1z_PE = np.expand_dims(det_dict['m1'] * (1 + det_dict['z']),1)
        m2z_PE = np.expand_dims(det_dict['m2'] * (1 + det_dict['z']),1)
        dL_PE = np.expand_dims(det_dict['lum_dist'], 1)
        log_PE_prior = np.zeros((len(det_dict['m1']),1))

    # cut out samples below SNR interpolation range. 
    # It is very unlikely that any samples are down there, but we do this just in case.
    # this does not change the prior shape, only the normalization
    n_samps = np.inf
    for i in range(len(m1z_PE)): # for each event
        m1z_PE[i], m2z_PE[i], dL_PE[i], log_PE_prior[i] = m1z_PE[i,m2z_PE[i]>0.05], \
            m2z_PE[i,m2z_PE[i]>0.05], dL_PE[i,m2z_PE[i]>0.05], log_PE_prior[i,m2z_PE[i]>0.05]
        if len(m1z_PE[i]) < n_samps:
            n_samps = len(m1z_PE[i])
    # make all events have the same # of samples
    if n_samps < N_SAMPLES_PER_EVENT:
        for i in range(len(m1z_PE)):
            m1z_PE[i], m2z_PE[i], dL_PE[i], log_PE_prior[i] = m1z_PE[i,:n_samps], \
                m2z_PE[i,:n_samps], dL_PE[i,:n_samps], log_PE_prior[i,:n_samps]
        N_SAMPLES_PER_EVENT = n_samps
    
    np.save(paths.data / "gw_data/m1z_PE.npy",m1z_PE)
    np.save(paths.data / "gw_data/m2z_PE.npy",m2z_PE)
    np.save(paths.data / "gw_data/dL_PE.npy",dL_PE)
    np.save(paths.data / "gw_data/log_PE_prior.npy",log_PE_prior)