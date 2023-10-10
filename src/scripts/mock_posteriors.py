
"""Module to make mock posteriors"""
from scipy import stats

def draw_param_samps(rng, mean,sigma,bounds,size):
    mock_samples = stats.truncnorm(
    (bounds[0] - mean) / sigma, (bounds[1] - mean) / sigma, loc=mean, scale=sigma
).rvs(size=size,random_state=rng).T
    return mock_samples

def gen_jittered_PE(rng, true_vals,sigma,bounds,n_samples,n_detections=None,return_maxL=False):
    if n_detections is None:
        n_detections = len(true_vals)
    maxL = draw_param_samps(rng,mean=true_vals,sigma=sigma,
                     bounds=bounds,size=(1,n_detections))
    mock_samples = draw_param_samps(rng,mean=maxL.T,sigma=sigma,bounds=bounds,
                                    size=(n_samples,n_detections))
    if return_maxL:
        return mock_samples, maxL
    else:
        return mock_samples
    
def gen_snr_scaled_PE(rng, true_m1s, true_dl, osnr_interp, reference_distance, n_samples,
                      H0, Om0, mc_sigma=0.04,eta_sigma=0.01,snr_thresh=8):
    """ Wrapper function for GWMockCat functions"""
    from GWMockCat import posterior_utils, cosmo_utils
    import astropy.cosmology as ap_cosmology 
    from astropy import units as u
    import numpy as np

    uncert_gwmc = {'threshold_snr': snr_thresh, 'snr': 1.0, 'mc': mc_sigma, 'Theta': 0.15, 'eta': eta_sigma}
    cosmo = ap_cosmology.FlatLambdaCDM(H0=H0, Om0=Om0)
    cosmo_gwmc = cosmo_utils.interp_cosmology(zmin=0.,zmax=20.,cosmology=cosmo,dist_unit=u.Unit(reference_distance[-3:]))
    true_z = cosmo_gwmc['z_at_dL'](true_dl)
    print(true_z.max())
    observed, detected_dict = posterior_utils.generate_obs_from_true_list(m1t=true_m1s,m2t=true_m1s,
                                                                                    zt=true_z,osnr_interp=osnr_interp,
                                                                                    cosmo_dict=cosmo_gwmc,rng=rng,
                                                                                    PEsamps=n_samples,uncert=uncert_gwmc,
                                                                                    osnr_interp_dist=reference_distance
    )
    mc_det = observed.sel(parameter='Mc_det_samps').to_array().to_numpy()
    eta = observed.sel(parameter='eta_samps').to_array().to_numpy()
    rho = observed.sel(parameter='rho_samps').to_array().to_numpy()
    theta = observed.sel(parameter='theta_samps').to_array().to_numpy()

    # transform to useful parameters
    m1det_PE, m2det_PE = posterior_utils.m1m2_from_mceta(mc_det,eta)
    dl_PE = theta * osnr_interp(m1det_PE,m2det_PE,grid=False)/rho
    log_jacobian = - np.log(posterior_utils.dm1m2_dMceta(m1det_PE,m2det_PE)) -np.log(posterior_utils.ddL_drho(osnr_interp,m1det_PE,m2det_PE,rho,theta))
    log_prior_PE = np.log(0.25*mc_det.max()*rho.max()) + log_jacobian 
    return m1det_PE, dl_PE, log_prior_PE

    # can try: resample to a flat-in-detector frame mass and flat-in-dL prior
    
if  __name__ == "__main__":
    from GWMockCat.vt_utils import interpolate_optimal_snr_grid
    import numpy as np
    import matplotlib.pyplot as plt
    osnr, ref_dist = interpolate_optimal_snr_grid(fname="/Users/amandafarah/Library/Mobile Documents/com~apple~CloudDocs/projects/mock-PE/sensitivity/optimal_snr_aplus_design_O5.h5")
    m, d, lp = gen_snr_scaled_PE(np.random.default_rng(), np.array([5.2,7,30,50]), np.array([0.1,2,0.5,1.]), osnr, ref_dist,100,70,0.3)
    print(len(m))
    for i in range(len(m)):
        plt.hist(m[i])
    plt.show()
    for i in range(len(m)):
        plt.hist(d[i])
    plt.show()
    for i in range(len(m)):
        plt.hist(lp[i])
    plt.show()

# def observed_posteriors(m1z,dL,snr,n_samples,snr_opt,snr_th,fmin=10,Tobs,detector,*args):
#     #clevel = 1.65 #90% CL
#     sigma_Mc,sigma_eta,sigma_w = error_pe_detector(detector)
    
#     lower_snr, upper_snr = 0.0, snr_opt*10.
#     mu_snr, sigma_snr = snr, 1.
#     snr_obs = stats.truncnorm.rvs((lower_snr - mu_snr) / sigma_snr, (upper_snr - mu_snr) / sigma_snr, loc=mu_snr, scale=sigma_snr,size=n_samples)
    
#     Mz = gw.mchirp(m1z,m2z) #source frame masses
#     etas = gw.eta(m1z,m2z)
#     #Mz
#     logMz_obs = np.random.normal(loc=np.log(Mz),scale=sigma_Mc* snr_th/snr_obs,size=n_samples)
#     Mz_obs = np.exp(logMz_obs)
#     #eta
#     lower_etas, upper_etas = 0.0, 0.25
#     mu_etas, sigma_etas = etas, sigma_eta* snr_th/snr_obs
#     eta_obs = stats.truncnorm.rvs((lower_etas - mu_etas) / sigma_etas, (upper_etas - mu_etas) / sigma_etas, loc=mu_etas, scale=sigma_etas,size=n_samples)
#     #w
#     w = snr/snr_opt
#     lower_ws, upper_ws = 0.0, 1.0
#     mu_ws, sigma_ws = w, sigma_w* snr_th/snr_obs
#     w_obs = stats.truncnorm.rvs((lower_ws - mu_ws) / sigma_ws, (upper_ws - mu_ws) / sigma_ws, loc=mu_ws, scale=sigma_ws, size=n_samples)
    
#     M_obs = Mz_obs / eta_obs**(3./5.)
#     m1z_obs = (M_obs + np.sqrt(M_obs**2 - 4.* eta_obs * M_obs**2))/2.
#     m2z_obs = (M_obs - np.sqrt(M_obs**2 - 4.* eta_obs * M_obs**2))/2.
    
#     snr_opt_1Mpc = gw.vsnr(m1z_obs,m2z_obs,dL,fmin,Tobs,detector,*args)
#     dL_obs = dL*snr_opt_1Mpc * w_obs / snr_obs #Mpc
    
#     jacobian = np.abs(w_obs * snr_opt_1Mpc * (m1z_obs - m2z_obs) * np.power(gw.eta(m1z_obs,m2z_obs),3./5)/np.power(m1z_obs + m2z_obs,2) / np.power(dL_obs,2))

#     return w_obs, m1z_obs, m2z_obs, dL_obs, snr_obs, jacobian