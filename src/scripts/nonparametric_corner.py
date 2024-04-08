import matplotlib.pyplot as plt
import numpy as np
import arviz as az
from scipy.stats import gaussian_kde

from data_generation import H0_FID, OM0_FID, ZMIN, ZMAX
from gwcosmo import E_of_z

import paths
plt.style.use(paths.scripts / "matplotlibrc")
TEST_Z = np.linspace(ZMIN,ZMAX,num=100)

def calc_Hz(z,H0,Om0):
    E = E_of_z(z=z,Om0=Om0)
    return H0 * E

def Hz(id, ax=None,save=False,inset=True):
    samples = id.posterior
    try:
        Om0=samples['Om0'][0].values
        H0=samples['H0'][0].values
        nsamps = len(H0)
    except KeyError:
        pass
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
    H_z = np.zeros((nsamps,len(TEST_Z)))
    for i in range(nsamps):
        H_z[i] = calc_Hz(TEST_Z,H0[i],Om0[i])
        ax.plot(TEST_Z,H_z[i],lw=0.2, c="blue",alpha=0.1)
    ax.plot(TEST_Z, calc_Hz(TEST_Z,H0_FID,OM0_FID),c='k', lw=2,label="True value")
    ax.set_xlabel("$z$")
    ax.set_ylabel("$H(z)$ [km/s/Mpc]")
    ax.set_xlim(0,3)
    ax.set_ylim(0,500)
    ax.legend(framealpha=0,loc='lower right')
    
    if inset:
        # find the best-measured redshift
        a = np.argmin(H_z.std(axis=0))
        zbest=TEST_Z[a]
        H_of_zbest = H_z[:,a]
        with open(paths.output / "mostsensitivez.txt", "w") as f:
            print(f"{zbest:.1f}", file=f)
        with open(paths.output / "Hz_percent.txt", "w") as f:
            print(f"{H_z.std(axis=0)[a]/H_z.mean(axis=0)[a]*100:.1f}", file=f)

        # calculate the truth there
        H_of_zbest_true = calc_Hz(zbest,H0_FID,OM0_FID)            

        # make the inset
        iax = ax.inset_axes(bounds=[0.2,270,1.5,215],transform=ax.transData)
        iax.set_xlabel("$H(z=%1.1f)$ [km/s/Mpc]"%zbest,fontsize=8)
        iax.set_xticks([100,125,150])
        iax.set_xticklabels([100,125,150],fontsize=8)
        # iax.set_ylabel('posterior density',fontsize=8)

        # arrow from the inset to zbest
        # midpoint_x = 1.75/2 + 0.5
        # offset_y= 50
        # lowpoint_y = 450-offset_y
        # ax.arrow(x=zbest,y=H_of_zbest+offset_y,dx=midpoint_x-zbest,dy = lowpoint_y-(H_of_zbest+offset_y+10),
        #          head_width=0.1,overhang=0.2,length_includes_head=True, head_starts_at_zero=True,color='grey')
    
        # plot
        prior=np.linspace(H_of_zbest.min(),H_of_zbest.max(),num=200)
        kde=gaussian_kde(H_of_zbest)
        iax.plot(prior,kde(prior))
        iax.axvline(H_of_zbest_true,color='k',label="True value")
        iax.tick_params(left=False,labelleft=False)

    if save:
        fig.savefig(paths.figures / "O5_GP_Hz.pdf")
    else:
        return ax

def H0_Om_corner(id): 
    samples = id.posterior
    try:
        samples['Om0']
    except KeyError:
        pass
    fig = plt.figure(figsize=(7.5,4*.75),facecolor='none')
    subfigs = fig.subfigures(1,2,wspace=0.07,width_ratios=[1.2, 1])
    for sf in subfigs:
        sf.set_facecolor('none')

    axsLeft=subfigs[0].subplots(2,2)
    az.plot_pair(id,var_names=['H0','Om0'],marginals=True,kind='kde', ax=axsLeft,
                 kde_kwargs={'plot_kwargs':{'color':'b'},'contourf_kwargs':{'cmap':'Blues'}},
                 reference_values={'Om0':OM0_FID,'H0':H0_FID},
                 reference_values_kwargs={'marker':'+','color':'k','ms':10,'mew':2})
    # un-rotate Om0 plot
    axsLeft[1,1].cla()
    axsLeft[0,0].tick_params(bottom=False,labelbottom=False)
    az.plot_dist(samples['Om0'][0],ax=axsLeft[1,1])
    axsLeft[1,1].set_xlabel("$\Omega_m$")
    axsLeft[1,1].set_xticks([0,0.5,1])
    axsLeft[1,0].set_xticks([50,75,100])
    # label axes
    axsLeft[1,0].set_xlabel("$H_0$ [km/s/Mpc]")
    axsLeft[1,0].set_ylabel("$\Omega_m$")
    # axsLeft[1,0].plot([],[],**{'marker':'+','color':'k','ms':10,'mew':2,'lw':0},label='injected value')
    # axsLeft[1,0].legend(framealpha=0,handletextpad=0.1)
    
    axsRight=subfigs[1].subplots(1,1)
    Hz(id,ax=axsRight,save=False)
    fig.savefig(paths.figures / "O5_GP_corner.pdf")

id = az.InferenceData.from_netcdf(paths.data / "mcmc_nonparametric_fitOm0_old.nc4")
H0_Om_corner(id)