import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import paths
from data_generation import TRUEVALS

fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(8,4))
CIs = np.zeros(17)
for i in trange(17):
    id = az.from_netcdf(paths.data / f"bias/mcmc_parametric_PLP_{i}.nc4")
    num_less = np.sum(id.posterior['mmin'] < TRUEVALS['mmin'])
    frac_less = num_less/len(id.posterior['mmin'].T)
    if i==16:
        a=1
        l=3
        c='g'
        label="Run displayed in Figure 1"
        zorder=100
        axs[1].axvline(frac_less,color=c,alpha=a,lw=l,zorder=zorder,label=label)
    else:
        a=0.3
        l=1
        c='k'
        zorder=None
        if i==0:
            label=label="non-displayed runs"
        else:
            label=None
    CIs[i] = frac_less
    axs[0].hist(id.posterior['mmin'].T,density=True,histtype='step',color=c,alpha=a,lw=l,zorder=zorder,label=label)
axs[0].axvline(TRUEVALS['mmin'],c='k',label='Injected value')
axs[0].set_xlabel("$m_{\\min}\, [M_\\odot]$")
axs[1].hist(CIs,color='k',density=False,bins=15,histtype='step')
axs[1].set_xlabel("CI at which true $m_{\\min}$ value is recovered")
axs[0].legend()
plt.savefig(paths.figures / "mmin_hist_unifq.png")
plt.clf()

for i in trange(17):
    id = az.from_netcdf(paths.data / f"bias/mcmc_parametric_PLP_{i}.nc4")
    if i==16:
        a=1
        l=3
        c='g'
        label="Run displayed in Figure 1"
        zorder=100
        # plt.axvline(frac_less,color=c,alpha=a,lw=l,zorder=zorder,label=label)
    else:
        a=0.3
        l=1
        c='k'
        label=None
        zorder=None
    plt.hist(id.posterior['H0'].T,density=True,histtype='step',color=c,alpha=a,lw=l,zorder=zorder,label=label)
plt.axvline(TRUEVALS['H0'],c='k',label='Injected value')
plt.xlabel("$H0$")
plt.legend()
plt.savefig(paths.figures / "H0_hist_unifq.png")