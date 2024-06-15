import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import paths
from data_generation import TRUEVALS

CIs = np.zeros(50)
for i in trange(2):
    id = az.from_netcdf(paths.data / f"bias/mcmc_parametric_PLP_unifq_actually_{i}.nc4")
    num_less = np.sum(id.posterior['mmin'] < TRUEVALS['mmin'])
    frac_less = num_less/len(id.posterior['mmin'].T)
    # id_BPL = az.from_netcdf(paths.data / f"bias/mcmc_parametric_BPL_{i}.nc4")
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
    CIs[i] = frac_less
    # plt.hist(id_BPL.posterior['mmin'].T,density=True,histtype='step',color='orange',alpha=a,lw=l)
    plt.hist(id.posterior['mmin'].T,density=True,histtype='step',color=c,alpha=a,lw=l,zorder=zorder,label=label)
plt.axvline(TRUEVALS['mmin'],c='k',label='Injected value')
# plt.hist(CIs,color=c,density=False,bins=15,histtype='step',alpha=a,lw=l,label="non-displayed runs")
plt.xlabel("$m_{\\min}\, [M_\\odot]$")
plt.legend()
plt.savefig(paths.figures / "mmin_hist_unifq.png")
plt.clf()

for i in trange(2):
    id = az.from_netcdf(paths.data / f"bias/mcmc_parametric_PLP_unifq_actually_{i}.nc4")
    # id_BPL = az.from_netcdf(paths.data / f"bias/mcmc_parametric_BPL_{i}.nc4")
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
    # plt.hist(id_BPL.posterior['mmin'].T,density=True,histtype='step',color='orange',alpha=a,lw=l)
    plt.hist(id.posterior['H0'].T,density=True,histtype='step',color=c,alpha=a,lw=l,zorder=zorder,label=label)
plt.axvline(TRUEVALS['H0'],c='k',label='Injected value')
# plt.hist(CIs,color=c,density=False,bins=15,histtype='step',alpha=a,lw=l,label="non-displayed runs")
plt.xlabel("$H0$")
plt.legend()
plt.savefig(paths.figures / "H0_hist_unifq.png")