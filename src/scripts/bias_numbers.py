import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from pm_H0_twopanel import color_PLP, color_BPL, color_GP
from data_generation import H0_FID
import paths

# load data
id_PLP = az.InferenceData.from_netcdf(paths.data / "bias/mcmc_parametric_PLP_16.nc4")
id_BPL = az.InferenceData.from_netcdf(paths.data / "bias/mcmc_parametric_BPL_16.nc4")
bias_PLP = np.loadtxt(paths.data / "bias/bias_PLP.txt")
bias_BPL = np.loadtxt(paths.data / "bias/bias_BPL.txt")

# calculate summary statistics and save
N_CATALOGS = len(bias_BPL)
percent_bias_PLP = np.sum(bias_PLP>1)/N_CATALOGS * 100
percent_bias_BPL = np.sum(bias_BPL>1)/N_CATALOGS * 100
with open(paths.output / "PLP_bias_percent.txt", "w") as f:
    print(f"{percent_bias_PLP:.0f}", file=f)
with open(paths.output / "BPL_bias_percent.txt", "w") as f:
    print(f"{percent_bias_BPL:.0f}", file=f)

# parametric summary stats that we only need for one catalog
with open(paths.output / "PLPh0offset.txt","w") as f:
    print(f"{bias_PLP[16]:.1f}",file=f)
with open(paths.output / "PLPh0percent.txt","w") as f:
    print(f"{np.std(id_PLP.posterior['H0'][0]).values/np.mean(id_PLP.posterior['H0'][0]).values*100:.0f}",file=f)
with open(paths.output / "BPLh0offset.txt","w") as f:
    print(f"{bias_BPL[16]:.1f}",file=f)

# same thing for nonparametric case
id = az.InferenceData.from_netcdf(paths.data / "bias/mcmc_nonparametric_16.nc4")
h0samps = id.posterior['H0'][0].values

with open(paths.output / "nonparh0percent.txt", "w") as f:
    print(f"{np.std(h0samps)/np.mean(h0samps)*100:.1f}", file=f)
lower = np.mean(h0samps)-np.percentile(h0samps,5)
upper = np.percentile(h0samps,95)-np.mean(h0samps)
with open(paths.output / "nonparh0CI.txt", "w") as f:
    print(f"${np.mean(h0samps):.1f}"+"^{+"+f"{lower:.1f}"+"}"+"_{-"+f"{upper:.1f}"+"}$", file=f)

nonpar_offset = np.abs(h0samps.mean()-H0_FID)/h0samps.std()
with open(paths.output / "nonparh0offset.txt","w") as f:
    print(f"{nonpar_offset:.1f}",file=f)

# make a plot of recovered H0 posteriors
fig = plt.figure(figsize=(7.5/2,4*.75),facecolor='none')
prior=np.linspace(30,110,num=100)
a=0.3
l=1
for i in range(N_CATALOGS):
    id_PLP = az.from_netcdf(paths.data / f"bias/mcmc_parametric_PLP_{i}.nc4")
    id_BPL = az.from_netcdf(paths.data / f"bias/mcmc_parametric_BPL_{i}.nc4")
    kde_PLP = gaussian_kde(id_PLP.posterior['H0'][0].values)
    kde_BPL = gaussian_kde(id_BPL.posterior['H0'][0].values)
    plt.plot(prior,kde_PLP(prior),color=color_PLP,alpha=a,lw=l)
    plt.plot(prior,kde_BPL(prior),color=color_BPL,alpha=a,lw=l)
plt.plot([],c=color_PLP,label=r'\textsc{Power Law + Peak}')
plt.plot([],c=color_BPL,label=r'\textsc{Broken Power Law}')
# plt.plot([],c='k',lw=l,alpha=a,label="all other runs")
plt.ylabel('posterior density')
leg=plt.legend(framealpha=1)
leg.get_frame().set_linewidth(0.0)
plt.xlabel('$H_0$ [km/s/Mpc]')
plt.axvline(H0_FID, c='k')
plt.ylim(0,0.12)
plt.xlim(30,110)
plt.savefig(paths.figures / "bias.pdf")
plt.clf()