import numpy as np
import arviz as az

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