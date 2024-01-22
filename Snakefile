rule datagen:
    output:
        directory("src/data/gw_data")
    cache:
        True
    input:
        "src/data/optimal_snr_aplus_design_O5.h5"
    script:
        "src/scripts/data_generation.py"
rule nonparinference:
    output:
        "src/data/mcmc_nonparametric.nc4"
    cache:
        True
    input:
        "src/data/gw_data"
    script:
        "src/scripts/nonparametric_inference.py"
rule nonparplots:
    output:
        "src/tex/figures/O5_GP.pdf"
    input:
        "src/data/mcmc_nonparametric.nc4"
    script:
        "src/scripts/nonparametric_twopanel.py"