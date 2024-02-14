rule datagen:
    output:
        directory("src/data/gw_data")
        "src/tex/output/H0_FID.txt"
        "src/tex/output/num_found_events.txt"
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
rule nonparh0CI:
    output:
        "src/tex/output/nonparh0CI.txt"
    input:
        "src/data/mcmc_nonparametric.nc4" 
    script:
        "src/scripts/nonparametric_numbers.py"
rule nonparh0percent:
    output:
        "src/tex/output/nonparh0percent.txt"
    input:
        "src/data/mcmc_nonparametric.nc4" 
    script:
        "src/scripts/nonparametric_numbers.py"