rule datagen:
    output:
        directory("src/data/gw_data")
    cache:
        True
    input:
        "src/data/optimal_snr_aplus_design_O5.h5"
    script:
        "src/scripts/data_generation.py"
rule numevs:
    output:
        "src/tex/output/num_found_events.txt"
    input:
        "src/data/gw_data"
    script:
        "src/scripts/data_numbers.py"
rule h0fid:
    output:
        "src/tex/output/H0_FID.txt"
    input:
        "src/data/gw_data"
    script:
        "src/scripts/data_numbers.py"

rule parinference:
    output:
        "src/data/mcmc_parametric*.nc4"
    cache:
        True
    input:
        "src/data/gw_data"
    script:
        "src/scripts/parametric_inference.py"
rule parbias:
    output:
        "src/tex/output/PLP_bias_percent.txt"
    cache:
        True
    input:
        "src/data/gw_data"
    script:
        "src/scripts/bias_study.py"
rule nonparinference:
    output:
        "src/data/mcmc_nonparametric.nc4"
    cache:
        True
    input:
        "src/data/gw_data"
    script:
        "src/scripts/nonparametric_inference.py"
rule nonparinference_fitOm0:
    output:
        "src/data/mcmc_nonparametric_fitOm0.nc4"
    cache:
        True
    input:
        "src/data/gw_data"
    script:
        "src/scripts/nonparametric_inference_fitOm0.py"

rule nonparpm:
    output:
        "src/tex/figures/O5_GP_pm.pdf"
    input:
        "src/data/mcmc_nonparametric.nc4"
    script:
        "src/scripts/nonparametric_twopanel.py"
rule nonparcorner:
    output:
        "src/tex/figures/O5_GP_corner.pdf"
    input:
        "src/data/mcmc_nonparametric_fitOm0.nc4"
    script:
        "src/scripts/nonparametric_corner.py"

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
rule mostsensitivez:
    output:
        "src/tex/output/mostsensitivez.txt"
    input:
        "src/data/mcmc_nonparametric_fitOm0.nc4"
    script:
        "src/scripts/nonparametric_corner.py"