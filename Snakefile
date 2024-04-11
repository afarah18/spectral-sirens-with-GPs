rule unzipdat:
    output:
        directory("src/data/gw_data")
    input:
        "src/data/gw_data.zip"
    script:
        "src/scripts/unzip_dat.sh"
rule unzipbias:
    output:
        directory("src/data/bias")
    input:
        "src/data/bias.zip"
    script:
        "src/scripts/unzip_bias.sh"

rule h0fid:
    output:
        "src/tex/output/H0_FID.txt"
    input:
        "src/data/optimal_snr_aplus_design_O5.h5"
    script:
        "src/scripts/data_numbers.py"
rule numevs:
    output:
        "src/tex/output/num_found_events.txt"
    input:
        "src/data/gw_data"
    script:
        "src/scripts/data_numbers.py"

rule nonparpm:
    output:
        "src/tex/figures/O5_pm.pdf"
    input:
        "src/tex/output/PLP_bias_percent.txt"
    script:
        "src/scripts/pm_H0_twopanel.py"
rule nonparcorner:
    output:
        "src/tex/figures/O5_GP_corner.pdf"
    input:
        "src/data/mcmc_nonparametric_fitOm0.nc4"
    script:
        "src/scripts/nonparametric_corner.py"

rule mostsensitivez:
    output:
        "src/tex/output/mostsensitivez.txt"
    input:
        "src/data/mcmc_nonparametric_fitOm0.nc4"
    script:
        "src/scripts/nonparametric_corner.py"
rule Hz_percent:
    output:
        "src/tex/output/Hz_percent.txt"
    input:
        "src/data/mcmc_nonparametric_fitOm0.nc4"
    script:
        "src/scripts/nonparametric_corner.py"

rule algo:
    output:
        "src/tex/output/priors_placeholder.txt"
    script:
        "src/scripts/priors.py"

rule GPeg:
    output:
        "src/tex/figures/GP_example.pdf"
    input:
        "src/data/bias/"
    script:
        "src/scripts/GP_example_plot.py"
