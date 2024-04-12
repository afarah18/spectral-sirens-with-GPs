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

rule datanumbers:
    output:
        "src/tex/output/H0_FID.txt",
        "src/tex/output/num_found_events.txt"
    input:
        "src/data/gw_data"
    script:
        "src/scripts/data_numbers.py"

rule biasnumbers:
    output:
        "src/tex/output/nonparh0percent.txt",
        "src/tex/output/nonparh0offset.txt",
        "src/tex/output/nonparh0CI.txt",
        "src/tex/output/PLPh0percent.txt",
        "src/tex/output/PLPh0offset.txt",
        "src/tex/output/BPLh0offset.txt",
        "src/tex/output/BPL_bias_percent.txt",
        "src/tex/output/PLP_bias_percent.txt"
    input:
        "src/data/bias"
    script:
        "src/scripts/bias_numbers.py"
rule nonparpm:
    output:
        "src/tex/figures/O5_pm.pdf"
    input:
        "src/data/bias"
    script:
        "src/scripts/pm_H0_twopanel.py"
rule nonparcorner:
    output:
        "src/tex/figures/O5_GP_corner.pdf",
        "src/tex/output/mostsensitivez.txt",
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
        "src/data/bias",
        "src/data/bias.zip"
    script:
        "src/scripts/GP_example_plot.py"