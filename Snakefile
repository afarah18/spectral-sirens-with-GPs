# rule datagen:
#     output:
#         directory("src/data/gw_data")
#     cache:
#         True
#     script:
#         "src/sripts/data_generation.py"
# rule nonparinference:
#     output:
#         "src/data/mcmc_nonparametric.nc4"
#     cache:
#         True
#     input:
#         "src/data/gw_data"
#     script:
#         "src/scripts/nonparametric_inference.py"
# rule nonparplots:
#     output:
#         "O5_GP.pdf"
#     input:
#         "src/data/mcmc_nonparametric.nc4" # should I include the gw_data directory too?
#     script:
#         "src/scripts/nonparametric_plots.py"