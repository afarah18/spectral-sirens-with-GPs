from data_generation import H0_FID
import paths
import numpy as np

dL_PE = np.load(paths.data / "gw_data/dL_PE.npy")

with open(paths.output / "num_found_events.txt", "w") as f:
    print(len(dL_PE), file=f)
with open(paths.output / "H0_FID.txt", "w") as f:
    print(H0_FID, file=f)