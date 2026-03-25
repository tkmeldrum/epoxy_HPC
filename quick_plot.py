import numpy as np
from BatchBayesian_simple import plot_results

data = np.load("mcmc_samples/NMR_DAP2_40C_simple_fitdata.npz")
plot_results(data['samples'], data['t_data'], data['a_data'], 'NMR', 'DAP2', 40)
