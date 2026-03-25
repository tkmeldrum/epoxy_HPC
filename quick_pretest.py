# test_DAP2.py
import numpy as np
from scipy.io import loadmat
from BatchBayesian_kuro import process_single, nmr_index, mat

# Quick diagnostic first
print('ii =', nmr_index[('DAP2', 25)])
t = mat['NMR'][0, 0]['clean_time']
print('NMR clean_time shape:', t.shape)
print('Number of experiments:', t.shape[1])

# Monkey-patch to tiny run
import BatchBayesian_kuro as bb
bb.nsteps = 100
bb.burnin = 20
bb.nwalkers = 16

result = process_single(('NMR', 'DAP2', 25))
print("Result:", result)