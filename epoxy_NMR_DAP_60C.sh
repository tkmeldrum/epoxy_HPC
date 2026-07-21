#!/bin/tcsh
#SBATCH --job-name=nmr_km_DAP_60C
#SBATCH --output=/sciclone/home/tkmeldrum/epoxy_kinetics/logs/%x_%j.out
#SBATCH --error=/sciclone/home/tkmeldrum/epoxy_kinetics/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tkmeldrum@wm.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00

pwd
echo "Running on host: `hostname`"
echo "Using $SLURM_CPUS_PER_TASK CPUs on node"
echo "Starting run at: `date`"

conda activate epoxy

set BASE = ~/epoxy_kinetics
mkdir -p $BASE/logs
cd $BASE

# Single-dataset MCMC fit, not the 12-array epoxy_NMR_mcmc.sh.
# r = max(alpha_data) per dataset -- do NOT use fit_kuro.py (free r).
# NMR alpha(t) for DAP/60C comes from cpmg_fit_results/DAP_60C.csv (relative
# path -- must run with CWD = $BASE), since it's not yet in epoxy_data_13Mar2026.mat.
python3 fit_kuro_fixedr.py NMR DAP 60
echo "MCMC done at: `date`"

conda deactivate
