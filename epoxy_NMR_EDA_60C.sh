#!/bin/tcsh
#SBATCH --job-name=nmr_km_EDA_60C
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

# Single-dataset MCMC fit, not the 12-array epoxy_NMR_mcmc.sh.
# r = max(alpha_data) per dataset -- do NOT use fit_kuro.py (free r).
python3 $BASE/fit_kuro_fixedr.py NMR EDA 60
echo "MCMC done at: `date`"

conda deactivate
