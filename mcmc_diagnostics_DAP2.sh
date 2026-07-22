#!/bin/tcsh
#SBATCH --job-name=mcmc_diag_DAP2
#SBATCH --output=/sciclone/home/tkmeldrum/epoxy_kinetics/logs/%A/%x_%a.out
#SBATCH --error=/sciclone/home/tkmeldrum/epoxy_kinetics/logs/%A/%x_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tkmeldrum@wm.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=28G
#SBATCH --time=00:15:00
#SBATCH --array=1-3

echo "Running on host: `hostname`"
echo "Starting task at: `date`"
echo "SLURM task ID: $SLURM_ARRAY_TASK_ID"

source /sciclone/apps/miniforge3-24.9.2-0/etc/profile.d/conda.csh
conda activate epoxy

set TEMPS = (25 33 40)
set TEMP = $TEMPS[${SLURM_ARRAY_TASK_ID}]

set NPZ_FILE = ~/epoxy_kinetics/mcmc_samples/NMR_DAP2_${TEMP}C_fitdata.npz

echo "Processing file: $NPZ_FILE"
python3 ~/epoxy_kinetics/BatchBayesian_plots.py "$NPZ_FILE"

echo "Finished task at: `date`"
conda deactivate