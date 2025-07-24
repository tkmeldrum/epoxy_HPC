#!/bin/tcsh
#SBATCH --job-name=mcmc_diag
#SBATCH --output=/sciclone/home/tkmeldrum/epoxy_kinetics/logs/%x_%A_%a.out
#SBATCH --error=/sciclone/home/tkmeldrum/epoxy_kinetics/logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tkmeldrum@wm.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=1-27  # Update this once you count the lines in your list

echo "Running on host: `hostname`"
echo "Starting task at: `date`"
echo "SLURM task ID: $SLURM_ARRAY_TASK_ID"

# Activate your environment
conda activate epoxy

# Read the .npz file path from the file list
set FILE_LIST = ~/epoxy_kinetics/npz_file_list.txt
set LINE_NUM = $SLURM_ARRAY_TASK_ID
set NPZ_FILE = `sed -n "${LINE_NUM}p" $FILE_LIST`

echo "Processing file: $NPZ_FILE"
python3 ~/epoxy_kinetics/MCMC_diagnostics.py "$NPZ_FILE"
python3 ~/epoxy_kinetics/BatchBayesian_plots.py "$NPZ_FILE"

echo "Finished task at: `date`"
conda deactivate
