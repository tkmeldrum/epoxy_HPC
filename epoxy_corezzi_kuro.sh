#!/bin/tcsh
#SBATCH --job-name=corezzi_nmr
#SBATCH --output=/sciclone/home/tkmeldrum/epoxy_kinetics/logs/%x_%A_%a.out
#SBATCH --error=/sciclone/home/tkmeldrum/epoxy_kinetics/logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tkmeldrum@wm.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --array=1-12

pwd
echo "Running on host: `hostname`"
echo "Using $SLURM_CPUS_PER_TASK CPUs on node"
echo "Starting run at: `date`"

# Activate conda
source /sciclone/apps/miniforge3-24.9.2-0/etc/profile.d/conda.csh
conda activate epoxy

# Define arrays (tcsh syntax — 1-indexed)
set SAMPLES = (EDA  EDA  EDA  DAP  DAP  DAP  DAP2 DAP2 DAP2 DAB  DAB  DAB)
set TEMPS   = (25C  33C  40C  25C  33C  40C  25C  33C  40C  25C  33C  40C)

# Pull parameters for this task ID
set SAMPLE = $SAMPLES[${SLURM_ARRAY_TASK_ID}]
set TEMP   = $TEMPS[${SLURM_ARRAY_TASK_ID}]

echo "Running task ${SLURM_ARRAY_TASK_ID}: $SAMPLE $TEMP"

# Run the Corezzi MCMC fit
python3 ~/epoxy_kinetics/BatchBayesian_nmr_corezzi.py --mcmc $SAMPLE $TEMP

echo "Finished task ${SLURM_ARRAY_TASK_ID}: $SAMPLE $TEMP"
echo "Finished at: `date`"

# Deactivate conda
conda deactivate
