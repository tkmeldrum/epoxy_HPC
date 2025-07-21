#!/bin/tcsh
#SBATCH --job-name=mcmc_epoxy
#SBATCH --output=/sciclone/home/tkmeldrum/epoxy_kinetics/logs/%x_%A_%a.out
#SBATCH --error=/sciclone/home/tkmeldrum/epoxy_kinetics/logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tkmeldrum@wm.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=36:00:00
#SBATCH --array=1-27

pwd
echo "Running on host: `hostname`"
echo "Using $SLURM_CPUS_PER_TASK CPUs on node"
echo "Starting run at: `date`"

# Activate conda
conda activate epoxy

# Define arrays (tcsh syntax)
set METHODS = (NMR NMR NMR NMR NMR NMR NMR NMR NMR \
               DSC DSC DSC DSC DSC DSC \
               DSC DSC DSC DSC DSC DSC \
               DSC DSC DSC DSC DSC DSC)

set SAMPLES = (EDA EDA EDA DAP DAP DAP DAB DAB DAB \
               EDA EDA EDA EDA EDA EDA \
               DAP DAP DAP DAP DAP DAP \
               DAB DAB DAB DAB DAB DAB)

set TEMPS = (25 33 40 25 33 40 25 33 40 \
             25 33 50 60 80 100 \
             25 33 50 60 80 100 \
             25 33 50 60 80 100)

# Pull parameters for this task ID
set METHOD = $METHODS[${SLURM_ARRAY_TASK_ID}]
set SAMPLE = $SAMPLES[${SLURM_ARRAY_TASK_ID}]
set TEMP   = $TEMPS[${SLURM_ARRAY_TASK_ID}]

echo "Running task ${SLURM_ARRAY_TASK_ID}: $METHOD $SAMPLE $TEMP"

# Run the Python script
python3 ~/epoxy_kinetics/BatchBayesian_kuro.py $METHOD $SAMPLE $TEMP

echo "Finished task ${SLURM_ARRAY_TASK_ID}: $METHOD $SAMPLE $TEMP"
echo "Finished at: `date`"

# Deactivate conda
conda deactivate
