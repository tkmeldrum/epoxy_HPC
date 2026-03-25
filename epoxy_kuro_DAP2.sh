#!/bin/tcsh
#SBATCH --job-name=mcmc_epoxy_DAP2
#SBATCH --output=/sciclone/home/tkmeldrum/epoxy_kinetics/logs/%x_%A_%a.out
#SBATCH --error=/sciclone/home/tkmeldrum/epoxy_kinetics/logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tkmeldrum@wm.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --array=1-3

pwd
echo "Running on host: `hostname`"
echo "Using $SLURM_CPUS_PER_TASK CPUs on node"
echo "Starting run at: `date`"

set TEMPS = (25 33 40)
set TEMP = $TEMPS[${SLURM_ARRAY_TASK_ID}]

echo "Running NMR DAP2 ${TEMP}C"

conda activate epoxy

# Free-r version
python3 ~/epoxy_kinetics/BatchBayesian_kuro.py NMR DAP2 $TEMP

# Fixed-r version
# python3 ~/epoxy_kinetics/BatchBayesian_fixedr_kuro.py NMR DAP2 $TEMP

echo "Finished task ${SLURM_ARRAY_TASK_ID} at: `date`"
conda deactivate