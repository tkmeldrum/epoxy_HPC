#!/bin/tcsh
#SBATCH --job-name=test_log
#SBATCH --output=~/epoxy_kinetics/logs/%x_%j.out
#SBATCH --error=~/epoxy_kinetics/logs/%x_%j.err
#SBATCH --array=1-1

echo "Hello from task $SLURM_ARRAY_TASK_ID"