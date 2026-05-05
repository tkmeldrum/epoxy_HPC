#!/bin/tcsh
#SBATCH --job-name=nmr_km_postprocess
#SBATCH --output=/sciclone/home/tkmeldrum/epoxy_kinetics/logs/%x_%j.out
#SBATCH --error=/sciclone/home/tkmeldrum/epoxy_kinetics/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tkmeldrum@wm.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00

# Post-processing for epoxy_NMR_mcmc.sh array job.
# Submit with dependency so this runs only after all 12 MCMC tasks complete:
#
#   FITJOB=$(sbatch --parsable epoxy_NMR_mcmc.sh)
#   sbatch --dependency=afterok:$FITJOB epoxy_NMR_postprocess.sh

pwd
echo "Running on host: `hostname`"
echo "Starting post-processing at: `date`"

conda activate epoxy

set BASE = ~/epoxy_kinetics

# Re-process all 12 .npz files together — writes posterior_summary_NMR_fixedr.csv
# and posterior_summary_NMR_fixedr.provenance.txt recording MD5/mtime of all chains.
python3 $BASE/plot_mcmc.py \
    --input-dir $BASE/mcmc_samples \
    --outdir    $BASE/fit_plots_nmr \
    --summary   $BASE/posterior_summary_NMR_fixedr.csv

echo "Posterior summary written at: `date`"
echo "Finished at: `date`"

conda deactivate
