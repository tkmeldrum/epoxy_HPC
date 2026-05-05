#!/bin/tcsh
#SBATCH --job-name=nmr_km_fixedr
#SBATCH --output=/sciclone/home/tkmeldrum/epoxy_kinetics/logs/%x_%A_%a.out
#SBATCH --error=/sciclone/home/tkmeldrum/epoxy_kinetics/logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tkmeldrum@wm.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00
#SBATCH --array=1-12

pwd
echo "Running on host: `hostname`"
echo "Using $SLURM_CPUS_PER_TASK CPUs on node"
echo "Starting run at: `date`"

conda activate epoxy

set BASE = ~/epoxy_kinetics

# 12 NMR datasets: EDA/DAP/DAP2/DAB x 25/33/40 C
# Temps are integers (no C suffix) — matches fit_kuro_fixedr.py argument parser
set SAMPLES = (EDA  EDA  EDA  DAP  DAP  DAP  DAP2 DAP2 DAP2 DAB  DAB  DAB)
set TEMPS   = (25   33   40   25   33   40   25   33   40   25   33   40)

set SAMPLE = $SAMPLES[${SLURM_ARRAY_TASK_ID}]
set TEMP   = $TEMPS[${SLURM_ARRAY_TASK_ID}]
set LABEL  = NMR_${SAMPLE}_${TEMP}C

echo "Running task ${SLURM_ARRAY_TASK_ID}: $LABEL"

# ── Step 1: MCMC fit ────────────────────────────────────────────────────────
# Fixed r = max(alpha_data) per dataset — do NOT use fit_kuro.py (free r)
python3 $BASE/fit_kuro_fixedr.py NMR $SAMPLE $TEMP
echo "MCMC done: $LABEL at `date`"

# ── Step 2: Convergence diagnostics ────────────────────────────────────────
set NPZ = $BASE/mcmc_samples/${LABEL}_fitdata.npz
if ( -f $NPZ ) then
    python3 $BASE/MCMC_diagnostics.py $NPZ
    echo "Diagnostics done: $LABEL at `date`"
else
    echo "WARNING: $NPZ not found — MCMC may have failed for $LABEL"
endif

# ── Step 3: Posterior plots and per-task summary row ───────────────────────
# Each task writes its own CSV to avoid concurrent-append races.
# These are merged into posterior_summary_NMR_fixedr.csv by the post-process
# step below (submit separately with --dependency=afterok).
if ( -f $NPZ ) then
    set TASK_CSV = $BASE/posterior_summary_parts/posterior_summary_${LABEL}.csv
    mkdir -p $BASE/posterior_summary_parts
    python3 $BASE/plot_mcmc.py $NPZ \
        --outdir $BASE/fit_plots_nmr \
        --summary $TASK_CSV
    echo "Plots done: $LABEL at `date`"
endif

echo "Finished task ${SLURM_ARRAY_TASK_ID}: $LABEL"
echo "Finished at: `date`"

conda deactivate

# ── Post-processing (submit separately after all array tasks complete) ──────
#
# After all 12 tasks finish, merge per-task CSVs and write final provenance:
#
#   sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} epoxy_NMR_postprocess.sh
#
# Or run interactively on HPC:
#
#   python3 ~/epoxy_kinetics/plot_mcmc.py \
#       --input-dir ~/epoxy_kinetics/mcmc_samples \
#       --outdir    ~/epoxy_kinetics/fit_plots_nmr \
#       --summary   ~/epoxy_kinetics/posterior_summary_NMR_fixedr.csv
#
# This re-processes all 12 .npz files together and writes the provenance file.
