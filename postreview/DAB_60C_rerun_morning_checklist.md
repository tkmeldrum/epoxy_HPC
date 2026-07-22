# DAB/60C MCMC rerun — morning checklist

Context: DAB/60C's first MCMC run had one stuck walker (R̂ up to 1.08, `log_k1` CI −9.9 to −3.9).
Resubmitted the identical job unchanged (no seed is set, so walker init positions differ run
to run) to see if it resolves on its own.

## 1. Check the job finished
```bash
squeue -u tkmeldrum
cat logs/nmr_km_DAB_60C_<jobid>.out   # look for "MCMC done at: ..."
```

## 2. Pull the new chain back — use a DIFFERENT local filename so it doesn't
   clobber the original (already reviewed) one in mcmc_samples_nmr/
```bash
rsync -av tkmeldrum@kuro.sciclone.wm.edu:~/epoxy_kinetics/mcmc_samples_nmr/NMR_DAB_60C_fitdata.npz \
    /Users/tyler/Documents/GitHub/epoxy_HPC/mcmc_samples_nmr/NMR_DAB_60C_rerun_fitdata.npz
```

## 3. Diagnostics on the rerun
```bash
cd /Users/tyler/Documents/GitHub/epoxy_HPC
python MCMC_diagnostics.py mcmc_samples_nmr/NMR_DAB_60C_rerun_fitdata.npz
```
Check R̂ for all 5 params — compare against the original run's 1.04–1.08. Look at
`diagnostic_plots/NMR_DAB_60C_rerun_trace.png` for whether every walker now tracks the main pack
(no permanently-offset gray trace like before).

## 4. Post-process into a NEW review folder (keep the original DAB_60C postreview intact for comparison)
```bash
mkdir -p postreview/DAB_60C_rerun
python plot_mcmc.py mcmc_samples_nmr/NMR_DAB_60C_rerun_fitdata.npz \
    --outdir  postreview/DAB_60C_rerun/fit_plots \
    --summary postreview/DAB_60C_rerun/posterior_summary_DAB_60C.csv
```

## 5. Compare against the original
- `postreview/DAB_60C/posterior_summary_DAB_60C.csv` (original) vs
  `postreview/DAB_60C_rerun/posterior_summary_DAB_60C.csv` (rerun) — especially `log_k1_median`/CI width.
- `postreview/DAB_60C_rerun/fit_plots/NMR_DAB_60C_alpha_ci.png` — still a clean sigmoidal fit?

## 6. If the rerun is better (tight R̂, no stuck walker, narrower log_k1 CI):
Replace the DAB inputs used earlier and redo the fold-in:
```bash
cp postreview/DAB_60C_rerun/posterior_summary_DAB_60C.csv \
   posterior_summary_parts/posterior_summary_NMR_DAB_60C.csv
python merge_nmr_parts.py                    # verify via git diff: only DAB_60C row should change
python build_combined_arrhenius.py           # new dated combined_arrhenius_*.csv
python final_results_to_plots.py combined_arrhenius_<newdate>.csv
# then re-run the pub_figs/*.py --outdir postreview/pub_figs_60C step from before
```

## 7. If the rerun is NOT better (same or different stuck walker):
Keep the original DAB/60C result as the accepted-with-caveat version (per the earlier decision) —
no action needed, this was just worth trying since it was cheap.
