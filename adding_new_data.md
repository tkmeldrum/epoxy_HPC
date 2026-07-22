# Adding a new NMR dataset (new sample or new temperature)

Quick-reference pipeline, based on how EDA/DAP/DAB at 60°C were added. See
`postreview/plan_60C_km_arrhenius_eyring.md` and `postreview/datachanges.md`
for the full worked example and lessons learned.

## 1. Get the raw data onto this Mac, with real timestamps intact

The whole α(t) analysis depends on genuine per-scan acquisition timestamps.
**Do not route raw data through OneDrive** — it's been observed to silently
overwrite file mtimes with upload time, collapsing an hours-long cure
experiment into a few-second window. Copy from the instrument's own file
share instead (Z: drive / equivalent), preserving timestamps at every hop
(`robocopy /COPY:DAT`, `scp -p`, `rsync -a`).

Before zipping, verify the timestamps actually look right:
```bash
find <raw_folder> -name "data.2d" -exec stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" {} \; | sort | head -5
# ...| sort | tail -5
```
Spread should match real scan cadence (minutes apart, hours total) — not
everything clustered in a few seconds.

## 2. Zip it, verify the zip preserved the timestamps

```bash
cd <raw_folder_parent>
zip -r <ZIP_ROOT>/<name>.zip <folder>
```
Then re-check with `unzip -l` / a quick `zipfile.ZipFile(...).infolist()`
timestamp check — `zip -r` **appends** to an existing archive rather than
overwriting, so delete any stale zip of the same name first if re-zipping.

## 3. Register the dataset in `cpmg_batch_fit.py` and `pub_figs/cpmg_fit.py`

Both files have an `EXPERIMENTS_60C`-style dict (name it for whatever new
temperature/condition you're adding) keyed by sample, e.g.:
```python
EXPERIMENTS_60C = {
    "EDA": {"zip": ZIP_ROOT / "DGEBA_EDA_60.zip", "prefix": "debugger4"},
    "DAP": {"zip": ZIP_ROOT / "DGEBA_DAP_60.zip", "prefix": "debugger"},
    "DAB": {"zip": ZIP_ROOT / "DGEBA_DAB_60.zip", "prefix": "debugger"},
}
```
`prefix` is the path inside the zip to the numbered scan folders. Add a
`run_all()` block and a `--diagnose` branch for it (mirror the existing
`DAP2`-style blocks). Keep both copies of this dict in sync.

## 4. Run and verify `cpmg_batch_fit.py`

There's no way to process just the new dataset — a full run reprocesses
everything. Since existing raw data doesn't change, this is deterministic:
```bash
python cpmg_batch_fit.py --diagnose <SAMPLE> <TEMP>C   # sanity check first
python cpmg_batch_fit.py                                # full run
git diff --stat cpmg_fit_results/                       # confirm 0 changed rows, only additions
```
If any existing row changed, stop — that means the fit isn't as
deterministic as expected; investigate before proceeding.

## 5. Visual QC before trusting anything downstream

- Check `cpmg_fit_results/decays/<sample>_<temp>_decays.png` (raw fits) and
  `cpmg_fit_results/t2_alpha/<sample>_<temp>_t2alpha.png` (Corezzi fit) for
  anything nonphysical (non-monotonic T2, α outside [0,1], degenerate B/a0).
- For the nicer `pub_figs/cpmg_fit.py` version, call the underlying function
  directly with an explicit `stem` rather than running `main()` — its
  standalone entry writes to a **fixed** filename
  (`pub_figs/figures/cpmg_fit.pdf`) and would silently overwrite whatever
  dataset is currently the published example:
  ```bash
  python -c "
  import sys; sys.path.insert(0, 'pub_figs')
  from cpmg_fit import make_cpmg_figure
  make_cpmg_figure('<SAMPLE>', '<TEMP>C', stem='../postreview/<name>/cpmg_fit_<SAMPLE>_<TEMP>C')
  "
  ```

## 6. Kamal-Malkin MCMC fit (`fit_kuro_fixedr.py`)

If the new dataset isn't yet in `epoxy_data_13Mar2026.mat` (it won't be for a
fresh addition), `load_nmr_alpha_t()` in `fit_kuro_fixedr.py` automatically
falls back to reading `cpmg_fit_results/<sample>_<temp>C.csv` directly for any
`(sample, temp)` not in `nmr_index` — no code change needed for this step.

Run it on the cluster (it's compute-heavy — 250,000 MCMC steps per dataset
per `mcmc_config.py`), one dataset at a time via a small standalone SLURM
script (copy `epoxy_NMR_EDA_60C.sh` as a template — **not** the 12-array
`epoxy_NMR_mcmc.sh`). Make sure the script `cd`s into the repo root before
running, since the CSV fallback path is relative.

```bash
# on the cluster, from ~/epoxy_kinetics:
git pull
sbatch epoxy_NMR_<SAMPLE>_<TEMP>C.sh
```

Pull the resulting `.npz` back (it's large — ~1.3 GB per dataset, uncompressed
full chain by design):
```bash
rsync -av <user>@<cluster>:~/epoxy_kinetics/mcmc_samples_nmr/NMR_<SAMPLE>_<TEMP>C_fitdata.npz \
    mcmc_samples_nmr/
```

## 7. Diagnostics — check for convergence problems before trusting the fit

```bash
python MCMC_diagnostics.py mcmc_samples_nmr/NMR_<SAMPLE>_<TEMP>C_fitdata.npz
```
Check R̂ for all 5 params (want ≈1.00–1.02; investigate anything higher) and
look at `diagnostic_plots/NMR_<...>_trace.png` for a walker permanently
offset from the rest of the pack (a stuck-walker pathology seen once with
DAB/60°C — resolved by simply resubmitting the job, since no random seed is
set, giving different walker init positions).

## 8. Post-process into an isolated review folder — don't touch shared files

```bash
mkdir -p postreview/<SAMPLE>_<TEMP>C
python plot_mcmc.py mcmc_samples_nmr/NMR_<SAMPLE>_<TEMP>C_fitdata.npz \
    --outdir  postreview/<SAMPLE>_<TEMP>C/fit_plots \
    --summary postreview/<SAMPLE>_<TEMP>C/posterior_summary_<SAMPLE>_<TEMP>C.csv
```
This never touches `fit_plots_nmr/`, `posterior_summary_NMR_fixedr.csv`, etc.
— review the plots here first.

## 9. Fold into the KM/Arrhenius/Eyring comparison (once satisfied)

```bash
cp postreview/<SAMPLE>_<TEMP>C/posterior_summary_<SAMPLE>_<TEMP>C.csv \
   posterior_summary_parts/posterior_summary_NMR_<SAMPLE>_<TEMP>C.csv
python merge_nmr_parts.py                          # rebuilds posterior_summary_NMR_fixedr.csv
git diff posterior_summary_NMR_fixedr.csv           # verify: only new rows, nothing existing changed
python build_combined_arrhenius.py                  # new dated combined_arrhenius_*.csv
python final_results_to_plots.py combined_arrhenius_<date>.csv   # new timestamped PDFs in results/
```

## 10. Regenerate publication figures/tables into a review folder

Every `pub_figs/*.py` script now supports `--outdir` (added for this exact
purpose) — default behavior with no flag is unchanged, so this never touches
`pub_figs/figures/` or the external manuscript repo:
```bash
mkdir -p postreview/pub_figs_review
for script in arrhenius eyring_analysis ea_bars parameters make_si make_table_km; do
    python pub_figs/$script.py --outdir postreview/pub_figs_review
done
```
If you added a **new temperature** (not just a new sample), check whether
`NMR_TEMPS` in `make_si.py`/`make_table_km.py` needs extending — neither
script infers temperatures from the data. If DAP2/NMR2 doesn't share the new
temperature, keep its own temp list (`NMR2_TEMPS` in `make_si.py`) separate
rather than assuming it matches.

Verify nothing official changed:
```bash
git status pub_figs/figures/                          # should be empty
git -C ../Epoxy-Kinetics-2025 status Ea_table.tex Ea_only_table.tex   # should be empty
```

## 11. When ready to officially adopt into the manuscript

Point the manuscript's `\includegraphics`/`\input` paths at
`postreview/pub_figs_review/...` instead of `pub_figs/figures/...` (they're
absolute paths, so this is a straightforward find-and-replace — see
`postreview/manuscript_incorporation_todo.md` for the exact lines from the
60°C round as a template). Update any hardcoded temperature-range text in
figure/table captions (`eyring_analysis.py`'s `Ea_table.tex` caption
hardcodes the NMR temperature list as literal text) so it doesn't silently
misstate what's now in the table.
