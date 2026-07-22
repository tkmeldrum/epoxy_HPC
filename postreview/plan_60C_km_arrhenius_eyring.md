# Fold 60°C NMR data into KM/Arrhenius/Eyring analysis, without touching existing results

## Context

The three new 60°C NMR MCMC fits (EDA, DAP, DAB) are complete and verified in `postreview/{EDA,DAP,DAB}_60C/`, but isolated from the rest of the pipeline. The next stage — comparing KM parameters across temperature, then Arrhenius/Eyring analysis — reads from `posterior_summary_NMR_fixedr.csv`, which currently has **zero** 60°C rows. Getting the new data into that comparison means touching the downstream `pub_figs/` publication scripts, and exploration found that **every one of them writes to a hardcoded, fixed filename** (`pub_utils.savefig()`, plus two `.tex` files `eyring_analysis.py` writes directly into a sibling manuscript repo `../../Epoxy-Kinetics-2025/`) — there is no existing versioning/collision protection anywhere in that pipeline. Since you'll need this same "compare new data without clobbering the official figures" workflow again in the future, the plan adds a small reusable output-redirection mechanism rather than a one-off workaround.

**DSC data is never written to anywhere in this plan** — `posterior_summary_DSC.csv`, `epoxy_data_13Mar2026.mat`, and every DSC-related script are read-only inputs throughout. Every write in this plan is either a new NMR-only file, a new dated/timestamped file, or routed into the `postreview/` review folder.

## Phase 0 — Save this plan for reference

Copy this plan file itself to `postreview/plan_60C_km_arrhenius_eyring.md`, so it travels alongside the review outputs it describes.

Two tiers of work:
1. **Data merge** (small, low-risk, additive) — fold the 3 new posterior summaries into `posterior_summary_NMR_fixedr.csv` and produce a new dated `combined_arrhenius_*.csv`, then run the already-safe (auto-timestamped) `final_results_to_plots.py` for an immediate first look at KM-parameter/Arrhenius trends.
2. **Publication figures** (`pub_figs/`) — add an opt-in `--outdir` override so `arrhenius.py`, `eyring_analysis.py`, `ea_bars.py`, `parameters.py`, `make_si.py`, `make_table_km.py`, and the standalone `main()`s of `cpmg_fit.py`/`representative.py` can all be re-run into a review folder instead of overwriting `pub_figs/figures/` (or the external manuscript `.tex` files). Default behavior (no flag) is unchanged — fully backward compatible.

## Phase A — Merge the 3 new datasets into `posterior_summary_NMR_fixedr.csv`

`merge_nmr_parts.py` rebuilds this file from scratch each run by globbing `posterior_summary_parts/posterior_summary_NMR_*.csv` (`.tail(1)` of each, concatenated, overwritten to the output file) — it's additive as long as the existing 12 part files stay in place.

1. Copy (don't move) the postreview CSVs into that directory, renamed to match the glob pattern and existing naming convention:
   - `postreview/EDA_60C/posterior_summary_EDA_60C.csv` → `posterior_summary_parts/posterior_summary_NMR_EDA_60C.csv`
   - same for DAP, DAB. Also copy each `.provenance.txt` alongside, matching the existing per-part convention.
   - No content changes needed — the `Label` column inside is already `NMR_EDA_60C` etc.
2. Run `python merge_nmr_parts.py` → regenerates `posterior_summary_NMR_fixedr.csv` (12 → 15 rows).
3. **Verify via `git diff`**: the 12 existing rows must be byte-identical; only 3 new rows appended. (Same verification pattern used throughout this work — if any existing row changed, stop and investigate before committing.)

## Phase B — New `combined_arrhenius_{date}.csv` (fills a real gap)

No script currently produces this file — the one example in the repo (`combined_arrhenius_20260327.csv`) was hand-built and predates `fit_kuro_fixedr.py` entirely. Write a small, permanent, reusable script `build_combined_arrhenius.py` (repo root):

```python
# reads posterior_summary_DSC.csv + posterior_summary_NMR_fixedr.csv
# (both already share the Label/log_k1_median/... schema final_results_to_plots.py expects)
# concatenates them, writes combined_arrhenius_{YYYYMMDD}.csv -- new dated file every run,
# never overwrites a prior date's file
```

This directly addresses the README's "regenerate by re-running the merge inline" gap, and gives you a reusable command for next time instead of another one-off.

## Phase C — Immediate KM-parameter / Arrhenius look (already safe, zero code changes)

`final_results_to_plots.py` already writes minute-timestamped output (`results/fit_trends_{timestamp}.pdf`, `results/arrhenius_fits_{timestamp}.pdf}`) with no hardcoded temperature list — it parses `Label` directly, so 60°C rows flow through automatically:

```bash
python final_results_to_plots.py combined_arrhenius_{date}.csv
```

This is the fastest path to literally answering "how does this impact KM parameters / Arrhenius" — no `pub_figs/` changes required for this first look.

## Phase D — Add `--outdir` support to `pub_utils.py` and every `pub_figs/*.py` script

In `pub_figs/pub_utils.py`: add a module-level override + setter, and route `savefig()`/`write_provenance()` through it:

```python
_OUTPUT_DIR_OVERRIDE = None

def set_output_dir(path):
    global _OUTPUT_DIR_OVERRIDE
    _OUTPUT_DIR_OVERRIDE = path
    os.makedirs(path, exist_ok=True)

def _figures_dir():
    return _OUTPUT_DIR_OVERRIDE or _FIGURES
```

`savefig()` (currently `os.path.join(_FIGURES, name)`) and `write_provenance()` (currently `figures/<script_stem>.provenance.txt`) both switch to `_figures_dir()`. Untouched default (`_OUTPUT_DIR_OVERRIDE is None`) reproduces exactly today's paths.

Then, in each script's entry point, add `--outdir` (argparse) that calls `pu.set_output_dir(args.outdir)` before any figure/table is generated, for:
- `arrhenius.py`, `eyring_analysis.py`, `ea_bars.py`, `parameters.py` — straightforward, all route solely through `pu.savefig`/`pu.write_provenance`.
- `make_si.py`, `make_table_km.py` — same, plus **extend `NMR_TEMPS = [25, 33, 40]` → include `60`** in both (`make_si.py:37`, `make_table_km.py:14`) — a real, permanent correctness fix (neither script infers temperatures from data), not something the `--outdir` flag alone fixes. The per-dataset SI figures (`make_cpmg_figure`/`make_rep_figure`) are already stem-parameterized/safe; only the aggregating `si_figures.tex`/`table_km.tex` + provenance need the redirect.
- `cpmg_fit.py`, `representative.py` — only their standalone `main()`s need `--outdir` (the reusable `make_cpmg_figure()`/`make_rep_figure()` helpers already take an explicit `stem` and are safe as called from `make_si.py`).
- `eyring_analysis.py` specifically: also redirect `LATEX_TABLE_OUT`/`LATEX_EA_ONLY_OUT` (currently hardcoded to `../../Epoxy-Kinetics-2025/Ea_table.tex` and `Ea_only_table.tex`) to land inside the review outdir when `--outdir` is passed, instead of overwriting the live manuscript files.

**Not fixed automatically, flagged for your manual review instead:**
- `eyring_analysis.py`'s LaTeX table captions hardcode the temperature list as literal text (`\qtylist{25;33;40}{\celsius}` for NMR, lines ~268-269/334-335) — these would silently become factually wrong once 60°C NMR data is in the table. This is paper-text content; better for you to update deliberately when you're ready to officially adopt 60°C in the manuscript, not as a side effect of this pass.
- `ea_bars.py` has hand-placed annotation positions/arrows tuned to specific known out-of-range Ea values (lines ~65-80) — adding 60°C data shifts the underlying weighted Arrhenius fit and could make these look wrong without erroring. Worth a visual check in the review output.

## Phase E — Run everything into a review folder

```bash
mkdir -p postreview/pub_figs_60C
for script in arrhenius eyring_analysis ea_bars parameters make_si make_table_km cpmg_fit representative; do
    python pub_figs/$script.py --outdir postreview/pub_figs_60C
done
```

(`cpmg_fit.py`/`representative.py` standalone runs use whatever `SAMPLE`/`TEMP_STR` module constants are currently set — check those first, or call their helper functions directly if you want a specific 60°C dataset rather than the current default.)

## Verification

- After Phase A: `git diff posterior_summary_NMR_fixedr.csv` shows only 3 added rows, 0 changed/deleted.
- After Phase B: `combined_arrhenius_{date}.csv` is a new file; `combined_arrhenius_20260327.csv` untouched (`git status` clean on it).
- After Phase C: new timestamped PDFs in `results/`; nothing else in `results/` touched.
- After Phase D+E: `git status` / `git diff` on `pub_figs/figures/` and the external `../../Epoxy-Kinetics-2025/` directory both show **zero changes** — confirms the redirect actually worked and nothing official was touched. All new output lives under `postreview/pub_figs_60C/`.
- Visual spot-check: `postreview/pub_figs_60C/arrhenius.pdf`/`eyring.pdf`/`parameters.pdf` show 60°C points alongside the existing 25/33/40°C NMR series and full DSC range; `ea_bars.pdf` checked for annotation sanity per the flag above.

## After the plan: your review checklist

Everything above lands in `postreview/pub_figs_60C/` plus the two new root-level files (`posterior_summary_NMR_fixedr.csv` update, `combined_arrhenius_{date}.csv`) and new `results/*_{timestamp}.pdf`. Nothing official is touched yet. To review:

1. **`results/fit_trends_{timestamp}.pdf`** — do EDA/DAP/DAB's k1, k2, m, n at 60°C extend the existing 25–40°C trends sensibly, or jump/reverse unexpectedly?
2. **`results/arrhenius_fits_{timestamp}.pdf`** — do the new points fall near the existing Arrhenius line (ln k vs 1/T), or pull the fit/Ea noticeably? Compare Ea before/after by eye.
3. **`postreview/pub_figs_60C/parameters.pdf`** — the polished version of #1; same check.
4. **`postreview/pub_figs_60C/arrhenius.pdf` + `eyring.pdf`** — same check as #2, plus Eyring's ΔH‡/ΔS‡ output.
5. **`postreview/pub_figs_60C/ea_bars.pdf`** — check the hand-placed annotation arrows/asterisks still land sensibly (flagged above as a known fragile spot).
6. **`postreview/pub_figs_60C/table_km.tex` / `si_figures.tex`** — confirm 60°C rows/figures appear and look complete for all three samples.
7. **DAB's wide `log_k1` CI** (from the stuck-walker MCMC chain) — check whether it visibly widens DAB's point in the Arrhenius/Eyring plots enough to matter for the fitted Ea, now that it's sitting alongside the other temperatures rather than viewed alone.

## If you decide to incorporate the results officially

None of this is done automatically — each is a deliberate, separate action:

1. Commit `posterior_summary_parts/posterior_summary_NMR_{EDA,DAP,DAB}_60C.csv` (+ `.provenance.txt`) and the regenerated `posterior_summary_NMR_fixedr.csv`.
2. Commit `build_combined_arrhenius.py` and the `combined_arrhenius_{date}.csv` it produced (or regenerate fresh at commit time).
3. **Preferred over re-running scripts without `--outdir`**: point the manuscript at the review output directly, since `../Epoxy-Kinetics-2025/`'s `.tex` files already reference figures by **absolute filesystem path** rather than a local copy:
   - `main.tex:170,191,199` and matching lines in `SI.tex`, `redline.tex`, `rev1.tex` — `\includegraphics{/Users/tyler/Documents/GitHub/epoxy_HPC/pub_figs/figures/{parameters,arrhenius,ea_bars}.pdf}` → change the `pub_figs/figures` segment to `postreview/pub_figs_60C`.
   - `SI.tex:85` — `\input{/Users/tyler/Documents/GitHub/Epoxy-Kinetics-2025/Ea_table.tex}` — this file is written directly by `eyring_analysis.py`'s `LATEX_TABLE_OUT`; with `--outdir postreview/pub_figs_60C` it lands at `postreview/pub_figs_60C/Ea_table.tex` instead, so update this `\input` path to match.
   - This avoids ever touching `pub_figs/figures/` for real, and is trivially reversible (revert the path edits) if the 60°C data doesn't pan out — cleaner than the overwrite-then-commit approach.
4. For `eyring_analysis.py`'s `.tex` output specifically: still manually update the hardcoded caption temperature-list text (flagged above) so the table caption matches whatever temperature range you end up pointing the manuscript at.
5. Optionally extend `nmr_temps = [25, 33, 40]` in `fit_kuro_fixedr.py`/`fit_kuro.py` to include `60`, and the 12→15-element arrays in the batch SLURM scripts (`epoxy_NMR_mcmc.sh` etc.) — only needed if you want future *batch* ("run everything") re-fits to automatically include 60°C; your per-dataset SLURM scripts already work fine without this.
6. Update the README's CPMG/Corezzi "Observed trends" section with any new temperature-dependence conclusions from the 60°C data, same pattern as the existing entries.

## Reference: current state snapshot (before any of this plan runs)

For comparison after the new analysis. All paths relative to `epoxy_HPC/` unless noted.

**`pub_figs/figures/`** (official, published figures — untouched by this plan):
`arrhenius.{pdf,png}` (Jun 5), `eyring.{pdf,png}` (Jun 5), `ea_bars.{pdf,png}` (May 5), `parameters.{pdf,png}` (May 18), `cpmg_fit.{pdf,png}` (Jun 5, currently shows EDA/25C per its module-level default), `representative.{pdf,png}` (Apr 27), `si_figures.tex` + `table_km.tex` (May 18), plus 7 stale `data_YYYYMMDD/` input snapshots (Apr 24 – Jun 5) and `SI_figures/` (86 files: `cpmg_fit_{sample}_{25,33,40}C.*` for DAB/DAP/DAP2/EDA, `rep_DSC_{sample}_{temp}.*` for all 6 DSC temps × 3 samples — **no 60°C entries yet**).

**`results/`**: 14 `arrhenius_fits_{timestamp}.pdf` + 14 `fit_trends_{timestamp}.pdf`, oldest `20250724_1745`, newest `20260504_2128` (Apr 20 – May 5) — all pre-date this work.

**Root-level CSVs**:
- `posterior_summary_NMR_fixedr.csv` — 12 rows (DAB/DAP/DAP2/EDA × 25/33/40C), last modified May 5.
- `posterior_summary_DSC.csv` — 18 rows, includes 60°C already (unaffected by this plan).
- `posterior_summary.csv` (14668 bytes), `posterior_summary_combined.csv` (10267 bytes), `posterior_summary_NMR_28Mar.csv` — older/alternate files, not directly touched by this plan.
- `combined_arrhenius_20260327.csv` — the one existing hand-built example (18 DSC + 12 NMR rows), predates `fit_kuro_fixedr.py`'s DAP2/60C work.

**`posterior_summary_parts/`**: 12 CSV + 12 `.provenance.txt` pairs (DAB/DAP/DAP2/EDA × 25/33/40C), all dated May 5 — this plan adds 3 more pairs for 60C, existing 12 untouched.

**`../Epoxy-Kinetics-2025/`** (sibling manuscript repo, separate git history):
- `Ea_table.tex` (2025 bytes) + `Ea_only_table.tex` (1096 bytes) — written directly by `eyring_analysis.py`, last modified Jun 5.
- `main.tex` (Jun 9), `SI.tex`, `redline.tex` (Jul 17), `rev1.tex` (Jul 17) — reference `pub_figs/figures/*.pdf` and `Ea_table.tex` by absolute path (see incorporation step 3 above for exact line numbers).
- `Figures/` — a separate, manually-curated set of figures (`{DAB,DAP,EDA}_{K1,K2,k_b,lnK1,lnK2,m_n}.pdf`, etc.) that appear hand-copied/renamed rather than generated by `pub_figs/`, unrelated to this plan.
- `bundle_submission.py` — a submission-packaging script that copies figures from `pub_figs/figures` into a build output dir; irrelevant unless/until you're preparing an actual submission bundle.
