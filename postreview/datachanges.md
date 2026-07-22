# What changed beyond "collected new data and ran it"

Four things came up during this work that go beyond simply adding EDA/DAP/DAB
at 60°C. Three are code-correctness fixes; the fourth is a data-handling
lesson worth keeping in mind. None of them require retracting or correcting
anything currently in the manuscript — see the "does this affect the
manuscript" note under each.

## 1. CPMG stretched-exponential fit bug (cpmg_batch_fit.py)

**What was wrong:** `_fit_echo_array`'s initial-guess logic computed the
amplitude guess (`A0`) as `max()` over the *entire* echo array, and located
the half-max crossing via `np.searchsorted` on `-y` — which silently assumes
`y` is sorted. Once real T2 signal has fully decayed and the rest of a scan's
echo train is noise, both assumptions break: a late-time noise spike could be
mistaken for the initial amplitude, and the "half-max crossing" could land on
an arbitrary point in the noise. The existing drop criteria (T2 relative
uncertainty, β range) didn't always catch this — a spurious fit could look
falsely confident.

**Fix:** amplitude guess restricted to the first 15% of points; half-max
crossing found as the *first* index where `y` actually drops below it
(floored at half an echo spacing), rather than via `searchsorted`.

**What it changed:** Re-running the full CPMG batch with this fix reproduced
every existing dataset to floating-point noise (~1e-9 relative) *except*:
- **EDA/40°C**: `T2_0` (the Corezzi reference value) dropped from ≈0.112 s to
  ≈0.037 s. The README had documented this 0.112 s value as an unexplained
  "anomaly... ~3× other samples at 40°C." It turned out to be exactly this
  bug: a single late-time, fully-vitrified scan (elapsed 132.9 min) got a
  spuriously large T2 fit that slipped past the drop criteria and became the
  series' `T2_0` reference. The old fit also produced a nonphysical α ≈ −2
  point in the Corezzi R2(α) plot; the new fit correctly flags that scan as
  unreliable (β pegged at the upper bound, `dropped=True`).
- **DAP/40°C**: smaller, real shift in `B`/`a0` (same underlying cause, milder).

**Does this affect the manuscript?** No. The affected quantities are
`T2_0`/`B`/`a0` from the Corezzi R2(α) fit
(`cpmg_fit_results/t2_alpha_fits.csv`) — checked, and none of these appear
anywhere in `main.tex`/`SI.tex`/`redline.tex`/`rev1.tex`/`Ea_table.tex`
("Corezzi" only appears as a literature citation). The manuscript's KM/
Arrhenius/Eyring numbers come from a separately-built `.mat` file via
`fit_kuro_fixedr.py`, which this bug never touched. **If you ever add a
Corezzi T2(α) table/figure to the manuscript, use the corrected values.**

## 2. DAB/60°C MCMC — one stuck walker, resolved by rerunning

The first MCMC run for DAB/60°C had 1 of 64 walkers permanently parked in a
different region of parameter space for the whole post-burnin run (visible in
the trace plot, especially `log_sigma`), inflating Gelman-Rubin R̂ to 1.04–1.08
(vs ~1.001 for EDA/DAP at the same temperature). No random seed is set
anywhere in `fit_kuro_fixedr.py`, so simply resubmitting the identical job
gave different walker starting positions and resolved it (R̂ 1.006–1.02 on
the rerun).

Worth knowing: fixing the convergence issue did **not** meaningfully narrow
DAB's `log_k1` credible interval (essentially unchanged, ~6 log units wide
either way). That wide uncertainty is a genuine feature of the data — NMR
poorly constrains k1 in general (documented elsewhere in the README) — not an
artifact of the stuck walker. The stuck walker was a convergence problem
worth fixing on principle (an R̂-failing chain isn't trustworthy regardless of
whether the answer "looks" similar), not a result-accuracy problem.

**Does this affect the manuscript?** No — this is entirely new data, not a
correction to anything existing.

## 3. MCMC_diagnostics.py — arviz 1.0 API break (tooling only)

`arviz` was upgraded to 1.0 at some point, which rewrote `from_dict()` around
a new DataTree backend and removed the old `posterior=` keyword argument.
This silently crashed `MCMC_diagnostics.py` before it could compute R̂ or
produce trace/diagnostic plots — for *any* dataset, not just the new one.
Fixed by wrapping the dict as `{"posterior": ...}` per the new API (two call
sites). No scientific results are affected; this only restores the ability to
compute convergence diagnostics at all with the currently-installed arviz.

## 4. Two latent (never-triggered) table-generation bugs

Extending `pub_figs/make_table_km.py` and `pub_figs/make_si.py`'s
`NMR_TEMPS` list from `[25, 33, 40]` to include `60` exposed two bugs that
were always present but never manifested, because the hardcoded numbers
happened to equal `len(NMR_TEMPS) == 3` before:
- `make_table_km.py` hardcoded `\multirow{3}{*}{NMR}` and `6 + 3 + n_nmr2` for
  row-span counts instead of computing them from `len(NMR_TEMPS)` — would have
  produced a LaTeX table with misaligned `\multirow` spans once a 4th
  temperature was added. Now computed from actual data availability per
  sample (DAP2/NMR2 was never re-measured at 60°C, so its row count
  legitimately differs from EDA/DAB's NMR row count — the fix accounts for
  this rather than assuming they match).
- `make_si.py`'s `NMR2_SETS`/`CPMG_SETS` assumed the DAP2 replicate shares
  every entry in `NMR_TEMPS` — crashed (`IndexError`) once `NMR_TEMPS`
  included 60°C, since no DAP2/60°C raw data exists. Fixed by giving DAP2 its
  own `NMR2_TEMPS = [25, 33, 40]` list.

**Does this affect the manuscript?** No — since these only manifest with 4+
NMR temperatures, every existing published table (limited to 25/33/40°C) was
generated correctly.

## 5. Data-provenance lesson: OneDrive silently strips acquisition timestamps

The first attempt at getting the raw EDA/60°C data (via a OneDrive copy) had
all 900 files' timestamps collapsed into an 18-second window — OneDrive (or
whatever synced it there) recorded upload time, not original acquisition
time. Since the whole α(t) cure-kinetics analysis depends on real per-scan
timestamps, this would have silently produced meaningless results if not
caught. Re-copied from the source instrument's own file share instead, with
timestamps verified (spread over hours, matching expected scan cadence)
before zipping. Worth remembering for any future data collection that passes
through OneDrive.
