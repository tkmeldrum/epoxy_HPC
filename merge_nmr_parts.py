#!/usr/bin/env python3
"""Merge per-task NMR posterior CSVs into a single summary file.

Run from the repo root:
    python merge_nmr_parts.py
"""
import glob, os
import pandas as pd

parts = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'posterior_summary_parts', 'posterior_summary_NMR_*.csv')))
if not parts:
    raise FileNotFoundError('No per-task CSVs found in posterior_summary_parts/')

df = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'posterior_summary_NMR_fixedr.csv')
df.to_csv(out, index=False)
print(f'Wrote {len(df)} rows to {out}')
print(f'Columns: {list(df.columns)}')
