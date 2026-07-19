"""evaluate.py — Consolidate results from all runs into a single summary table.

Usage
-----
cd Salt-Segmentation-UNet
python evaluate.py
# Output: ../results/summary.csv
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = os.path.join('..', 'results')


def load_results() -> pd.DataFrame:
    """Read all result.csv files from individual run directories."""
    csv_files = glob.glob(os.path.join(RESULTS_DIR, '**', 'result.csv'), recursive=True)
    if not csv_files:
        print('[WARN] No result.csv files found. Run train.py first.')
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in csv_files]
    return pd.concat(dfs, ignore_index=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std per scenario (across seeds)."""
    rows = []
    for scenario in sorted(df['scenario'].unique()):
        grp = df[df['scenario'] == scenario]
        for n_real in sorted(grp['n_real'].unique()):
            sub = grp[grp['n_real'] == n_real]
            rows.append({
                'scenario':       scenario,
                'n_real':         int(n_real),
                'n_synth':        int(sub['n_synth'].iloc[0]),
                'n_seeds':        len(sub),
                'test_iou_mean':  round(sub['test_iou'].mean(),  4),
                'test_iou_std':   round(sub['test_iou'].std(),   4),
                'test_dice_mean': round(sub['test_dice'].mean(), 4),
                'test_dice_std':  round(sub['test_dice'].std(),  4),
            })
    return pd.DataFrame(rows)


def paired_ttest(df: pd.DataFrame) -> None:
    """Print paired t-test (A vs B) for each n_real value."""
    print('\n--- Paired t-test: Scenario A vs B ---')
    for n_real in sorted(df['n_real'].unique()):
        a_vals = df[(df['scenario'] == 'A') & (df['n_real'] == n_real)]['test_iou'].values
        b_vals = df[(df['scenario'] == 'B') & (df['n_real'] == n_real)]['test_iou'].values
        if len(a_vals) < 2 or len(b_vals) < 2 or len(a_vals) != len(b_vals):
            print(f'  n_real={n_real}: insufficient data for t-test '
                  f'(A={len(a_vals)}, B={len(b_vals)} samples)')
            continue
        t_stat, p_val = stats.ttest_rel(b_vals, a_vals)
        delta = b_vals.mean() - a_vals.mean()
        sig   = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else
                ('*' if p_val < 0.05 else 'n.s.'))
        print(f'  n_real={n_real:5d}: ΔIoU={delta:+.4f}  t={t_stat:.3f}  '
              f'p={p_val:.4f} {sig}')


def main():
    df = load_results()
    if df.empty:
        return

    print('\n=== All individual runs ===')
    print(df.to_string(index=False))

    summary = summarise(df)
    print('\n=== Summary (mean ± std across seeds) ===')
    print(summary.to_string(index=False))

    out_path = os.path.join(RESULTS_DIR, 'summary.csv')
    summary.to_csv(out_path, index=False)
    print(f'\n[INFO] Summary saved → {out_path}')

    # Manuscript table format
    print('\n=== Manuscript table (LaTeX-ready) ===')
    for _, row in summary.iterrows():
        label = (f"Scenario {row['scenario']} — Real only (N={row['n_real']})"
                 if row['scenario'] == 'A'
                 else f"Scenario {row['scenario']} — Real+Synth (N={row['n_real']}+{row['n_synth']})")
        print(f"  {label:<50} | IoU={row['test_iou_mean']:.4f}±{row['test_iou_std']:.4f} "
              f"| Dice={row['test_dice_mean']:.4f}±{row['test_dice_std']:.4f}")

    paired_ttest(df)


if __name__ == '__main__':
    main()
