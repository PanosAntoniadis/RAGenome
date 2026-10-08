"""
Compute the training windows of every pretraining stage.

Windows of `--window` bp are placed every `--stride` bp (half a window) along every chromosome.
Every window is scored by the 75th percentile of the PhastCons 100-way conservation scores it
contains; windows with more than 50% missing scores are dropped. Selection: the `--top_frac`
most conserved windows plus a random 0.1% of the remaining ones.
"""

import argparse
import numpy as np
import pandas as pd
import pyBigWig
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

PERC      = 75
RAND_FRAC = 0.001
MAX_NAN_FRAC = 0.5
CHUNK     = 20_000_000  # 20 Mb per fetch; must be > WINDOW

CHROMS = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']


def process_chrom(bw, chrom, chrom_len, window, stride):
    rows = []
    # Process in overlapping chunks so windows at boundaries aren't missed.
    # Each chunk fetches CHUNK bp; windows start every STRIDE within the chunk.
    # We advance chunk_start by (CHUNK - WINDOW + STRIDE) so no window is split.
    step = CHUNK - window + stride

    pos = 0
    while pos < chrom_len:
        fetch_end = min(pos + CHUNK, chrom_len)
        vals = bw.values(chrom, pos, fetch_end, numpy=True).astype(np.float32)

        # sliding_window_view gives shape (N, window) where N = len(vals)-window+1
        if len(vals) < window:
            break
        wins = sliding_window_view(vals, window)[::stride]  # (K, window)

        nan_frac = np.isnan(wins).mean(axis=1)              # (K,)
        valid    = nan_frac <= MAX_NAN_FRAC

        if valid.any():
            scores = np.nanpercentile(wins[valid], PERC, axis=1)
            starts = pos + np.where(valid)[0] * stride
            ends   = starts + window
            for s, e, sc in zip(starts, ends, scores):
                if e <= chrom_len:
                    rows.append((chrom, int(s), int(e), float(sc)))

        pos += step

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bw', required=True, help='hg38.phastCons100way.bw (UCSC)')
    parser.add_argument('--out', required=True, help='output BED path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--window', type=int, default=128)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--top_frac', type=float, default=0.05)
    parser.add_argument('--dump_pool', default=None,
                         help='if set, also write the FULL valid-window candidate pool '
                              '(before any top_frac selection) to this bed path.')
    args = parser.parse_args()

    bw = pyBigWig.open(args.bw)
    chrom_sizes = {k: v for k, v in bw.chroms().items() if k in CHROMS}

    all_rows = []
    for chrom in CHROMS:
        if chrom not in chrom_sizes:
            continue
        print(f'  {chrom} ({chrom_sizes[chrom]/1e6:.0f} Mb)...', flush=True)
        rows = process_chrom(bw, chrom, chrom_sizes[chrom], args.window, args.stride)
        all_rows.extend(rows)
        print(f'    {len(rows):,} valid windows', flush=True)

    bw.close()

    df = pd.DataFrame(all_rows, columns=['chrom', 'start', 'end', 'phastcons_p75'])
    print(f'\nTotal valid windows: {len(df):,}')

    if args.dump_pool:
        pool_out = Path(args.dump_pool)
        pool_out.parent.mkdir(parents=True, exist_ok=True)
        pool_df = df.copy()
        pool_df['label'] = 1
        pool_df.to_csv(pool_out, sep='\t', index=False, header=False,
                        columns=['chrom', 'start', 'end', 'phastcons_p75', 'label'])
        print(f'Full candidate pool ({len(pool_df):,} windows) saved: {pool_out}')

    threshold = df['phastcons_p75'].quantile(1 - args.top_frac)
    top_mask  = df['phastcons_p75'] >= threshold
    top_df    = df[top_mask].copy()
    top_df['is_conserved'] = 1

    rest_df  = df[~top_mask]
    n_rand   = max(1, int(len(rest_df) * RAND_FRAC)) if len(rest_df) > 0 else 0
    rand_df  = rest_df.sample(n=n_rand, random_state=args.seed).copy() if n_rand > 0 else rest_df.copy()
    rand_df['is_conserved'] = 0

    selected = pd.concat([top_df, rand_df], ignore_index=True)
    selected.sort_values(['chrom', 'start'], inplace=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out, sep='\t', index=False, header=False,
                    columns=['chrom', 'start', 'end', 'phastcons_p75', 'is_conserved'])

    print(f'Top-{args.top_frac*100:.0f}%% threshold (PhastCons p75): {threshold:.4f}')
    print(f'Conserved: {len(top_df):,}  Random: {len(rand_df):,}  Total: {len(selected):,}')
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
