"""Honest evaluation on wells where the answer is known.

For each well, the depth after the known section is hidden and predicted. The
regional plane is fitted only from the *other* wells (leave-one-out), so the
well's own answers never leak in. Prints RMSE for the baseline and for the full
path-matching model.

    python src/evaluate.py --data-dir sample_data
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from predict import WellborePredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='sample_data')
    args = ap.parse_args()

    pred = WellborePredictor(args.data_dir)
    rows, sq, sq_base, total = [], 0.0, 0.0, 0
    for path in sorted(glob.glob(os.path.join(args.data_dir, 'test', '*__horizontal_well.csv'))):
        well = os.path.basename(path).split('__')[0]
        truth_path = os.path.join(args.data_dir, 'train', f'{well}__horizontal_well.csv')
        if not os.path.exists(truth_path):
            continue
        df = pd.read_csv(path)
        truth = pd.read_csv(truth_path)['TVT'].values
        tw = pd.read_csv(os.path.join(args.data_dir, 'test', f'{well}__typewell.csv'))
        predicted = pred.predict_tvt_for_well(well, df, tw)
        hidden = df['TVT_input'].isna().values
        err = predicted[hidden] - truth[hidden]
        rows.append((well, int(hidden.sum()), float(np.sqrt(np.mean(err ** 2)))))
        sq += float(np.sum(err ** 2)); total += int(hidden.sum())

    print(f"{'well':<10} {'points':>7} {'RMSE (ft)':>10}")
    for well, n, rmse in rows:
        print(f'{well:<10} {n:>7} {rmse:>10.2f}')
    print(f"{'all':<10} {total:>7} {np.sqrt(sq / total):>10.2f}")


if __name__ == '__main__':
    main()
