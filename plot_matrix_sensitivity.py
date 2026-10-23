"""Plot measured ablation and emergency-weight sensitivity (manuscript Fig. 7)."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_root', type=Path,
                        help='Directory containing across_training_seeds.csv')
    args = parser.parse_args()
    with (args.results_root / 'across_training_seeds.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for prefix, name in (('ablation_', 'figure7_ablation.png'),
                         ('emergency_weight_', 'figure7_emergency_weight.png')):
        selected = [r for r in rows if r['group'].startswith(prefix)
                    and r['strategy'] == 'MADDPG' and int(r['n_cars']) == 520]
        groups = sorted({r['group'] for r in selected})
        if not groups:
            continue
        metrics = [('social_speed', 'Social speed (cells/step)'),
                   ('emergency_travel_time', 'Emergency travel time (steps)')]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
        for ax, (metric, label) in zip(axes, metrics):
            values = []
            for group in groups:
                record = next((r for r in selected if r['group'] == group and r['metric'] == metric), None)
                values.append(float(record['mean_across_seeds'])
                              if record and record['mean_across_seeds'] else np.nan)
            x = np.arange(len(groups))
            ax.plot(x, values, 'o-', linewidth=1.4)
            ax.set_xticks(x, [g.removeprefix(prefix) for g in groups], rotation=35, ha='right')
            ax.set_ylabel(label)
            ax.set_xlabel('Reward configuration' if prefix == 'ablation_' else 'Emergency weight')
            ax.grid(alpha=.2)
            if not np.any(np.isfinite(values)):
                ax.text(.5, .5, 'No complete travel-time group', transform=ax.transAxes,
                        ha='center', va='center')
        fig.savefig(args.results_root / name, dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()
