"""Summarize test results with training seed as the unit of replication."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ('social_speed', 'social_speed_variance', 'emergency_travel_time',
           'lane_changes_normalized')


def read_rows(root):
    rows = []
    seen_rules = set()
    for path in root.glob('*/seed_*/episodes.csv'):
        group, train_seed = path.parent.parent.name, int(path.parent.name.split('_')[-1])
        with path.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                row.update(group=group, training_seed=train_seed,
                           n_cars=int(row['n_cars']), scene_seed=int(row['seed']))
                if row['strategy'] in ('AlwaysClose', 'AlwaysOpen', 'DensityEmergencyThreshold'):
                    identity = (group, row['strategy'], row['n_cars'], row['scene_seed'])
                    if identity in seen_rules:
                        continue
                    seen_rules.add(identity)
                    row['training_seed'] = 0
                rows.append(row)
    return rows


def summarize(rows):
    by_seed = defaultdict(list)
    references = {(r['group'], r['n_cars'], r['scene_seed']): r for r in rows
                  if r['strategy'] == 'DensityEmergencyThreshold'}
    for row in rows:
        by_seed[(row['group'], row['strategy'], row['n_cars'],
                 row['training_seed'])].append(row)
    seed_means = []
    for (group, strategy, density, seed), records in sorted(by_seed.items()):
        for metric in METRICS:
            values = [float(r[metric]) for r in records if r[metric] not in ('', None)]
            complete = len(values) == len(records)
            seed_means.append({'group': group, 'strategy': strategy, 'n_cars': density,
                               'training_seed': seed, 'metric': metric,
                               'n_scenes': len(records), 'n_complete_scenes': len(values),
                               'mean': float(np.mean(values)) if complete and values else ''})
            if strategy != 'DensityEmergencyThreshold' and metric in METRICS[:3]:
                differences = []
                for record in records:
                    reference = references.get((group, density, record['scene_seed']))
                    if reference and record[metric] not in ('', None) and reference[metric] not in ('', None):
                        differences.append(float(record[metric]) - float(reference[metric]))
                seed_means.append({'group': group, 'strategy': strategy, 'n_cars': density,
                                   'training_seed': seed, 'metric': metric + '_vs_threshold',
                                   'n_scenes': len(records), 'n_complete_scenes': len(differences),
                                   'mean': float(np.mean(differences))
                                   if len(differences) == len(records) and differences else ''})
    grouped = defaultdict(list)
    for row in seed_means:
        grouped[(row['group'], row['strategy'], row['n_cars'], row['metric'])].append(row)
    aggregate = []
    for (group, strategy, density, metric), records in sorted(grouped.items()):
        values = [float(r['mean']) for r in records if r['mean'] != '']
        complete = len(values) == len(records)
        mean = float(np.mean(values)) if complete and values else None
        sd = float(np.std(values, ddof=1)) if complete and len(values) > 1 else None
        critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
                    6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
                    10: 2.262}.get(len(values), 1.96)
        half = critical * sd / np.sqrt(len(values)) if sd is not None else None
        aggregate.append({'group': group, 'strategy': strategy, 'n_cars': density,
                          'metric': metric, 'n_training_seeds': len(records),
                          'n_complete_seeds': len(values),
                          'mean_across_seeds': mean if mean is not None else '',
                          'sd_across_seeds': sd if sd is not None else '',
                          'ci95_low': mean - half if half is not None else '',
                          'ci95_high': mean + half if half is not None else ''})
    return seed_means, aggregate


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_root', type=Path)
    args = parser.parse_args()
    rows = read_rows(args.results_root)
    if not rows:
        parser.error('No evaluation episodes.csv files found')
    seed_means, aggregate = summarize(rows)
    write_csv(args.results_root / 'training_seed_means.csv', seed_means)
    write_csv(args.results_root / 'across_training_seeds.csv', aggregate)


if __name__ == '__main__':
    main()
