"""Plot measured paper_experiments.py results; incomplete trips are annotated."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def number(value):
    return float(value) if value not in ('', None) else np.nan


def grouped(rows, out):
    densities = sorted({int(r['n_cars']) for r in rows})
    strategies = sorted({r['strategy'] for r in rows})
    metrics = [('social_speed', 'Social speed (cells/step)'),
               ('social_speed_variance', 'Speed variance'),
               ('emergency_travel_time', 'Emergency travel time (steps)')]
    x = np.arange(len(densities))
    width = 0.8 / len(strategies)
    def draw_metric(ax, metric, label):
        for j, strategy in enumerate(strategies):
            selected = [next((r for r in rows if int(r['n_cars']) == d and r['strategy'] == strategy), None)
                        for d in densities]
            means = [number(r.get(metric + '_mean')) if r else np.nan for r in selected]
            sds = [number(r.get(metric + '_sd')) if r else np.nan for r in selected]
            errors = np.nan_to_num(sds, nan=0)
            ax.bar(x + (j - (len(strategies)-1)/2)*width, means, width,
                   yerr=errors, label=strategy, capsize=2)
            if metric == 'emergency_travel_time':
                for k, row in enumerate(selected):
                    if row and int(row['emergency_completed']) < int(row['n']):
                        ax.text(x[k] + (j - (len(strategies)-1)/2)*width,
                                means[k] if np.isfinite(means[k]) else 0,
                                f"{row['emergency_completed']}/{row['n']}",
                                rotation=90, fontsize=7, ha='center', va='bottom')
        ax.set_xticks(x, densities)
        ax.set_xlabel('Social vehicles')
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.2)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for ax, (metric, label) in zip(axes, metrics):
        draw_metric(ax, metric, label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(len(labels), 4),
               bbox_to_anchor=(.5, 1.05), fontsize=8)
    fig.savefig(out / 'figure1_metrics.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    draw_metric(ax, 'lane_changes_normalized', 'Lane changes / 100 cars / 100 steps')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.13),
              ncol=min(len(strategies), 4), fontsize=8)
    fig.savefig(out / 'figure5_lane_changes.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for d in densities:
        selected = [r for r in rows if int(r['n_cars']) == d]
        for r in selected:
            sx, ty = number(r['social_speed_mean']), number(r['emergency_travel_time_mean'])
            if np.isfinite(sx) and np.isfinite(ty):
                ax.scatter(sx, ty, label=f"{d}: {r['strategy']}")
                ax.annotate(f"{d} {r['strategy']}", (sx, ty), fontsize=7, xytext=(3, 3),
                            textcoords='offset points')
    ax.set_xlabel('Social speed (cells/step)')
    ax.set_ylabel('Emergency travel time (steps; completed only)')
    ax.grid(alpha=.2)
    fig.savefig(out / 'figure2_tradeoff.png', dpi=180)
    plt.close(fig)


def process(traces, out):
    episodes = traces['episodes']
    for density in sorted({e['n_cars'] for e in episodes}):
        seed = min(e['seed'] for e in episodes if e['n_cars'] == density)
        chosen = [e for e in episodes if e['n_cars'] == density and e['seed'] == seed]
        fig, axes = plt.subplots(len(chosen), 2, figsize=(14, 3.2*len(chosen)),
                                 constrained_layout=True, squeeze=False)
        for row, episode in zip(axes, chosen):
            trace = episode['trace']
            steps = [r['step'] for r in trace]
            actions = np.asarray([r['actions'] for r in trace]).T
            row[0].imshow(actions, aspect='auto', interpolation='nearest', origin='lower',
                          extent=[steps[0], steps[-1], -.5, actions.shape[0]-.5],
                          vmin=0, vmax=1, cmap='Blues')
            row[0].set_ylabel('Control segment')
            row[0].set_title(f"{episode['strategy']}: open=blue; emergency distance=red")
            position = [r['emergency_cumulative_distance'] for r in trace]
            second = row[0].twinx()
            second.plot(steps, position, color='crimson', linewidth=.8)
            second.set_ylabel('Emergency cumulative distance (cells)')
            changes = [r['lane_changes'] for r in trace]
            variance = [r['social_speed_variance'] for r in trace]
            row[1].plot(steps, changes, label='Lane changes', linewidth=.9)
            another = row[1].twinx()
            another.plot(steps, variance, color='darkorange', label='Speed variance', linewidth=.9)
            switches = [r['step'] for before, r in zip(trace, trace[1:])
                        if before['actions'] != r['actions']]
            for switch in switches:
                row[1].axvline(switch, color='grey', alpha=.18, linewidth=.6)
            row[1].set_ylabel('Lane changes / step')
            another.set_ylabel('Speed variance')
            row[1].set_title(f"{episode['strategy']}: grey lines = action changes")
            for ax in row:
                ax.set_xlabel('Simulation step')
        fig.savefig(out / f'figure4_and_6_density_{density}_seed_{seed}.png', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, help='Directory written by paper_experiments.py')
    args = parser.parse_args()
    rows = read_csv(args.results / 'summary.csv')
    with (args.results / 'traces.json').open(encoding='utf-8') as handle:
        traces = json.load(handle)
    grouped(rows, args.results)
    process(traces, args.results)


if __name__ == '__main__':
    main()
