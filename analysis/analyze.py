#!/usr/bin/env python
"""Merge the sharded evaluation outputs and compute paired statistics.

The evaluation runs every strategy on the SAME list of episode seeds, so the
per-episode rows can be matched seed-by-seed.  Comparing paired differences is
far more sensitive than comparing two independent means, because the dominant
source of variance (the randomly drawn traffic scenario) cancels out.

    python analysis/analyze.py --build
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = 'MADDPG_argmax'

try:
    from scipy import stats as _scipy_stats
    HAVE_SCIPY = True
except Exception:                                   # noqa: BLE001
    HAVE_SCIPY = False


def load(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def save(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def merge_glob(pattern, out, keep_meta_from=None):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    merged, meta = {}, None
    for fp in files:
        doc = load(fp)
        if isinstance(doc, dict) and 'results' in doc:
            merged.update(doc['results'])
            if meta is None or fp == keep_meta_from:
                meta = doc.get('meta')
        else:
            merged.update(doc)
    payload = {'meta': meta or {}, 'results': merged}
    save(out, payload)
    print(f"[merge] {len(files)} file(s) -> {out}  ({len(merged)} entries)")
    return payload


# two-sided 95% Student-t critical values, indexed by degrees of freedom
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 14: 2.145, 16: 2.120,
        18: 2.101, 20: 2.086, 24: 2.064, 28: 2.048, 30: 2.042, 40: 2.021,
        50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980}


def t_crit(n, alpha=0.05):
    """Two-sided t critical value; falls back to a table, then to the normal."""
    if n < 2:
        return float('nan')
    if HAVE_SCIPY:
        return float(_scipy_stats.t.ppf(1 - alpha / 2, n - 1))
    df = n - 1
    if df in _T95:
        return _T95[df]
    ks = sorted(_T95)
    if df < ks[0]:
        return _T95[ks[0]]
    if df > ks[-1]:
        return 1.96                  # normal limit
    lo = max(k for k in ks if k < df)          # linear interpolation
    hi = min(k for k in ks if k > df)
    f = (df - lo) / (hi - lo)
    return _T95[lo] + f * (_T95[hi] - _T95[lo])


def p_value(d):
    if not HAVE_SCIPY or len(d) < 2:
        return float('nan')
    return float(_scipy_stats.ttest_1samp(d, 0.0).pvalue)


def paired_vs_reference(results, metric, reference=REFERENCE):
    if reference not in results:
        return {}
    ref = {r['seed']: r for r in results[reference]['_episodes']}
    out = {}
    for name, rec in results.items():
        if name == reference:
            continue
        other = {r['seed']: r for r in rec['_episodes']}
        seeds = sorted(set(ref) & set(other))
        d = []
        for s in seeds:
            a, b = ref[s].get(metric), other[s].get(metric)
            if a is None or b is None:
                continue
            d.append(float(a) - float(b))
        if len(d) < 2:
            continue
        d = np.array(d)
        n = len(d)
        sem = float(np.std(d, ddof=1) / np.sqrt(n))
        tc = t_crit(n)
        out[name] = {
            'metric': metric,
            'n_pairs': n,
            'mean_diff': float(np.mean(d)),
            'sem_diff': sem,
            'ci95_low': float(np.mean(d) - tc * sem),
            'ci95_high': float(np.mean(d) + tc * sem),
            'p_value': p_value(d),
            'win_rate': float(np.mean(d > 0)),
            'significant': bool(abs(np.mean(d)) > tc * sem),
        }
    return out


def density_table(results, buckets=(450, 500, 530, 560, 580, 601)):
    """Mean reward / open rate per density bucket for every strategy."""
    table = {}
    for name, rec in results.items():
        rows = rec['_episodes']
        per_bucket = []
        for lo, hi in zip(buckets[:-1], buckets[1:]):
            sel = [r for r in rows if lo <= r['n_cars'] < hi]
            if not sel:
                continue
            per_bucket.append({
                'range': f'{lo}-{hi - 1}',
                'n': len(sel),
                'reward': float(np.mean([r['reward'] for r in sel])),
                'open_rate': float(np.mean([r['open_rate'] for r in sel])),
                'social_speed': float(np.mean([r['social_speed'] for r in sel])),
                'emergency_speed': float(np.mean([r['emergency_speed']
                                                  for r in sel
                                                  if r['emergency_speed'] is not None]))
                if any(r['emergency_speed'] is not None for r in sel) else None,
                'lane2_frac': float(np.mean([r['lane2_frac'] for r in sel])),
            })
        table[name] = per_bucket
    return table


METRICS = ['reward', 'social_speed', 'emergency_speed', 'open_rate', 'lane2_frac']


def build(args):
    full = merge_glob(os.path.join(HERE, args.full_glob),
                      os.path.join(HERE, args.merged_name))
    screen = merge_glob(os.path.join(HERE, 'screen_shard_*.json'),
                        os.path.join(HERE, 'screen.json'))
    landscape = merge_glob(os.path.join(HERE, 'landscape_*.json'),
                           os.path.join(HERE, 'landscape.json'))
    if full is None:
        raise SystemExit(f'no {args.full_glob} found - run eval_paired.py full first')

    results = full['results']
    stats = {m: paired_vs_reference(results, m) for m in METRICS}
    dens = density_table(results)

    payload = {
        'reference': REFERENCE,
        'scipy_available': HAVE_SCIPY,
        'paired_stats': stats,
        'density_table': dens,
        'screen': screen,
        'landscape': landscape,
    }
    save(os.path.join(HERE, args.summary), payload)
    print(f'[analyze] scipy available: {HAVE_SCIPY} (t-table fallback otherwise)')
    print(f'[analyze] written: {args.summary}')

    # ---------------- console report ----------------
    print('\n' + '=' * 104)
    print(f'PAIRED COMPARISON vs {REFERENCE}   (positive = reference better)')
    print('=' * 104)
    print(f"{'策略':<18}{'指标':<16}{'配对差':>12}{'95% CI':>24}{'胜率':>9}{'显著':>7}")
    print('-' * 104)
    for m in METRICS:
        for name, s in stats[m].items():
            ci = f"[{s['ci95_low']:+.2f}, {s['ci95_high']:+.2f}]"
            print(f"{name:<18}{m:<16}{s['mean_diff']:>+12.3f}{ci:>24}"
                  f"{s['win_rate']:>9.1%}{'  YES' if s['significant'] else '   no':>7}")
        print('-' * 104)

    print('\n' + '=' * 104)
    print('REFERENCE (MADDPG argmax) BY DENSITY BUCKET')
    print('=' * 104)
    print(f"{'bucket':<12}{'n':>5}{'reward':>12}{'open_rate':>12}"
          f"{'social':>10}{'emergency':>11}{'lane2_frac':>12}")
    for row in dens[REFERENCE]:
        print(f"{row['range']:<12}{row['n']:>5}{row['reward']:>12.1f}"
              f"{row['open_rate']:>12.3f}{row['social_speed']:>10.3f}"
              f"{(row['emergency_speed'] if row['emergency_speed'] is not None else float('nan')):>11.3f}"
              f"{row['lane2_frac']:>12.4f}")
    print('=' * 104)
    return payload


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--build', action='store_true')
    p.add_argument('--full-glob', default='full_*.json')
    p.add_argument('--merged-name', default='full.json')
    p.add_argument('--summary', default='analysis_summary.json')
    a = p.parse_args()
    if not a.build:
        p.error('pass --build')
    build(a)
    sys.exit(0)
