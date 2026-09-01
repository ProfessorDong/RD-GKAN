#!/usr/bin/env python3
"""Merge the three T-scaling verification runs into one canonical results file.

The verification was split across three jobs for throughput; this consolidates
them into results/t_scaling_verify.json covering the full 3 kinetics x 6 T grid.
Refuses to write unless every cell is present.

Author: Liang Dong
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')
KINS = ['linear', 'hill', 'michaelis_menten']
TS = ['20', '50', '100', '200', '500', '1000']

merged = {'_meta': {
    'source': 'verbatim re-run of experiment_t_scaling() in run_synthetic_rd.py',
    'reason': 'the original synthetic_results.json was not on disk at camera-ready',
    'T_values': [int(t) for t in TS], 'N': 200, 'n_seeds': 5, 'G': 10,
    'kinetics': {'linear': 'f=-0.1c', 'hill': 'f=c^2/(1+c^2)-0.3c',
                 'michaelis_menten': 'f=c/(0.5+c)-0.2c'},
    'scripts': ['run_t_scaling_verify.py', 'run_t_scaling_hill.py',
                'run_t_scaling_rest.py', 'merge_t_scaling.py']}}

for f in ['t_scaling_verify.json', 't_scaling_verify_hill.json', 't_scaling_verify_rest.json']:
    path = os.path.join(R, f)
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    for k in KINS:
        if k in d:
            merged.setdefault(k, {}).update(d[k])

missing = [(k, t) for k in KINS for t in TS if t not in merged.get(k, {})]
if missing:
    print("INCOMPLETE, not writing. Missing cells:", missing)
    sys.exit(1)

for k in KINS:
    rates = [merged[k][t]['support_recovery_rate'] for t in TS]
    merged[k]['_summary'] = {
        'rates_by_T': dict(zip(TS, rates)),
        'min': min(rates), 'max': max(rates), 'mean': sum(rates) / len(rates)}

out = os.path.join(R, 't_scaling_verify.json')
json.dump(merged, open(out, 'w'), indent=2)
print("wrote", out)
for k in KINS:
    s = merged[k]['_summary']
    print(f"  {k:18s} rates={[f'{r:.0%}' for r in s['rates_by_T'].values()]} "
          f"range {s['min']:.0%}-{s['max']:.0%} mean {s['mean']:.1%}")
