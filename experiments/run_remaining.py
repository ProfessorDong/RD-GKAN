#!/usr/bin/env python3
"""Run remaining experiments: ablations, stability, QS."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_synthetic_rd import experiment_ablations, experiment_stability, experiment_qs_illustration, RESULTS_DIR

results = {}
results['ablations'] = experiment_ablations()
results['stability'] = experiment_stability()
results['qs'] = experiment_qs_illustration()

out = os.path.join(RESULTS_DIR, 'remaining_results.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {out}")
