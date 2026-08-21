"""Export the most recent within-breath trace as CSV.

Usage:
    python examples/respiratory_cycle_trace.py airway_obstruction
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys
import numpy as np

from openhumsim_rl import HumanHomeostasisEnv

scenario = sys.argv[1] if len(sys.argv) > 1 else "baseline"
env = HumanHomeostasisEnv(scenario=scenario)
env.reset(seed=42)
zero = np.zeros(8, dtype=np.float32)
for _ in range(6):
    _, _, terminated, truncated, info = env.step(zero)
    if terminated or truncated:
        break

trace = env.model.respiratory_cycle.last_trace
out = Path(f"respiratory_cycle_{scenario}.csv")
keys = list(trace)
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(keys)
    for row in zip(*(trace[k] for k in keys)):
        writer.writerow([float(x) for x in row])

s = info["state"]
print(out)
print(
    f"VT={s['tidal_volume_l']:.3f} L | "
    f"auto-PEEP={s['respiratory_cycle_auto_peep_cmH2O']:.2f} cmH2O | "
    f"Wmus={s['respiratory_cycle_muscle_work_j_breath']:.3f} J/breath | "
    f"AI={s['respiratory_ventilator_asynchrony_index_pct']:.1f}%"
)
