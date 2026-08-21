from __future__ import annotations
import numpy as np

from openhumsim_rl import HumanHomeostasisEnv


SCENARIOS = [
    "baseline",
    "fasting",
    "oral_glucose_75g",
    "meal",
    "dehydrated",
    "hypoventilation",
    "respiratory_acidosis",
    "reduced_renal_function",
    "hyperkalemia",
    "transient_lactic_acidosis",
]

for scenario in SCENARIOS:
    env = HumanHomeostasisEnv(scenario=scenario)
    _, _ = env.reset(seed=1)
    info = None
    for _ in range(144):
        _, _, terminated, truncated, info = env.step(np.zeros(8, dtype=np.float32))
        if terminated or truncated:
            break
    s = info["state"]
    print(
        f"{scenario:22s} "
        f"G={s['glucose_mg_dl']:6.1f} "
        f"MAP={s['map_mmHg']:5.1f} "
        f"pH={s['ph_arterial']:.3f} "
        f"Na={s['sodium_mmol_l']:6.1f} "
        f"K={s['potassium_mmol_l']:4.2f} "
        f"GFR={s['gfr_ml_min']:6.1f} "
        f"ADH={s['adh_relative']:4.2f} "
        f"Renin={s['renin_relative']:4.2f}"
    )
