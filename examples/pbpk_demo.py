from __future__ import annotations

import numpy as np

from openhumsim_rl import HumanHomeostasisEnv


def run(scenario: str):
    env = HumanHomeostasisEnv(scenario=scenario)
    _, info = env.reset(seed=17)

    # Give a one-off 100 mg dose using four 25 mg model actions.
    for i in range(72):  # 6 h
        action = np.zeros(8, dtype=np.float32)
        if i < 4:
            action[7] = 1.0
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return info


for scenario in ("baseline", "reduced_renal_function"):
    info = run(scenario)
    s = info["state"]
    pk = info["pbpk"]
    print(
        f"{scenario:24s} "
        f"GFR={s['gfr_ml_min']:6.1f} mL/min | "
        f"Cplasma={pk['plasma_concentration_mg_l']:.4f} mg/L | "
        f"CLrenal={pk['renal_clearance_l_min']:.4f} L/min | "
        f"elim_renal={pk['renal_eliminated_mg']:.3f} mg | "
        f"mass_error={pk['mass_balance_error_mg']:.3e} mg"
    )
