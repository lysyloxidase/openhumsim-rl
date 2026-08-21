"""Very small hand-written controller for the toy PK target task.

This is not treatment logic. It only demonstrates that the RL task has a
closed action-observation-reward loop for the generic reference compound.
"""
from __future__ import annotations

import numpy as np

from openhumsim_rl import HumanHomeostasisEnv


env = HumanHomeostasisEnv(scenario="pk_target", render_mode="ansi")
_, info = env.reset(seed=3)

total_reward = 0.0
for _ in range(144):
    action = np.zeros(8, dtype=np.float32)
    effect = info["state"]["probe_effect_site_mg_l"]

    if effect < 0.65:
        action[7] = 0.35
    elif effect < 0.75:
        action[7] = 0.10

    _, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if int(info["time_min"]) % 60 == 0:
        print(env.render(), "effect=", round(info["state"]["probe_effect_site_mg_l"], 3))
    if terminated or truncated:
        break

print("total_reward:", round(total_reward, 3))
