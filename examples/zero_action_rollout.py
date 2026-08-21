from __future__ import annotations
import numpy as np

from openhumsim_rl import HumanHomeostasisEnv


env = HumanHomeostasisEnv(scenario="oral_glucose_75g", render_mode="ansi")
obs, info = env.reset(seed=7)

for _ in range(300):
    action = np.zeros(8, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    if int(info["time_min"]) % 30 == 0:
        print(env.render())

    if terminated or truncated:
        break
