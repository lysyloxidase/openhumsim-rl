from __future__ import annotations
import numpy as np

from openhumsim_rl import HumanHomeostasisEnv


env = HumanHomeostasisEnv(scenario="dehydrated", render_mode="ansi")
_, info = env.reset(seed=42)
print("START")
print(env.render())

for _ in range(8):
    action = np.zeros(8, dtype=np.float32)
    action[6] = 0.70  # abstract oral-water intervention
    _, reward, terminated, truncated, info = env.step(action)
    print(env.render(), "reward=", round(reward, 3))
    if terminated or truncated:
        break
