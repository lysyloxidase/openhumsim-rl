from __future__ import annotations
import numpy as np

from openhumsim_rl import HumanHomeostasisEnv


env = HumanHomeostasisEnv(
    scenario="respiratory_acidosis",
    render_mode="ansi",
)
_, info = env.reset(seed=42)

print("START")
print(env.render())

for _ in range(6):
    action = np.zeros(8, dtype=np.float32)

    # Toy intervention: supplemental O2 + ventilatory support.
    action[4] = 0.20
    action[5] = 0.12

    _, reward, terminated, truncated, info = env.step(action)
    print(env.render(), "reward=", round(reward, 3))

    if terminated or truncated:
        break
