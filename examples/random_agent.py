from __future__ import annotations

from openhumsim_rl import HumanHomeostasisEnv


env = HumanHomeostasisEnv(scenario="oral_glucose_75g", render_mode="ansi")
obs, info = env.reset(seed=42)

total_reward = 0.0
for _ in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if int(info["time_min"]) % 60 == 0:
        print(env.render())

    if terminated or truncated:
        break

print("total_reward:", round(total_reward, 3))
print("final_state:", info["state"])
