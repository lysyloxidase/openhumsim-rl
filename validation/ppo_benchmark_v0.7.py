from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
import numpy as np

from openhumsim_rl import HumanHomeostasisEnv, sample_virtual_cohort
from historical_version_guard import require_exact_version

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "ppo_benchmark_v0.7.json"
MODEL_OUT = ROOT / "validation" / "ppo_respiratory_v0.7.pt"


def action(o2=0.0, vent=0.0):
    x = np.zeros(8, dtype=np.float32)
    x[4] = o2
    x[5] = vent
    return x

ACTIONS = [
    action(),
    action(0.0, 0.06),
    action(0.0, 0.12),
    action(0.08, 0.10),
    action(0.15, 0.0),
]


def env_for(vp, seed, accurate=False):
    dt = 0.02 if accurate else 0.04
    cfg = replace(vp.config, episode_minutes=30.0, cv_internal_step_s=dt)
    env = HumanHomeostasisEnv(config=cfg, scenario="respiratory_acidosis")
    obs, info = env.reset(seed=seed)
    return env, obs, info


def heuristic(info):
    s = info["state"]
    if s["spo2_pct"] < 92:
        return 3
    if s["paco2_mmHg"] > 52:
        return 2
    if s["paco2_mmHg"] > 45:
        return 1
    return 0


def evaluate_policy(policy_fn, cohort, seeds, accurate=True):
    returns, finals, interventions = [], [], []
    for vp, seed in zip(cohort, seeds):
        env, obs, info = env_for(vp, seed, accurate=accurate)
        total = 0.0
        used = 0
        while True:
            ai = int(policy_fn(obs, info))
            if ai != 0:
                used += 1
            obs, r, term, trunc, info = env.step(ACTIONS[ai])
            total += float(r)
            if term or trunc:
                break
        s = info["state"]
        returns.append(total)
        interventions.append(used)
        finals.append({
            "paco2": float(s["paco2_mmHg"]),
            "pH": float(s["ph_arterial"]),
            "spo2": float(s["spo2_pct"]),
            "ventilation_efficiency": float(s["ventilation_efficiency"]),
        })
    return {
        "returns": returns,
        "mean_return": float(np.mean(returns)),
        "sd_return": float(np.std(returns)),
        "mean_nonzero_actions": float(np.mean(interventions)),
        "finals": finals,
    }


def main():
    require_exact_version("0.7.0")
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.distributions import Categorical
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the v0.7 PPO benchmark") from exc

    torch.manual_seed(20260817)
    np.random.seed(20260817)
    random.seed(20260817)
    torch.set_num_threads(1)

    train_cohort = sample_virtual_cohort(40, seed=7001)
    test_cohort = sample_virtual_cohort(12, seed=7002)

    probe_env, probe_obs, _ = env_for(train_cohort[0], 1, accurate=False)
    obs_dim = int(probe_obs.shape[0])
    n_actions = len(ACTIONS)

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(obs_dim, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
            )
            self.pi = nn.Linear(64, n_actions)
            self.v = nn.Linear(64, 1)

        def forward(self, x):
            h = self.body(x)
            return self.pi(h), self.v(h).squeeze(-1)

    net = ActorCritic()
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    gamma, lam = 0.96, 0.95
    clip_eps, entropy_coef, value_coef = 0.20, 0.01, 0.5
    train_returns = []
    batch = []

    def finish_trajectory(traj):
        rewards = np.asarray([x[4] for x in traj], dtype=np.float32)
        values = np.asarray([x[3] for x in traj] + [0.0], dtype=np.float32)
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            adv[t] = gae
        ret = adv + values[:-1]
        out = []
        for item, a, r in zip(traj, adv, ret):
            obs, ai, old_logp, val, rew = item
            out.append((obs, ai, old_logp, float(a), float(r)))
        return out

    updates = 0
    for ep in range(40):
        vp = train_cohort[ep % len(train_cohort)]
        env, obs, info = env_for(vp, 1000 + ep, accurate=False)
        traj = []
        total = 0.0
        while True:
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, value = net(x)
                dist = Categorical(logits=logits)
                ai = dist.sample()
                logp = dist.log_prob(ai)
            next_obs, reward, term, trunc, info = env.step(ACTIONS[int(ai.item())])
            traj.append((obs.copy(), int(ai.item()), float(logp.item()), float(value.item()), float(reward)))
            total += float(reward)
            obs = next_obs
            if term or trunc:
                break
        train_returns.append(total)
        batch.extend(finish_trajectory(traj))

        if (ep + 1) % 5 == 0:
            obs_b = torch.tensor(np.asarray([x[0] for x in batch]), dtype=torch.float32)
            act_b = torch.tensor([x[1] for x in batch], dtype=torch.long)
            oldlog_b = torch.tensor([x[2] for x in batch], dtype=torch.float32)
            adv_b = torch.tensor([x[3] for x in batch], dtype=torch.float32)
            ret_b = torch.tensor([x[4] for x in batch], dtype=torch.float32)
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

            for _ in range(6):
                logits, values = net(obs_b)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act_b)
                ratio = torch.exp(logp - oldlog_b)
                s1 = ratio * adv_b
                s2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = ((values - ret_b) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.8)
                optimizer.step()
            batch = []
            updates += 1

    torch.save(net.state_dict(), MODEL_OUT)

    def ppo_policy(obs, info):
        with torch.no_grad():
            logits, _ = net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(logits, dim=-1).item())

    noop = lambda obs, info: 0
    heur = lambda obs, info: heuristic(info)
    rng = np.random.default_rng(1234)
    random_policy = lambda obs, info: int(rng.integers(n_actions))
    seeds = list(range(3000, 3012))

    result = {
        "version": "0.7.0",
        "task": "30-min respiratory-acidosis control; neural PPO over five discrete O2/ventilation actions",
        "training": {
            "episodes": 40,
            "updates": updates,
            "virtual_patients": 40,
            "training_cv_dt_s": 0.04,
            "final_10_episode_mean_return": float(np.mean(train_returns[-10:])),
            "framework": "direct PyTorch PPO; no Gymnasium/SB3 dependency",
        },
        "evaluation": {
            "virtual_patients": 12,
            "held_out_lhs_seed": 7002,
            "evaluation_cv_dt_s": 0.02,
            "no_op": evaluate_policy(noop, test_cohort, seeds),
            "random": evaluate_policy(random_policy, test_cohort, seeds),
            "heuristic": evaluate_policy(heur, test_cohort, seeds),
            "ppo": evaluate_policy(ppo_policy, test_cohort, seeds),
        },
        "interpretation": (
            "This benchmark tests learning and parameter-robust generalization in a simplified task. "
            "It is not evidence that the learned policy is clinically safe or optimal."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v["mean_return"] for k, v in result["evaluation"].items() if isinstance(v, dict) and "mean_return" in v}, indent=2))


if __name__ == "__main__":
    main()
