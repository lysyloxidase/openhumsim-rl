from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
import numpy as np

from openhumsim_rl import HumanHomeostasisEnv
from openhumsim_rl.population import ParameterSpec, latin_hypercube, virtual_patient_from_unit_row
from historical_version_guard import require_exact_version

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "ood_ppo_benchmark_v0.8.json"
MODEL = ROOT / "validation" / "ppo_respiratory_v0.7.pt"

OOD_SPECS = (
    ParameterSpec("body_weight_kg", 48.0, 105.0),
    ParameterSpec("blood_volume_ml_per_kg", 60.0, 85.0),
    ParameterSpec("tbw_fraction", 0.46, 0.68),
    ParameterSpec("ecf_fraction", 0.16, 0.25),
    ParameterSpec("hemoglobin_g_dl", 11.5, 17.5),
    ParameterSpec("baseline_gfr_ml_min", 75.0, 140.0),
    ParameterSpec("cv_resting_hr_bpm", 50.0, 95.0),
    ParameterSpec("cv_r_systemic_scale", 0.75, 1.30),
    ParameterSpec("cv_lv_emax_scale", 0.75, 1.25),
    ParameterSpec("baseline_aa_gradient_mmHg", 4.0, 20.0),
    ParameterSpec("dalla_insulin_sensitivity_scale", 0.65, 1.35),
    ParameterSpec("dalla_gastric_absorption_scale", 0.75, 1.25),
    ParameterSpec("respiratory_acidosis_efficiency", 0.45, 0.80),
    ParameterSpec("pbpk_fraction_unbound", 0.15, 0.85),
    ParameterSpec("pbpk_hepatic_clint_l_min", 0.07, 0.35),
)


def action(o2=0.0, vent=0.0):
    x = np.zeros(8, dtype=np.float32)
    x[4] = o2
    x[5] = vent
    return x

ACTIONS = [action(), action(0.0, 0.06), action(0.0, 0.12), action(0.08, 0.10), action(0.15, 0.0)]


def cohort(n=8, seed=8802):
    design = latin_hypercube(n, len(OOD_SPECS), seed=seed)
    return [virtual_patient_from_unit_row(row, f"OOD-{i:03d}", specs=OOD_SPECS) for i, row in enumerate(design)]


def env_for(vp, seed):
    cfg = replace(vp.config, episode_minutes=30.0, cv_internal_step_s=0.04)
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


def evaluate(fn, c, seeds):
    returns, terminations = [], []
    for vp, seed in zip(c, seeds):
        env, obs, info = env_for(vp, seed)
        total = 0.0
        while True:
            ai = int(fn(obs, info))
            obs, r, term, trunc, info = env.step(ACTIONS[ai])
            total += float(r)
            if term or trunc:
                break
        returns.append(total)
        terminations.append(info.get("termination_reason"))
    return {
        "returns": returns,
        "mean_return": float(np.mean(returns)),
        "sd_return": float(np.std(returns)),
        "n_terminated": int(sum(x is not None for x in terminations)),
        "termination_reasons": terminations,
    }


def main():
    require_exact_version("0.8.0")
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the OOD PPO benchmark") from exc

    torch.set_num_threads(1)
    torch.manual_seed(20260817)
    random.seed(20260817)
    np.random.seed(20260817)
    c = cohort()
    env, obs, _ = env_for(c[0], 1)
    obs_dim = int(obs.shape[0])

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
            self.pi = nn.Linear(64, len(ACTIONS))
            self.v = nn.Linear(64, 1)
        def forward(self, x):
            h = self.body(x)
            return self.pi(h), self.v(h).squeeze(-1)

    net = ActorCritic()
    net.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True))
    net.eval()

    def ppo(obs, info):
        with torch.no_grad():
            logits, _ = net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(logits, dim=-1).item())

    seeds = list(range(8800, 8800 + len(c)))
    result = {
        "version": "0.8.0",
        "task": "v0.7 respiratory PPO evaluated outside its training virtual-patient parameter box",
        "n_ood_patients": len(c),
        "ood_specs": [s.__dict__ for s in OOD_SPECS],
        "no_op": evaluate(lambda obs, info: 0, c, seeds),
        "heuristic": evaluate(lambda obs, info: heuristic(info), c, seeds),
        "ppo_v0.7": evaluate(ppo, c, seeds),
        "interpretation": (
            "OOD evaluation is deliberately harder than the v0.7 held-out in-distribution benchmark. "
            "A learned policy failing to dominate the heuristic is evidence of limited policy robustness, not a software failure."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k]["mean_return"] for k in ("no_op", "heuristic", "ppo_v0.7")}, indent=2))


if __name__ == "__main__":
    main()
