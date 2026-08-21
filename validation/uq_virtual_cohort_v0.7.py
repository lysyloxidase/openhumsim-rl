from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from dataclasses import replace

from openhumsim_rl import HumanHomeostasisEnv, sample_virtual_cohort
from historical_version_guard import require_exact_version

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "uq_virtual_cohort_v0.7.json"
REF = json.loads((ROOT / "validation" / "reference_targets_v0.7.json").read_text())


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(x), dtype=float)
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = ranks(np.asarray(x, dtype=float)), ranks(np.asarray(y, dtype=float))
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def qstats(x):
    a = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "sd": float(np.std(a)),
        "p05": float(np.quantile(a, 0.05)),
        "p50": float(np.quantile(a, 0.50)),
        "p95": float(np.quantile(a, 0.95)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def main(n=24, seed=20260817):
    require_exact_version("0.7.0")
    cohort = sample_virtual_cohort(n=n, seed=seed)
    rows = []
    for vp in cohort:
        baseline_cfg = replace(vp.config, cv_internal_step_s=0.02, cv_warmup_min=3.0)
        uq_cfg = replace(vp.config, cv_internal_step_s=0.04, cv_warmup_min=1.5)
        env = HumanHomeostasisEnv(config=baseline_cfg, scenario="baseline")
        _, info = env.reset(seed=123)  # common random numbers isolate parameter uncertainty
        s = info["state"]
        row = {"patient_id": vp.patient_id, **vp.latent}
        for key in [
            "cardiac_output_l_min", "cv_ejection_fraction", "map_mmHg",
            "pao2_mmHg", "paco2_mmHg", "ph_arterial", "gfr_ml_min",
            "arterial_o2_content_ml_dl", "oxygen_delivery_ml_min",
        ]:
            row[key] = float(s[key])

        # 75-g challenge: 90 min captures the nominal peak while keeping UQ tractable.
        meal = HumanHomeostasisEnv(config=uq_cfg, scenario="oral_glucose_75g")
        _, mi = meal.reset(seed=123)
        peak_g = mi["state"]["glucose_mg_dl"]
        peak_i = mi["state"]["insulin_uU_ml"]
        t_peak = 0.0
        for _ in range(18):
            _, _, term, trunc, mi = meal.step(np.zeros(8, dtype=np.float32))
            g = mi["state"]["glucose_mg_dl"]
            if g > peak_g:
                peak_g, t_peak = g, mi["time_min"]
            peak_i = max(peak_i, mi["state"]["insulin_uU_ml"])
            if term or trunc:
                break
        row["glucose_peak_75g_mg_dl"] = float(peak_g)
        row["glucose_tpeak_75g_min"] = float(t_peak)
        row["insulin_peak_75g_uU_ml"] = float(peak_i)
        rows.append(row)

    target_ranges = REF["targets"]
    coverage = {}
    for name, spec in target_ranges.items():
        vals = np.asarray([r[name] for r in rows], dtype=float)
        ok = (vals >= spec["low"]) & (vals <= spec["high"])
        coverage[name] = {
            "fraction_within_external_envelope": float(np.mean(ok)),
            "count": int(np.sum(ok)),
            "n": n,
            "range": [spec["low"], spec["high"]],
        }

    outputs = [
        "cardiac_output_l_min", "cv_ejection_fraction", "pao2_mmHg", "paco2_mmHg",
        "ph_arterial", "gfr_ml_min", "oxygen_delivery_ml_min",
        "glucose_peak_75g_mg_dl", "glucose_tpeak_75g_min", "insulin_peak_75g_uU_ml",
    ]
    latent_names = list(cohort[0].latent)
    sensitivity = {}
    for out in outputs:
        y = np.asarray([r[out] for r in rows], dtype=float)
        corr = []
        for p in latent_names:
            x = np.asarray([r[p] for r in rows], dtype=float)
            corr.append((p, spearman(x, y)))
        corr.sort(key=lambda z: abs(z[1]), reverse=True)
        sensitivity[out] = [{"parameter": p, "spearman_rho": rho} for p, rho in corr[:5]]

    result = {
        "version": "0.7.0",
        "design": {
            "method": "Latin hypercube",
            "n": n,
            "seed": seed,
            "common_reset_seed": 123,
            "baseline_cv_internal_step_s": 0.02,
            "baseline_cv_warmup_min": 3.0,
            "challenge_cv_internal_step_s": 0.04,
            "challenge_cv_warmup_min": 1.5,
            "interpretation": "Parameter ranges are robustness/UQ intervals, not fitted population distributions."
        },
        "baseline_summary": {k: qstats([r[k] for r in rows]) for k in outputs[:7]},
        "challenge_75g_summary": {k: qstats([r[k] for r in rows]) for k in outputs[7:]},
        "external_envelope_coverage": coverage,
        "rank_sensitivity_top5": sensitivity,
        "patients": rows,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({"coverage": coverage, "75g": result["challenge_75g_summary"]}, indent=2))


if __name__ == "__main__":
    main()
