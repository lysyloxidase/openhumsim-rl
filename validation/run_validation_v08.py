from __future__ import annotations

from historical_version_guard import require_exact_version
require_exact_version("0.8.0")

import json
from pathlib import Path
import numpy as np

from openhumsim_rl import __version__
from openhumsim_rl.cli import doctor, run_demo
from openhumsim_rl.external_data import REFERENCE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "validation_results_v0.8.json"


def check(name, value, passed, tier, note=""):
    return {"name": name, "value": value, "pass": bool(passed), "tier": tier, "note": note}


def main():
    checks = []

    legacy = json.loads((ROOT / "validation/validation_results_v0.7.json").read_text())
    checks.append(check(
        "v0.7 scientific regression suite",
        legacy["summary"],
        legacy["summary"]["failed"] == 0,
        "regression",
    ))

    checks.append(check("release version", __version__, __version__ == "0.8.0", "software-verification"))

    d = doctor()
    checks.append(check(
        "local CLI preflight",
        {"core_ready": d["core_ready"], "python": d["python"], "machine": d["machine"]},
        d["core_ready"],
        "local-reproducibility",
    ))

    demo = run_demo("baseline", minutes=10.0, seed=42)
    s = demo["state"]
    demo_ok = (
        not demo["terminated"]
        and np.isfinite(demo["return"])
        and 70 < s["glucose_mg_dl"] < 130
        and 60 < s["map_mmHg"] < 130
        and 7.25 < s["ph_arterial"] < 7.55
    )
    checks.append(check(
        "local CLI 10-min simulation",
        {"return": demo["return"], "glucose": s["glucose_mg_dl"], "MAP": s["map_mmHg"], "pH": s["ph_arterial"]},
        demo_ok,
        "local-reproducibility",
    ))

    ext = json.loads((ROOT / "validation/external_reference_shah2019_v0.8.json").read_text())
    pub = ext["published_aggregate_metrics"]
    checks.append(check(
        "independent published healthy-CGM reference encoded",
        {
            "n": pub["n_participants"],
            "median_TIR_70_140_pct": pub["median_time_70_140_pct"],
            "mean_CV_pct": pub["mean_within_person_cv_pct"],
        },
        pub["n_participants"] == 153
        and pub["median_time_70_140_pct"] == 96.0
        and pub["mean_within_person_cv_pct"] == 17.0,
        "external-data-reference",
        "The raw public Jaeb archive is optional and not bundled; the paper-level aggregate metrics are external human data.",
    ))

    posterior = json.loads((ROOT / "validation/posterior_v0.8.json").read_text())["result"]
    ess = posterior["effective_sample_size"]
    post = posterior["posterior_output_summary"]
    post_ok = (
        2.0 < ess <= posterior["n_prior"]
        and 4.0 < post["cardiac_output_l_min"]["mean"] < 6.5
        and 80 < post["map_mmHg"]["mean"] < 110
        and 85 < post["gfr_ml_min"]["mean"] < 135
        and 75 < post["pao2_mmHg"]["mean"] < 110
    )
    checks.append(check(
        "likelihood-weighted posterior virtual patients",
        {"ESS": ess, "outputs": post},
        post_ok,
        "bayesian-uq-smoke",
        "Uses bounded priors, Gaussian target likelihoods and explicit model-discrepancy variance; not clinical patient inference.",
    ))

    ood = json.loads((ROOT / "validation/ood_ppo_benchmark_v0.8.json").read_text())
    ppo = ood["ppo_v0.7"]["mean_return"]
    noop = ood["no_op"]["mean_return"]
    heur = ood["heuristic"]["mean_return"]
    checks.append(check(
        "OOD PPO remains finite and improves over no-op",
        {"PPO": ppo, "no_op": noop, "heuristic": heur},
        np.isfinite(ppo) and np.isfinite(noop) and np.isfinite(heur) and ppo > noop,
        "rl-ood-robustness",
        "PPO underperforms the heuristic OOD; this is retained as a limitation rather than tuned away.",
    ))

    local_files = [
        ROOT / ".vscode/settings.json",
        ROOT / ".vscode/tasks.json",
        ROOT / ".vscode/launch.json",
        ROOT / "scripts/bootstrap_macos.sh",
        ROOT / "LOCAL_VSCODE.md",
    ]
    checks.append(check(
        "VSCode/local bootstrap assets",
        [str(p.relative_to(ROOT)) for p in local_files],
        all(p.exists() for p in local_files),
        "local-reproducibility",
    ))

    raw_human_files = list((ROOT / "data/external").glob("*.zip"))
    checks.append(check(
        "no individual-level external human data bundled",
        [p.name for p in raw_human_files],
        len(raw_human_files) == 0,
        "data-governance",
        "External human data are opt-in downloads; only source metadata/reference summaries ship with the release.",
    ))

    result = {
        "version": "0.8.0",
        "summary": {
            "passed": sum(c["pass"] for c in checks),
            "failed": sum(not c["pass"] for c in checks),
            "total": len(checks),
        },
        "checks": checks,
        "research_findings": {
            "ood_policy": {
                "ppo_mean_return": ppo,
                "heuristic_mean_return": heur,
                "delta_ppo_vs_heuristic": ppo - heur,
                "interpretation": "The v0.7 PPO generalizes above no-op but not above the hand-coded heuristic under the expanded OOD parameter box.",
            },
            "posterior": {
                "effective_sample_size": ess,
                "fraction_of_prior": ess / posterior["n_prior"],
                "interpretation": "The literature-centered likelihood concentrates the bounded engineering prior, but the result is not a measured population posterior.",
            },
            "external_cgm": {
                "reference": REFERENCE.as_dict(),
                "interpretation": "Published independent human CGM aggregates are now encoded; raw individual-level comparison is available locally after opt-in download.",
            },
        },
        "credibility_classification": {
            "software_verification": "strong for tested paths",
            "calculation_verification": "strong reduced-order invariants/timestep regression inherited from v0.6-v0.7",
            "parameter_uncertainty": "bounded prior + LHS + likelihood-weighted posterior mechanics implemented",
            "external_human_data": "first independent aggregate CGM reference integrated; raw public dataset adapter added",
            "model_form_uncertainty": "represented only as target-level discrepancy SD in posterior; not yet a dynamic discrepancy process",
            "rl_generalization": "in-distribution positive; OOD limited and below heuristic",
            "patient_specific_validation": "absent",
            "clinical_decision_use": "not supported",
        },
        "known_missing_or_unvalidated": [
            "protocol-matched individual-level clinical calibration cohort",
            "raw Jaeb CGM comparison has to be run locally because the release environment cannot fetch the external archive",
            "dynamic model-form discrepancy process",
            "full electroneutrality and total inorganic carbon conservation",
            "distributed V/Q, shunt and diffusion limitation",
            "real-drug PBPK calibration",
            "subcutaneous insulin PK",
            "creatinine kinetics / KDIGO AKI implementation",
            "large multi-seed PPO/SAC benchmark with confidence intervals and safety constraints",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    for c in checks:
        print(("PASS" if c["pass"] else "FAIL"), c["tier"], c["name"])


if __name__ == "__main__":
    main()
