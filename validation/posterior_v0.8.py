from __future__ import annotations

import json
from pathlib import Path
from openhumsim_rl import GaussianTarget, importance_calibrate
from historical_version_guard import require_exact_version

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "posterior_v0.8.json"

TARGETS = [
    GaussianTarget("cardiac_output_l_min", 5.0, measurement_sd=0.50, model_discrepancy_sd=0.50),
    GaussianTarget("map_mmHg", 90.0, measurement_sd=7.0, model_discrepancy_sd=5.0),
    GaussianTarget("gfr_ml_min", 110.0, measurement_sd=15.0, model_discrepancy_sd=10.0),
    GaussianTarget("pao2_mmHg", 92.0, measurement_sd=7.0, model_discrepancy_sd=7.0),
]


def main():
    require_exact_version("0.8.0")
    result = importance_calibrate(TARGETS, n_prior=48, seed=8080, cv_internal_step_s=0.04)
    payload = {
        "version": "0.8.0",
        "method": "bounded LHS prior + Gaussian likelihood importance weighting",
        "result": result.as_dict(),
        "interpretation": (
            "Literature-centered posterior with explicit model-discrepancy variance. "
            "This is a calibration/UQ mechanics benchmark, not patient-specific Bayesian inference "
            "and not a substitute for protocol-matched individual-level clinical data."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "n_prior": result.n_prior,
        "effective_sample_size": result.effective_sample_size,
        "posterior_outputs": result.posterior_output_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
