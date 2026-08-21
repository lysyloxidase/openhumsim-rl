from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from openhumsim_rl import HumanConfig
from openhumsim_rl.calibration import CalibrationTarget, fit_reference_profile, reference_outputs
from historical_version_guard import require_exact_version

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "calibration_v0.7.json"


def main():
    require_exact_version("0.7.0")
    base = HumanConfig()
    # Literature-centered nominal profile. These targets are deliberately low-dimensional;
    # they are not a matched subject dataset and therefore do not constitute clinical fitting.
    targets = [
        CalibrationTarget("cardiac_output_l_min", 5.0, 0.8),
        CalibrationTarget("map_mmHg", 90.0, 10.0),
        CalibrationTarget("pao2_mmHg", 92.0, 8.0),
        CalibrationTarget("gfr_ml_min", 110.0, 15.0),
    ]
    bounds = {
        "cv_r_systemic_mmHg_s_ml": (0.85, 1.45),
        "cv_lv_emax": (1.8, 2.8),
        "baseline_aa_gradient_mmHg": (4.0, 18.0),
        "baseline_gfr_ml_min": (90.0, 130.0),
    }
    fit = fit_reference_profile(base, targets, bounds, seed=123)

    # Synthetic parameter-recovery check: verifies inverse solver mechanics separately
    # from external biological validation.
    true_cfg = replace(
        base,
        cv_r_systemic_mmHg_s_ml=1.24,
        cv_lv_emax=2.48,
        baseline_aa_gradient_mmHg=11.5,
        baseline_gfr_ml_min=104.0,
    )
    truth = reference_outputs(true_cfg, seed=321)
    synthetic_targets = [
        CalibrationTarget("cardiac_output_l_min", truth["cardiac_output_l_min"], 0.5),
        CalibrationTarget("map_mmHg", truth["map_mmHg"], 5.0),
        CalibrationTarget("pao2_mmHg", truth["pao2_mmHg"], 5.0),
        CalibrationTarget("gfr_ml_min", truth["gfr_ml_min"], 8.0),
    ]
    recovered = fit_reference_profile(base, synthetic_targets, bounds, seed=321)
    true_params = {k: float(getattr(true_cfg, k)) for k in bounds}
    rel_errors = {
        k: abs(recovered.fitted[k] - true_params[k]) / max(abs(true_params[k]), 1e-12)
        for k in bounds
    }

    result = {
        "version": "0.7.0",
        "reference_profile_fit": fit.__dict__,
        "synthetic_parameter_recovery": {
            "true_parameters": true_params,
            "recovered_parameters": recovered.fitted,
            "relative_errors": rel_errors,
            "max_relative_error": max(rel_errors.values()),
            "normalized_rmse": recovered.normalized_rmse,
            "success": recovered.success,
        },
        "interpretation": (
            "Reference-profile fitting demonstrates a calibration mechanism only. Synthetic recovery "
            "tests algorithmic identifiability under the selected observables. Neither is patient-level clinical validation."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
