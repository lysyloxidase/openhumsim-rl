from __future__ import annotations

from historical_version_guard import require_exact_version
require_exact_version("0.9.0")

import json
from pathlib import Path
import numpy as np

from openhumsim_rl import __version__
from openhumsim_rl.cgm import CGMObservationConfig, blood_to_cgm_trace
from openhumsim_rl.cgm_reference import calibrate_normative_cgm_reference
from openhumsim_rl.external_data import (
    JAEB_HEALTHY_CGM_URL,
    REFERENCE,
    SHAH_2019_AGE_STRATA,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "validation_results_v0.9.json"


def check(name, value, passed, tier, note=""):
    return {"name": name, "value": value, "pass": bool(passed), "tier": tier, "note": note}


def synthetic_subject_summary(n=30):
    metrics = []
    for i in range(n):
        mean = 96.0 + 0.28 * i
        cv = 14.0 + (i % 7) * 0.8
        tir = np.clip(98.0 - 0.25 * i, 82.0, 99.5)
        metrics.append({
            "subject": f"S{i:03d}",
            "n": 1000,
            "mean_mg_dl": float(mean),
            "cv_pct": float(cv),
            "time_70_140_pct": float(tir),
            "time_gt_140_pct": float(max(0.0, 100.0 - tir - 1.0)),
            "time_lt_70_pct": 1.0,
        })
    return {"subject_metrics": metrics}


def main():
    checks = []

    legacy = json.loads((ROOT / "validation/validation_results_v0.8.json").read_text())
    checks.append(check(
        "v0.8 scientific regression suite",
        legacy["summary"],
        legacy["summary"]["failed"] == 0,
        "regression",
    ))

    checks.append(check("release version", __version__, __version__ == "0.9.0", "software-verification"))

    # Observation-model verification: after one time constant, a first-order step
    # must cover 1-e^-1 of the blood-to-interstitial difference.
    blood = np.r_[np.full(2, 90.0), np.full(12, 160.0)]
    cgm = blood_to_cgm_trace(blood, dt_min=1.0, config=CGMObservationConfig(lag_tau_min=6.0))
    # index 7 = six 1-min updates after the step at index 2
    expected = 90.0 + (160.0 - 90.0) * (1.0 - np.exp(-1.0))
    lag_error = abs(float(cgm[7]) - expected)
    checks.append(check(
        "CGM first-order lag equation",
        {"observed_after_6min": float(cgm[7]), "expected": float(expected), "abs_error": lag_error},
        lag_error < 1e-10,
        "calculation-verification",
        "Default tau=6 min is an observation-model choice centered on published 5.3-6.2 min physiological lag estimates.",
    ))

    strata_n = sum(x.n for x in SHAH_2019_AGE_STRATA)
    older = [x for x in SHAH_2019_AGE_STRATA if x.age_group == ">=60"][0]
    ref_ok = (
        strata_n == 153
        and REFERENCE.n_participants == 153
        and REFERENCE.median_time_70_140_pct == 96.0
        and REFERENCE.mean_within_person_cv_pct == 17.0
        and older.mean_glucose_mg_dl == 104.0
        and older.time_70_140_median_pct == 93.0
    )
    checks.append(check(
        "Shah-2019 aggregate and age-stratified reference fidelity",
        {
            "n": strata_n,
            "overall_mean_typical_age_groups": REFERENCE.mean_average_glucose_mg_dl_typical_age_groups,
            "overall_TIR_70_140_median": REFERENCE.median_time_70_140_pct,
            "overall_CV_mean": REFERENCE.mean_within_person_cv_pct,
            "age_60plus_mean": older.mean_glucose_mg_dl,
            "age_60plus_TIR": older.time_70_140_median_pct,
        },
        ref_ok,
        "external-human-reference",
    ))

    # Split/calibration mechanics are executed on a synthetic fixture because this
    # release environment cannot retrieve the optional Jaeb archive from S3.
    cal = calibrate_normative_cgm_reference(synthetic_subject_summary(), seed=2019, n_boot=100)
    payload = cal.as_dict()
    leak = payload["split_report"]["leakage_check"]
    finite_eval = all(
        np.isfinite(v["mean_log_likelihood"])
        for part in payload["evaluation"].values()
        for v in part.values()
    )
    checks.append(check(
        "participant-level split prevents leakage",
        leak,
        leak["passed"] and leak["all_subjects_accounted_for"],
        "data-science-verification",
    ))
    checks.append(check(
        "TRAIN-only normative CGM calibration pipeline",
        {
            "train_n": payload["fitted_on_train"]["mean_mg_dl"]["train_n"],
            "validation_n": payload["evaluation"]["validation"]["mean_mg_dl"]["n"],
            "test_n": payload["evaluation"]["test"]["mean_mg_dl"]["n"],
        },
        finite_eval and payload["evaluation"]["test"]["mean_mg_dl"]["n"] > 0,
        "algorithm-verification",
        "Mechanics verified on synthetic subjects; participant-level Jaeb execution must be run after opt-in download.",
    ))

    raw_archives = list((ROOT / "data" / "external").glob("*.zip")) if (ROOT / "data" / "external").exists() else []
    checks.append(check(
        "individual-level Jaeb data not bundled",
        [p.name for p in raw_archives],
        len(raw_archives) == 0,
        "data-governance",
        "Public human data remain an explicit local opt-in download.",
    ))

    checks.append(check(
        "official Jaeb download target configured",
        JAEB_HEALTHY_CGM_URL,
        "live-jchrpublicdatasets.s3.amazonaws.com" in JAEB_HEALTHY_CGM_URL and "CGMND" in JAEB_HEALTHY_CGM_URL,
        "external-data-plumbing",
    ))

    result = {
        "version": "0.9.0",
        "summary": {
            "passed": sum(c["pass"] for c in checks),
            "failed": sum(not c["pass"] for c in checks),
            "total": len(checks),
        },
        "checks": checks,
        "external_reference": {
            "overall": REFERENCE.as_dict(),
            "age_strata": [x.as_dict() for x in SHAH_2019_AGE_STRATA],
            "jaeb_download_url": JAEB_HEALTHY_CGM_URL,
            "paper": "Shah VN et al. J Clin Endocrinol Metab. 2019;104(10):4356-4364. PMID 31127824; DOI 10.1210/jc.2018-02763",
            "dataset": "Dryad DOI 10.5061/dryad.h7d11cd; raw Jaeb CGM archive listed by Jaeb Center.",
        },
        "credibility_classification": {
            "software_verification": "strong for tested paths",
            "cgm_observation_model": "mechanistically plausible reduced-order first-order lag; not device-specific calibration",
            "external_human_reference": "real published healthy-nondiabetic CGM aggregates and age strata encoded",
            "participant_split_and_reference_model": "implemented and leakage-tested; raw Jaeb execution pending local data download",
            "mechanistic_patient_calibration": "not yet achieved because subject-matched meal/exercise/sleep inputs have not been reconstructed",
            "clinical_decision_use": "not supported",
        },
        "remaining_requirement": (
            "Run the participant-level Jaeb archive locally, inspect available meal/exercise/sleep logs, "
            "and only then construct protocol-matched likelihoods for Dalla Man / virtual-patient parameters."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    for c in checks:
        print(("PASS" if c["pass"] else "FAIL"), c["tier"], c["name"])


if __name__ == "__main__":
    main()
