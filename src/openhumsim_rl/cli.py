from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np

from . import __version__
from .env import HumanHomeostasisEnv, ACTION_NAMES
from .cgm import CGMObservationConfig, blood_to_cgm_trace
from .cgm_reference import calibrate_normative_cgm_reference
from .external_data import (
    JAEB_HEALTHY_CGM_URL,
    JAEB_DATASET_PAGE_URL,
    REFERENCE,
    download_jaeb_healthy_cgm,
    jaeb_download_instructions,
    list_archive_csvs,
    summarize_cgm_archive,
    save_summary,
    shah_2019_age_strata_reference,
    build_subject_split_report,
)
from .population import sample_virtual_cohort, DEFAULT_PARAMETER_SPECS
from .event_replay import (
    DUBOSE_REFERENCE,
    archive_schema_report,
    extract_events_from_archive,
    event_metrics_from_archive,
    summarize_event_metrics,
    calibrate_aggregate_event_profile,
    fit_mechanistic_event_profile,
    save_json as save_event_json,
)




def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _dep(name: str) -> dict:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"installed": False, "version": None}
    module = __import__(name)
    return {"installed": True, "version": getattr(module, "__version__", "unknown")}


def doctor() -> dict:
    result = {
        "openhumsim_version": __version__,
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": {name: _dep(name) for name in ("numpy", "scipy", "torch", "gymnasium", "pytest")},
        "torch_mps_available": False,
    }
    if result["dependencies"]["torch"]["installed"]:
        try:
            import torch
            result["torch_mps_available"] = bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )
        except Exception:
            pass
    core_ok = sys.version_info >= (3, 10) and result["dependencies"]["numpy"]["installed"]
    result["core_ready"] = bool(core_ok)
    result["full_validation_ready"] = bool(
        core_ok
        and result["dependencies"]["scipy"]["installed"]
        and result["dependencies"]["torch"]["installed"]
        and result["dependencies"]["pytest"]["installed"]
    )
    return result


def run_demo(scenario: str, minutes: float, seed: int) -> dict:
    env = HumanHomeostasisEnv(scenario=scenario)
    _, info = env.reset(seed=seed)
    zero = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    total = 0.0
    steps = max(1, int(np.ceil(minutes / env.config.agent_step_min)))
    term = trunc = False
    for _ in range(steps):
        _, r, term, trunc, info = env.step(zero)
        total += float(r)
        if term or trunc:
            break
    state = info["state"]
    compact = {
        key: float(state[key])
        for key in (
            "glucose_mg_dl", "insulin_uU_ml", "heart_rate_bpm", "map_mmHg",
            "cardiac_output_l_min", "pao2_mmHg", "paco2_mmHg", "ph_arterial",
            "sodium_mmol_l", "potassium_mmol_l", "gfr_ml_min",
            "oxygen_delivery_ml_min", "oxygen_extraction_ratio",
        )
        if key in state
    }
    return {
        "scenario": scenario,
        "requested_minutes": float(minutes),
        "simulated_minutes": float(info["time_min"]),
        "return": total,
        "terminated": bool(term),
        "truncated": bool(trunc),
        "termination_reason": info.get("termination_reason"),
        "summary_state": compact,
        "state": state,
    }


def _source_checkout_root() -> Path | None:
    """Find a checkout containing the version-locked validation assets."""

    candidates = [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
    seen: set[Path] = set()
    for candidate in candidates:
        root = candidate.resolve()
        if root in seen:
            continue
        seen.add(root)
        if (
            (root / "pyproject.toml").is_file()
            and (root / "tests").is_dir()
            and (root / "validation" / "run_validation_v22.py").is_file()
        ):
            return root
    return None


def _run_repo_command(args: list[str]) -> int:
    root = _source_checkout_root()
    if root is None:
        print(
            "openhumsim validate requires a source checkout containing tests/ "
            "and validation/. Clone https://github.com/lysyloxidase/openhumsim-rl "
            "and run the command from that checkout.",
            file=sys.stderr,
        )
        return 2
    proc = subprocess.run(args, cwd=root)
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openhumsim", description="OpenHumSim-RL local research CLI")
    parser.add_argument("--version", action="version", version=f"openhumsim-rl {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Check Python, dependencies and local accelerator availability")

    p_demo = sub.add_parser("demo", help="Run a zero-action local simulation smoke test")
    p_demo.add_argument("--scenario", default="baseline")
    p_demo.add_argument("--minutes", type=float, default=60.0)
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument("--full-state", action="store_true", help="Print all mechanistic state variables")

    p_val = sub.add_parser("validate", help="Run repository tests and v0.22 integrity checks")
    p_val.add_argument("--scientific-only", action="store_true")

    p_data = sub.add_parser("data", help="External-data utilities")
    data_sub = p_data.add_subparsers(dest="data_command", required=True)
    p_ref = data_sub.add_parser("reference", help="Print the published healthy-CGM reference metrics")
    p_dl = data_sub.add_parser("download-jaeb-cgm", help="Show the official Jaeb manual-download workflow (terms must be accepted on the website)")
    p_dl.add_argument("--out", default="data/external/CGMND.zip")
    data_sub.add_parser("jaeb-download-instructions", help="Print the official Jaeb download page and required manual steps")
    p_ins = data_sub.add_parser("inspect-jaeb-cgm", help="List CSV files inside a downloaded Jaeb archive")
    p_ins.add_argument("archive")
    p_sum = data_sub.add_parser("summarize-jaeb-cgm", help="Compute Shah-2019-style metrics from the Jaeb archive")
    p_sum.add_argument("archive")
    p_sum.add_argument("--glucose-column")
    p_sum.add_argument("--subject-column")
    p_sum.add_argument("--out", default="validation/jaeb_cgm_summary_local.json")
    p_age = data_sub.add_parser("shah-age-strata", help="Print exact Shah-2019 Table-2 age-stratified CGM targets")
    p_split = data_sub.add_parser("split-jaeb-cgm", help="Create participant-level train/validation/test splits from a Jaeb archive")
    p_split.add_argument("archive")
    p_split.add_argument("--seed", type=int, default=2019)
    p_split.add_argument("--glucose-column")
    p_split.add_argument("--subject-column")
    p_split.add_argument("--out", default="validation/jaeb_cgm_split_v0.9.json")
    p_fit = data_sub.add_parser("fit-jaeb-reference", help="Fit a TRAIN-only normative CGM distribution and evaluate validation/test subjects")
    p_fit.add_argument("archive")
    p_fit.add_argument("--seed", type=int, default=2019)
    p_fit.add_argument("--glucose-column")
    p_fit.add_argument("--subject-column")
    p_fit.add_argument("--out", default="validation/jaeb_cgm_reference_fit_v0.9.json")

    p_schema = data_sub.add_parser("inspect-jaeb-schema", help="Inventory all CSV schemas and event/CGM detection scores in a Jaeb ZIP")
    p_schema.add_argument("archive")
    p_schema.add_argument("--out", default="validation/jaeb_schema_v0.10.json")
    p_events = data_sub.add_parser("extract-jaeb-events", help="Auto-detect meal/exercise/sleep/wake diary events in a Jaeb ZIP")
    p_events.add_argument("archive")
    p_events.add_argument("--out", default="validation/jaeb_events_v0.10.json")
    p_em = data_sub.add_parser("evaluate-jaeb-events", help="Align Jaeb diary events to CGM using DuBose-2020 inclusion rules")
    p_em.add_argument("archive")
    p_em.add_argument("--out", default="validation/jaeb_event_metrics_v0.10.json")
    p_mech = data_sub.add_parser("fit-jaeb-event-model", help="TRAIN-only mechanistic event calibration with held-out participant validation/test")
    p_mech.add_argument("archive")
    p_mech.add_argument("--seed", type=int, default=2020)
    p_mech.add_argument("--out", default="validation/jaeb_mechanistic_event_fit_v0.10.json")
    p_pub = data_sub.add_parser("calibrate-published-event-reference", help="Calibrate the replay layer to published DuBose meal/exercise aggregate targets")
    p_pub.add_argument("--out", default="validation/published_event_calibration_v0.10.json")

    p_measure = sub.add_parser("measurement-demo", help="Compare realistic clinical measurements with hidden mechanistic truth")
    p_measure.add_argument("--scenario", default="baseline")
    p_measure.add_argument("--minutes", type=float, default=40.0)
    p_measure.add_argument("--seed", type=int, default=42)

    p_pop = sub.add_parser("population-demo", help="Sample the v0.21 correlated engineering virtual-patient prior")
    p_pop.add_argument("--n", type=int, default=8)
    p_pop.add_argument("--seed", type=int, default=2020)
    p_pop.add_argument("--independent", action="store_true", help="Use the legacy independent LHS prior")

    p_cgm = sub.add_parser("cgm-demo", help="Show blood-to-interstitial CGM observation-model lag")
    p_cgm.add_argument("--lag-min", type=float, default=6.0)
    p_cgm.add_argument("--step-min", type=float, default=1.0)

    args = parser.parse_args(argv)

    if args.command == "doctor":
        print(json.dumps(doctor(), indent=2))
        return 0
    if args.command == "demo":
        result = run_demo(args.scenario, args.minutes, args.seed)
        if not args.full_state:
            result = {k: v for k, v in result.items() if k != "state"}
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "validate":
        if not args.scientific_only:
            rc = _run_repo_command(
                [sys.executable, "-m", "pytest", "-q", "-ra", "-W", "error"]
            )
            if rc:
                return rc
        return _run_repo_command([sys.executable, "validation/run_validation_v22.py"])
    if args.command == "measurement-demo":
        env = HumanHomeostasisEnv(scenario=args.scenario, measurement_profile="realistic")
        _, info = env.reset(seed=args.seed)
        zero = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        steps = max(1, int(np.ceil(args.minutes / env.config.agent_step_min)))
        for _ in range(steps):
            _, _, term, trunc, info = env.step(zero)
            if term or trunc:
                break
        measured = {
            name: env.measurement_model.measurement_value(name, env.state)
            for name in ("sensor_glucose_mg_dl", "spo2_pct", "pao2_mmHg", "paco2_mmHg", "sodium_mmol_l", "potassium_mmol_l")
        }
        truth = {
            "blood_glucose_mg_dl": float(env.state.glucose_mg_dl),
            "spo2_pct": float(env.state.spo2_pct),
            "pao2_mmHg": float(env.state.pao2_mmHg),
            "paco2_mmHg": float(env.state.paco2_mmHg),
            "sodium_mmol_l": float(env.state.sodium_mmol_l),
            "potassium_mmol_l": float(env.state.potassium_mmol_l),
        }
        print(json.dumps({"scenario":args.scenario,"time_min":info["time_min"],"measured":measured,"hidden_truth_debug":truth,"measurement":info["measurement"]},indent=2))
        return 0
    if args.command == "population-demo":
        cohort = sample_virtual_cohort(args.n, seed=args.seed, correlated=not args.independent)
        print(json.dumps({
            "n": len(cohort),
            "prior": "independent_lhs" if args.independent else "correlated_engineering_lhs",
            "patients": [p.metadata() for p in cohort],
        }, indent=2))
        return 0
    if args.command == "cgm-demo":
        # A deterministic step challenge purely to visualize the observation model.
        blood = np.r_[np.full(10, 90.0), np.full(40, 160.0)].astype(float)
        cgm = blood_to_cgm_trace(
            blood,
            dt_min=args.step_min,
            config=CGMObservationConfig(lag_tau_min=args.lag_min),
        )
        print(json.dumps({
            "lag_tau_min": float(args.lag_min),
            "step_min": float(args.step_min),
            "blood_trace_mg_dl": blood.tolist(),
            "cgm_trace_mg_dl": [float(x) for x in cgm],
        }, indent=2))
        return 0
    if args.command == "data":
        if args.data_command == "reference":
            print(json.dumps({"official_dataset_page": JAEB_DATASET_PAGE_URL, "legacy_direct_object": JAEB_HEALTHY_CGM_URL, "published": REFERENCE.as_dict(), "event_reference": DUBOSE_REFERENCE.as_dict()}, indent=2))
            return 0
        if args.data_command in {"download-jaeb-cgm", "jaeb-download-instructions"}:
            payload = jaeb_download_instructions()
            payload["suggested_local_path"] = args.out if hasattr(args, "out") else "data/external/CGMND.zip"
            print(json.dumps(payload, indent=2))
            return 0
        if args.data_command == "inspect-jaeb-cgm":
            print(json.dumps(list_archive_csvs(args.archive), indent=2))
            return 0
        if args.data_command == "summarize-jaeb-cgm":
            summary = summarize_cgm_archive(args.archive, args.glucose_column, args.subject_column)
            save_summary(summary, args.out)
            print(json.dumps(summary["population"], indent=2))
            print(f"saved: {args.out}")
            return 0
        if args.data_command == "shah-age-strata":
            print(json.dumps(shah_2019_age_strata_reference(), indent=2))
            return 0
        if args.data_command == "split-jaeb-cgm":
            summary = summarize_cgm_archive(args.archive, args.glucose_column, args.subject_column)
            report = build_subject_split_report(
                summary, seed=args.seed, source_fingerprint=_sha256_file(args.archive)
            )
            save_summary(report, args.out)
            compact = {
                "leakage_check": report["leakage_check"],
                "split_sizes": {k: v["metrics"]["n_subjects"] for k, v in report["splits"].items()},
                "metrics": {k: v["metrics"] for k, v in report["splits"].items()},
            }
            print(json.dumps(compact, indent=2))
            print(f"saved: {args.out}")
            return 0
        if args.data_command == "fit-jaeb-reference":
            summary = summarize_cgm_archive(args.archive, args.glucose_column, args.subject_column)
            result = calibrate_normative_cgm_reference(summary, seed=args.seed)
            payload = result.as_dict()
            save_summary(payload, args.out)
            print(json.dumps({
                "split_sizes": {k: v["metrics"]["n_subjects"] for k, v in payload["split_report"]["splits"].items()},
                "test_evaluation": payload["evaluation"]["test"],
                "leakage_check": payload["split_report"]["leakage_check"],
            }, indent=2))
            print(f"saved: {args.out}")
            return 0
        if args.data_command == "inspect-jaeb-schema":
            payload = archive_schema_report(args.archive)
            save_event_json(payload, args.out)
            print(json.dumps({"tables": [{"file": x.get("file"), "n_rows": x.get("n_rows"), "event_score": x.get("event_score"), "cgm_score": x.get("cgm_score")} for x in payload["tables"]]}, indent=2))
            print(f"saved: {args.out}")
            return 0
        if args.data_command == "extract-jaeb-events":
            payload = extract_events_from_archive(args.archive)
            save_event_json(payload, args.out)
            print(json.dumps({"n_events": payload["n_events"], "counts": payload["counts"], "tables": payload["tables"]}, indent=2))
            print(f"saved: {args.out}")
            return 0
        if args.data_command == "evaluate-jaeb-events":
            payload = event_metrics_from_archive(args.archive)
            summary = summarize_event_metrics(payload)
            payload["summary"] = summary
            save_event_json(payload, args.out)
            print(json.dumps({"n_meal_metrics": payload["n_meal_metrics"], "n_exercise_metrics": payload["n_exercise_metrics"], "summary": summary, "exclusions": payload["exclusions"]}, indent=2))
            print(f"saved: {args.out}")
            return 0
        if args.data_command == "fit-jaeb-event-model":
            metrics = event_metrics_from_archive(args.archive)
            payload = fit_mechanistic_event_profile(metrics, seed=args.seed)
            save_event_json(payload, args.out)
            print(json.dumps({"profile": payload["profile"], "leakage_check": payload["leakage_check"], "errors": {k: v["error"] for k, v in payload["evaluation"].items()}}, indent=2))
            print(f"saved: {args.out}")
            return 0
        if args.data_command == "calibrate-published-event-reference":
            payload = calibrate_aggregate_event_profile()
            save_event_json(payload, args.out)
            print(json.dumps(payload, indent=2))
            print(f"saved: {args.out}")
            return 0
    parser.error("Unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
