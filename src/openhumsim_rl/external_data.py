from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import csv
import hashlib
import io
import json
import re
import zipfile
import numpy as np

# Public Jaeb Center dataset listed on the study website.
JAEB_DATASET_PAGE_URL = "https://public.jaeb.org/dataset/559"
# Retained for provenance and compatibility. The direct S3 URL is not automated
# because the official Jaeb page requires identifying fields and acceptance of
# the dataset terms before downloading.
JAEB_HEALTHY_CGM_URL = (
    "https://live-jchrpublicdatasets.s3.amazonaws.com/Diabetes/Public%20Datasets/"
    "CGMND-af920dee-2d6e-4436-bc89-7a7b51239837.zip"
)


@dataclass(frozen=True)
class HealthyCGMReference:
    """Published aggregate metrics from Shah et al., JCEM 2019.

    These are external human reference summaries, not calibration priors and not
    a replacement for protocol-matched individual-level validation.
    """

    n_participants: int = 153
    age_range_years: str = "7-80"
    mean_average_glucose_mg_dl_typical_age_groups: tuple[float, float] = (98.0, 99.0)
    mean_average_glucose_mg_dl_over_60: float = 104.0
    median_time_70_140_pct: float = 96.0
    time_70_140_iqr_pct: tuple[float, float] = (93.0, 98.0)
    mean_within_person_cv_pct: float = 17.0
    within_person_cv_sd_pct: float = 3.0
    median_time_gt_140_pct: float = 2.1
    median_time_lt_70_pct: float = 1.1
    citation: str = "Shah VN et al. J Clin Endocrinol Metab. 2019;104(10):4356-4364. PMID:31127824"

    def as_dict(self) -> dict:
        return asdict(self)


REFERENCE = HealthyCGMReference()


def jaeb_download_instructions() -> dict:
    """Return the official manual-download workflow for the Jaeb dataset.

    The Jaeb download page requires a name, email, institution/planned use and an
    explicit agreement to its terms. OpenHumSim therefore does not bypass that
    form by fetching the underlying S3 object directly.
    """
    return {
        "dataset_page": JAEB_DATASET_PAGE_URL,
        "expected_filename": "CGMND.zip",
        "steps": [
            "Open the official Jaeb dataset page.",
            "Provide the requested contact/institution/planned-use fields.",
            "Read and accept the Jaeb dataset terms.",
            "Download CGMND.zip and place it under data/external/CGMND.zip.",
        ],
    }


def download_jaeb_healthy_cgm(destination: str | Path) -> Path:
    """Deprecated safety guard: Jaeb requires manual acceptance of its terms."""
    raise PermissionError(
        "OpenHumSim v0.10 does not bypass the Jaeb download form. "
        f"Download CGMND.zip manually from {JAEB_DATASET_PAGE_URL} after accepting the terms, "
        f"then place it at {Path(destination)}."
    )


def list_archive_csvs(archive: str | Path) -> list[str]:
    with zipfile.ZipFile(archive) as zf:
        return sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _score_glucose_column(name: str) -> int:
    n = _norm(name)
    score = 0
    if "glucose" in n:
        score += 10
    if "sensor" in n or "cgm" in n:
        score += 5
    if n in {"sg", "sgv", "value", "glucosemgdl", "sensorglucose"}:
        score += 4
    if "mgdl" in n:
        score += 2
    return score


def _score_subject_column(name: str) -> int:
    n = _norm(name)
    score = 0
    for token in ("subject", "participant", "patient", "person"):
        if token in n:
            score += 5
    if n.endswith("id") or n in {"id", "pid", "subjectid", "participantid"}:
        score += 3
    return score


def _to_float(value: str) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def _read_csv_from_bytes(data: bytes) -> tuple[list[str], list[dict[str, str]]]:
    text = data.decode("utf-8-sig", errors="replace")
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    fields = list(reader.fieldnames or [])
    return fields, list(reader)


def detect_cgm_table(
    archive: str | Path,
    glucose_column: str | None = None,
    subject_column: str | None = None,
) -> dict:
    """Detect a likely CGM table in a ZIP and return rows + selected columns.

    The Jaeb archive format can evolve; therefore automatic detection is used only
    as a convenience. The CLI reports the chosen columns and lets the user override
    them explicitly.
    """
    archive = Path(archive)
    candidates = []
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            try:
                fields, rows = _read_csv_from_bytes(zf.read(name))
            except Exception:
                continue
            if not fields or not rows:
                continue
            gcol = glucose_column if glucose_column in fields else None
            if gcol is None:
                scored = sorted((( _score_glucose_column(f), f) for f in fields), reverse=True)
                if scored and scored[0][0] > 0:
                    gcol = scored[0][1]
            scol = subject_column if subject_column in fields else None
            if scol is None:
                scored = sorted(((_score_subject_column(f), f) for f in fields), reverse=True)
                if scored and scored[0][0] > 0:
                    scol = scored[0][1]
            if not gcol:
                continue
            numeric = sum(_to_float(r.get(gcol, "")) is not None for r in rows[:1000])
            candidates.append((numeric, len(rows), name, fields, rows, gcol, scol))
    if not candidates:
        raise ValueError("No CSV with a plausible numeric glucose column was found in the archive.")
    _, _, name, fields, rows, gcol, scol = max(candidates, key=lambda x: (x[0], x[1]))
    return {
        "file": name,
        "fields": fields,
        "rows": rows,
        "glucose_column": gcol,
        "subject_column": scol,
    }


def summarize_cgm_archive(
    archive: str | Path,
    glucose_column: str | None = None,
    subject_column: str | None = None,
) -> dict:
    table = detect_cgm_table(archive, glucose_column, subject_column)
    rows = table["rows"]
    gcol = table["glucose_column"]
    scol = table["subject_column"]

    groups: dict[str, list[float]] = {}
    for i, row in enumerate(rows):
        x = _to_float(row.get(gcol, ""))
        if x is None or not (20.0 <= x <= 600.0):
            continue
        sid = str(row.get(scol, "")) if scol else "all"
        if not sid:
            sid = f"row-{i}"
        groups.setdefault(sid, []).append(x)
    groups = {k: v for k, v in groups.items() if len(v) >= 10}
    if not groups:
        raise ValueError("No subject/group with at least 10 valid glucose readings was found.")

    subject_metrics = []
    for sid, vals in groups.items():
        a = np.asarray(vals, dtype=float)
        mean = float(np.mean(a))
        cv = float(100.0 * np.std(a, ddof=1) / mean) if len(a) > 1 and mean else 0.0
        subject_metrics.append({
            "subject": sid,
            "n": int(len(a)),
            "mean_mg_dl": mean,
            "cv_pct": cv,
            "time_70_140_pct": float(100.0 * np.mean((a >= 70.0) & (a <= 140.0))),
            "time_gt_140_pct": float(100.0 * np.mean(a > 140.0)),
            "time_lt_70_pct": float(100.0 * np.mean(a < 70.0)),
        })

    def arr(key):
        return np.asarray([m[key] for m in subject_metrics], dtype=float)

    summary = {
        "archive": str(Path(archive)),
        "detected_file": table["file"],
        "glucose_column": gcol,
        "subject_column": scol,
        "n_subjects": len(subject_metrics),
        "n_readings": int(sum(m["n"] for m in subject_metrics)),
        "population": {
            "mean_of_subject_mean_glucose_mg_dl": float(np.mean(arr("mean_mg_dl"))),
            "median_time_70_140_pct": float(np.median(arr("time_70_140_pct"))),
            "mean_within_person_cv_pct": float(np.mean(arr("cv_pct"))),
            "median_time_gt_140_pct": float(np.median(arr("time_gt_140_pct"))),
            "median_time_lt_70_pct": float(np.median(arr("time_lt_70_pct"))),
        },
        "published_reference": REFERENCE.as_dict(),
        "subject_metrics": subject_metrics,
        "caveat": (
            "CGM is interstitial glucose and this dataset is a free-living observational reference. "
            "It is not protocol-matched to the OpenHumSim meal challenge and is therefore an external "
            "distribution benchmark, not a direct dynamic-model likelihood without additional context."
        ),
    }
    return summary


def save_summary(summary: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


@dataclass(frozen=True)
class Shah2019AgeStratum:
    age_group: str
    n: int
    mean_glucose_mg_dl: float
    mean_glucose_sd_mg_dl: float
    mean_within_person_sd_mg_dl: float
    mean_within_person_cv_pct: float
    time_70_140_median_pct: float
    time_70_140_iqr_pct: tuple[float, float]
    time_gt_140_median_pct: float
    time_lt_70_median_pct: float

    def as_dict(self) -> dict:
        return asdict(self)


# Exact age-stratified values transcribed from Shah et al. JCEM 2019, Table 2.
# These are observational CGM summaries, not physiological set points.
SHAH_2019_AGE_STRATA: tuple[Shah2019AgeStratum, ...] = (
    Shah2019AgeStratum("6-<12", 27, 99.0, 7.0, 16.0, 16.0, 97.0, (94.0, 97.0), 1.7, 1.1),
    Shah2019AgeStratum("12-<18", 30, 98.0, 6.0, 15.0, 15.0, 97.0, (95.0, 98.0), 1.2, 1.7),
    Shah2019AgeStratum("18-<25", 29, 98.0, 6.0, 18.0, 18.0, 95.0, (91.0, 97.0), 2.4, 1.3),
    Shah2019AgeStratum("25-<60", 41, 99.0, 6.0, 16.0, 16.0, 97.0, (94.0, 98.0), 2.1, 1.0),
    Shah2019AgeStratum(">=60", 26, 104.0, 9.0, 18.0, 17.0, 93.0, (89.0, 96.0), 4.1, 1.4),
)


def shah_2019_age_strata_reference() -> list[dict]:
    return [x.as_dict() for x in SHAH_2019_AGE_STRATA]


def deterministic_subject_split(
    subject_ids: Iterable[str],
    seed: int = 2019,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> dict[str, list[str]]:
    """Deterministic participant-level train/validation/test split.

    Splitting is explicitly by subject identifier to prevent readings from one
    participant leaking across partitions.
    """
    ids = sorted({str(x) for x in subject_ids})
    if len(ids) < 3:
        raise ValueError("At least 3 distinct subjects are required for a split")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0,1)")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be in (0,1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train + validation fractions must be < 1")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(ids, dtype=object)[rng.permutation(len(ids))].tolist()
    n = len(shuffled)
    n_train = max(1, int(round(n * train_fraction)))
    n_val = max(1, int(round(n * validation_fraction)))
    if n_train + n_val >= n:
        n_val = 1
        n_train = n - 2
    return {
        "train": sorted(shuffled[:n_train]),
        "validation": sorted(shuffled[n_train:n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val:]),
    }


def _aggregate_subject_metrics(metrics: list[dict]) -> dict:
    if not metrics:
        raise ValueError("metrics cannot be empty")

    def a(key: str) -> np.ndarray:
        return np.asarray([float(m[key]) for m in metrics], dtype=float)

    return {
        "n_subjects": len(metrics),
        "mean_of_subject_mean_glucose_mg_dl": float(np.mean(a("mean_mg_dl"))),
        "sd_of_subject_mean_glucose_mg_dl": float(np.std(a("mean_mg_dl"), ddof=1)) if len(metrics) > 1 else 0.0,
        "mean_within_person_cv_pct": float(np.mean(a("cv_pct"))),
        "median_time_70_140_pct": float(np.median(a("time_70_140_pct"))),
        "median_time_gt_140_pct": float(np.median(a("time_gt_140_pct"))),
        "median_time_lt_70_pct": float(np.median(a("time_lt_70_pct"))),
    }


def _bootstrap_ci(
    metrics: list[dict],
    key: str,
    statistic: str,
    seed: int,
    n_boot: int = 1000,
) -> tuple[float, float]:
    values = np.asarray([float(m[key]) for m in metrics], dtype=float)
    if len(values) < 2:
        x = float(values[0])
        return (x, x)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = values[rng.integers(0, len(values), size=len(values))]
        stats[i] = np.mean(draw) if statistic == "mean" else np.median(draw)
    return (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975)))


def build_subject_split_report(
    summary: dict,
    seed: int = 2019,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    n_boot: int = 1000,
    source_fingerprint: str = "",
) -> dict:
    """Create leakage-safe subject splits and bootstrap metrics from a CGM summary."""
    subject_metrics = list(summary.get("subject_metrics", []))
    if not subject_metrics:
        raise ValueError("summary has no subject_metrics")
    by_id = {str(m["subject"]): m for m in subject_metrics}
    splits = deterministic_subject_split(
        by_id, seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    out = {
        "seed": int(seed),
        "fractions": {
            "train": float(train_fraction),
            "validation": float(validation_fraction),
            "test": float(1.0 - train_fraction - validation_fraction),
        },
        "splits": {},
        "leakage_check": {},
        "published_reference": REFERENCE.as_dict(),
    }
    for j, (name, ids) in enumerate(splits.items()):
        ms = [by_id[i] for i in ids]
        agg = _aggregate_subject_metrics(ms)
        agg["bootstrap_95_ci"] = {
            "mean_glucose_mg_dl": _bootstrap_ci(ms, "mean_mg_dl", "mean", seed + 10*j + 1, n_boot),
            "within_person_cv_pct": _bootstrap_ci(ms, "cv_pct", "mean", seed + 10*j + 2, n_boot),
            "time_70_140_pct": _bootstrap_ci(ms, "time_70_140_pct", "median", seed + 10*j + 3, n_boot),
            "time_gt_140_pct": _bootstrap_ci(ms, "time_gt_140_pct", "median", seed + 10*j + 4, n_boot),
            "time_lt_70_pct": _bootstrap_ci(ms, "time_lt_70_pct", "median", seed + 10*j + 5, n_boot),
        }
        out["splits"][name] = {"subject_ids": ids, "metrics": agg}

    sets = {k: set(v) for k, v in splits.items()}
    overlaps = {
        "train_validation": sorted(sets["train"] & sets["validation"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "validation_test": sorted(sets["validation"] & sets["test"]),
    }
    out["leakage_check"] = {
        "overlaps": overlaps,
        "passed": not any(overlaps.values()),
        "all_subjects_accounted_for": (
            len(sets["train"] | sets["validation"] | sets["test"]) == len(by_id)
        ),
    }
    # Freeze the held-out test identifiers in a tamper-evident hash.
    # This is a locked holdout within Jaeb, not a claim of an independent external
    # cohort; a separate dataset can use LockedCohortManifest from population.py.
    lock_payload = json.dumps({
        "dataset": "Jaeb healthy non-diabetic CGM",
        "source_fingerprint": str(source_fingerprint),
        "role": "locked_test",
        "subject_ids": splits["test"],
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    out["locked_test"] = {
        "dataset": "Jaeb healthy non-diabetic CGM",
        "source_fingerprint": str(source_fingerprint),
        "subject_ids": splits["test"],
        "sha256": hashlib.sha256(lock_payload).hexdigest(),
        "note": "locked holdout within Jaeb; not an independent external dataset",
    }
    return out
