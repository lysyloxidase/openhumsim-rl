from __future__ import annotations

from dataclasses import dataclass, asdict
from math import erf, log, pi, sqrt
import numpy as np

from .external_data import build_subject_split_report


_METRICS = (
    ("mean_mg_dl", "normal"),
    ("cv_pct", "normal"),
    ("time_70_140_pct", "logit_normal"),
    ("time_gt_140_pct", "logit_normal"),
    ("time_lt_70_pct", "logit_normal"),
)


def _logit_pct(x: np.ndarray) -> np.ndarray:
    # Continuity correction prevents 0%/100% subject metrics from becoming infinite.
    p = np.clip(np.asarray(x, dtype=float) / 100.0, 5e-4, 1.0 - 5e-4)
    return np.log(p / (1.0 - p))


def _inv_logit_pct(x: float | np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 100.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class FittedMetricDistribution:
    metric: str
    family: str
    location: float
    scale: float
    train_n: int
    interval_95: tuple[float, float]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class NormativeCGMCalibration:
    split_report: dict
    fitted_on_train: dict[str, FittedMetricDistribution]
    evaluation: dict[str, dict]

    def as_dict(self) -> dict:
        return {
            "split_report": self.split_report,
            "fitted_on_train": {
                k: v.as_dict() for k, v in self.fitted_on_train.items()
            },
            "evaluation": self.evaluation,
            "interpretation": (
                "This is a train/validation/test calibration of the external CGM "
                "observation distribution. It is not a calibration of the mechanistic "
                "OpenHumSim physiology because the free-living meal/exercise inputs are "
                "not yet protocol-matched to each participant."
            ),
        }


def _fit_distribution(values: np.ndarray, metric: str, family: str) -> FittedMetricDistribution:
    values = np.asarray(values, dtype=float)
    z = _logit_pct(values) if family == "logit_normal" else values
    loc = float(np.mean(z))
    scale = float(np.std(z, ddof=1)) if len(z) > 1 else 1e-6
    scale = max(scale, 1e-6)
    low_z, high_z = loc - 1.96 * scale, loc + 1.96 * scale
    if family == "logit_normal":
        low, high = [float(v) for v in _inv_logit_pct(np.asarray([low_z, high_z]))]
    else:
        low, high = float(low_z), float(high_z)
    return FittedMetricDistribution(metric, family, loc, scale, len(values), (low, high))


def _normal_logpdf(z: np.ndarray, loc: float, scale: float) -> np.ndarray:
    return -0.5 * ((z - loc) / scale) ** 2 - log(scale) - 0.5 * log(2.0 * pi)


def _evaluate(values: np.ndarray, fit: FittedMetricDistribution) -> dict:
    values = np.asarray(values, dtype=float)
    z = _logit_pct(values) if fit.family == "logit_normal" else values
    ll = _normal_logpdf(z, fit.location, fit.scale)
    low, high = fit.interval_95
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "mean_log_likelihood": float(np.mean(ll)),
        "coverage_95_pct": float(100.0 * np.mean((values >= low) & (values <= high))),
        "interval_95": [float(low), float(high)],
    }


def calibrate_normative_cgm_reference(
    summary: dict,
    seed: int = 2019,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    n_boot: int = 1000,
) -> NormativeCGMCalibration:
    """Fit a subject-level normative CGM observation model on TRAIN only.

    Validation and test subjects are never used in parameter estimation. This layer
    provides a real-human external likelihood target for later mechanistic calibration.
    It intentionally does *not* tune OpenHumSim physiology without participant-matched
    meals/exercise/sleep inputs.
    """
    split = build_subject_split_report(
        summary,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        n_boot=n_boot,
    )
    by_id = {str(m["subject"]): m for m in summary["subject_metrics"]}
    train = [by_id[i] for i in split["splits"]["train"]["subject_ids"]]

    fitted: dict[str, FittedMetricDistribution] = {}
    for metric, family in _METRICS:
        vals = np.asarray([float(m[metric]) for m in train], dtype=float)
        fitted[metric] = _fit_distribution(vals, metric, family)

    evaluation: dict[str, dict] = {}
    for part in ("train", "validation", "test"):
        metrics = [by_id[i] for i in split["splits"][part]["subject_ids"]]
        evaluation[part] = {}
        for metric, _ in _METRICS:
            vals = np.asarray([float(m[metric]) for m in metrics], dtype=float)
            evaluation[part][metric] = _evaluate(vals, fitted[metric])

    return NormativeCGMCalibration(split, fitted, evaluation)
