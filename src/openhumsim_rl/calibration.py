from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable
import numpy as np

from .config import HumanConfig
from .env import HumanHomeostasisEnv


@dataclass(frozen=True)
class CalibrationTarget:
    name: str
    value: float
    scale: float


@dataclass
class CalibrationResult:
    parameter_names: list[str]
    initial: dict[str, float]
    fitted: dict[str, float]
    target_values: dict[str, float]
    fitted_outputs: dict[str, float]
    normalized_rmse: float
    success: bool
    message: str


def reference_outputs(config: HumanConfig, seed: int = 123) -> dict[str, float]:
    env = HumanHomeostasisEnv(config=config, scenario="baseline")
    _, info = env.reset(seed=seed)
    s = info["state"]
    return {
        "cardiac_output_l_min": float(s["cardiac_output_l_min"]),
        "map_mmHg": float(s["map_mmHg"]),
        "gfr_ml_min": float(s["gfr_ml_min"]),
        "paco2_mmHg": float(s["paco2_mmHg"]),
        "pao2_mmHg": float(s["pao2_mmHg"]),
    }


def fit_reference_profile(
    base_config: HumanConfig,
    targets: list[CalibrationTarget],
    parameter_bounds: dict[str, tuple[float, float]],
    seed: int = 123,
) -> CalibrationResult:
    """Least-squares fit of a small parameter subset to a reference profile.

    This is an estimation utility, not clinical calibration. In particular, fitting
    to literature centroids/ranges does not create subject-specific validity.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("fit_reference_profile requires scipy; install openhumsim-rl[uq]") from exc

    names = list(parameter_bounds)
    x0 = np.asarray([getattr(base_config, n) for n in names], dtype=float)
    lo = np.asarray([parameter_bounds[n][0] for n in names], dtype=float)
    hi = np.asarray([parameter_bounds[n][1] for n in names], dtype=float)
    target_map = {t.name: t for t in targets}

    def cfg_from_x(x):
        return replace(base_config, **{n: float(v) for n, v in zip(names, x)})

    def residual(x):
        out = reference_outputs(cfg_from_x(x), seed=seed)
        return np.asarray([
            (out[t.name] - t.value) / max(1e-12, t.scale) for t in targets
        ], dtype=float)

    opt = least_squares(residual, x0=x0, bounds=(lo, hi), max_nfev=80, xtol=1e-8, ftol=1e-8)
    fitted_cfg = cfg_from_x(opt.x)
    fitted_out = reference_outputs(fitted_cfg, seed=seed)
    r = residual(opt.x)
    return CalibrationResult(
        parameter_names=names,
        initial={n: float(v) for n, v in zip(names, x0)},
        fitted={n: float(v) for n, v in zip(names, opt.x)},
        target_values={t.name: float(t.value) for t in targets},
        fitted_outputs={k: float(v) for k, v in fitted_out.items()},
        normalized_rmse=float(np.sqrt(np.mean(r*r))),
        success=bool(opt.success),
        message=str(opt.message),
    )
