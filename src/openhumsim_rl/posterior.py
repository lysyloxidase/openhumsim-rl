from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Sequence
import numpy as np

from .calibration import reference_outputs
from .config import HumanConfig
from .population import ParameterSpec, DEFAULT_PARAMETER_SPECS, virtual_patient_from_unit_row, correlated_latin_hypercube, latin_hypercube


@dataclass(frozen=True)
class GaussianTarget:
    name: str
    mean: float
    measurement_sd: float
    model_discrepancy_sd: float = 0.0

    @property
    def total_sd(self) -> float:
        return float(np.hypot(self.measurement_sd, self.model_discrepancy_sd))


@dataclass
class PosteriorResult:
    n_prior: int
    effective_sample_size: float
    log_evidence_relative: float
    targets: list[dict]
    posterior_parameter_summary: dict[str, dict[str, float]]
    posterior_output_summary: dict[str, dict[str, float]]
    top_particles: list[dict]

    def as_dict(self) -> dict:
        return asdict(self)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    c = np.cumsum(w)
    return float(np.interp(q, c, v))


def importance_calibrate(
    targets: Sequence[GaussianTarget],
    n_prior: int = 64,
    seed: int = 8080,
    base_config: HumanConfig | None = None,
    specs: tuple[ParameterSpec, ...] = DEFAULT_PARAMETER_SPECS,
    cv_internal_step_s: float | None = None,
    correlated_prior: bool = True,
) -> PosteriorResult:
    """Likelihood-weighted posterior over a bounded virtual-patient prior.

    This is a small importance-sampling calibration layer. It uses the
    engineering rank-correlated prior by default; set correlated_prior=False to
    select the independent-LHS prior. It explicitly includes model-discrepancy
    variance, preventing literature centroids from being treated
    as exact subject-level measurements. It remains a research credibility tool,
    not patient-specific Bayesian inference.
    """
    if n_prior < 4:
        raise ValueError("n_prior must be at least 4")
    base = base_config or HumanConfig()
    design = (
        correlated_latin_hypercube(n_prior, specs=specs, seed=seed)
        if correlated_prior
        else latin_hypercube(n_prior, len(specs), seed=seed)
    )
    particles = []
    logw = []
    for i, row in enumerate(design):
        vp = virtual_patient_from_unit_row(row, f"POST-{i:04d}", base_config=base, specs=specs)
        # Use the actual model configuration unless a caller explicitly requests
        # a numerical-ablation step, keeping calibration aligned with the
        # environment configuration it is meant to fit.
        cfg = (
            vp.config
            if cv_internal_step_s is None
            else replace(vp.config, cv_internal_step_s=float(cv_internal_step_s))
        )
        out = reference_outputs(cfg, seed=123)
        lw = 0.0
        for t in targets:
            sd = max(1e-9, t.total_sd)
            z = (out[t.name] - t.mean) / sd
            lw += -0.5 * z * z - np.log(sd * np.sqrt(2.0 * np.pi))
        particles.append((vp, out))
        logw.append(float(lw))

    logw = np.asarray(logw, dtype=float)
    m = float(np.max(logw))
    w = np.exp(logw - m)
    w /= np.sum(w)
    ess = float(1.0 / np.sum(w * w))

    param_summary = {}
    for spec in specs:
        vals = np.asarray([p.latent[spec.name] for p, _ in particles], dtype=float)
        param_summary[spec.name] = {
            "mean": float(np.sum(w * vals)),
            "q05": _weighted_quantile(vals, w, 0.05),
            "q50": _weighted_quantile(vals, w, 0.50),
            "q95": _weighted_quantile(vals, w, 0.95),
        }

    output_names = sorted(set(t.name for t in targets))
    output_summary = {}
    for name in output_names:
        vals = np.asarray([o[name] for _, o in particles], dtype=float)
        output_summary[name] = {
            "mean": float(np.sum(w * vals)),
            "q05": _weighted_quantile(vals, w, 0.05),
            "q50": _weighted_quantile(vals, w, 0.50),
            "q95": _weighted_quantile(vals, w, 0.95),
        }

    top_idx = np.argsort(w)[::-1][: min(10, n_prior)]
    top = []
    for j in top_idx:
        vp, out = particles[int(j)]
        top.append({
            "patient_id": vp.patient_id,
            "weight": float(w[j]),
            "latent": dict(vp.latent),
            "outputs": {k: float(v) for k, v in out.items()},
        })

    return PosteriorResult(
        n_prior=n_prior,
        effective_sample_size=ess,
        log_evidence_relative=float(m + np.log(np.sum(np.exp(logw - m))) - np.log(n_prior)),
        targets=[asdict(t) | {"total_sd": t.total_sd} for t in targets],
        posterior_parameter_summary=param_summary,
        posterior_output_summary=output_summary,
        top_particles=top,
    )
