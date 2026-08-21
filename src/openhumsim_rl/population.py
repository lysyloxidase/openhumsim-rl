from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Iterable
from statistics import NormalDist
import hashlib
import json
import numpy as np

from .config import HumanConfig


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    low: float
    high: float
    distribution: str = "uniform"
    description: str = ""


@dataclass(frozen=True)
class VirtualPatient:
    patient_id: str
    config: HumanConfig
    latent: dict[str, float]

    def metadata(self) -> dict:
        return {"patient_id": self.patient_id, "latent": dict(self.latent)}


# These are engineering uncertainty intervals for robustness/UQ, not claims of
# population distributions. They deliberately span plausible adult variability.
DEFAULT_PARAMETER_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec("body_weight_kg", 55.0, 95.0, description="adult body mass"),
    ParameterSpec("blood_volume_ml_per_kg", 65.0, 80.0, description="blood-volume scaling"),
    ParameterSpec("tbw_fraction", 0.50, 0.65, description="total-body-water fraction of body mass"),
    ParameterSpec("ecf_fraction", 0.18, 0.23, description="ECF fraction of body mass"),
    ParameterSpec("hemoglobin_g_dl", 12.5, 16.5),
    ParameterSpec("baseline_gfr_ml_min", 90.0, 130.0),
    ParameterSpec("cv_resting_hr_bpm", 55.0, 85.0),
    ParameterSpec("cv_r_systemic_scale", 0.85, 1.20),
    ParameterSpec("cv_lv_emax_scale", 0.85, 1.15),
    # V/Q heterogeneity parameter. baseline_aa_gradient_mmHg is compatibility
    # reset scaffolding and is overwritten by the pulmonary exchange solve.
    ParameterSpec("pulmonary_baseline_vq_log_sd", 0.08, 0.30),
    ParameterSpec("dalla_insulin_sensitivity_scale", 0.75, 1.25),
    ParameterSpec("dalla_gastric_absorption_scale", 0.85, 1.15),
    ParameterSpec("respiratory_acidosis_efficiency", 0.55, 0.72),
    ParameterSpec("pbpk_fraction_unbound", 0.25, 0.75),
    ParameterSpec("pbpk_hepatic_clint_l_min", 0.10, 0.28),
)


# Engineering rank-correlation prior. These coefficients intentionally encode
# only a few robust physiologic dependencies; they are NOT a fitted population
# covariance matrix. Marginal bounds remain engineering uncertainty intervals.
DEFAULT_RANK_CORRELATIONS: dict[tuple[str, str], float] = {
    ("body_weight_kg", "tbw_fraction"): -0.35,
    ("tbw_fraction", "ecf_fraction"): 0.50,
    ("body_weight_kg", "blood_volume_ml_per_kg"): -0.20,
    ("blood_volume_ml_per_kg", "hemoglobin_g_dl"): 0.20,
    ("body_weight_kg", "baseline_gfr_ml_min"): 0.15,
    ("cv_resting_hr_bpm", "cv_r_systemic_scale"): 0.20,
    ("dalla_insulin_sensitivity_scale", "body_weight_kg"): -0.20,
}


def correlation_matrix_for_specs(
    specs: tuple[ParameterSpec, ...] = DEFAULT_PARAMETER_SPECS,
    correlations: dict[tuple[str, str], float] | None = None,
) -> np.ndarray:
    """Build a symmetric positive-definite engineering correlation matrix.

    If the sparse requested matrix is not numerically positive definite, it is
    projected by eigenvalue clipping and renormalized to unit diagonal.
    """
    corr = DEFAULT_RANK_CORRELATIONS if correlations is None else correlations
    names = [s.name for s in specs]
    idx = {n: i for i, n in enumerate(names)}
    R = np.eye(len(specs), dtype=float)
    for (a, b), value in corr.items():
        if a not in idx or b not in idx:
            continue
        v = float(np.clip(value, -0.95, 0.95))
        R[idx[a], idx[b]] = v
        R[idx[b], idx[a]] = v
    eigval, eigvec = np.linalg.eigh(R)
    eigval = np.maximum(eigval, 1e-6)
    # Keep this small deterministic reconstruction off platform BLAS.  Some
    # Accelerate/NumPy combinations emit spurious overflow warnings for the
    # equivalent 15x15 matmul even though its result is finite.
    R = np.einsum("ik,k,jk->ij", eigvec, eigval, eigvec, optimize=False)
    scale = np.sqrt(np.diag(R))
    R = R / np.outer(scale, scale)
    np.fill_diagonal(R, 1.0)
    return R


def correlated_latin_hypercube(
    n: int,
    specs: tuple[ParameterSpec, ...] = DEFAULT_PARAMETER_SPECS,
    seed: int | None = None,
    correlations: dict[tuple[str, str], float] | None = None,
) -> np.ndarray:
    """LHS design with approximately requested rank correlation.

    The Iman-Conover-style rank reorder preserves every original LHS marginal
    stratum while imposing correlation on ranks. No SciPy dependency is needed.
    """
    U = latin_hypercube(n, len(specs), seed=seed)
    if n < 3:
        return U
    R = correlation_matrix_for_specs(specs, correlations)
    nd = NormalDist()
    eps = 1e-9
    Z = np.empty_like(U)
    for j in range(U.shape[1]):
        Z[:, j] = [nd.inv_cdf(float(np.clip(u, eps, 1.0-eps))) for u in U[:, j]]
    L = np.linalg.cholesky(R)
    Y = np.einsum("ij,kj->ik", Z, L, optimize=False)
    out = np.empty_like(U)
    for j in range(U.shape[1]):
        source = np.sort(U[:, j])
        order = np.argsort(Y[:, j], kind="mergesort")
        out[order, j] = source
    return out


@dataclass(frozen=True)
class LockedCohortManifest:
    """Leakage-safe calibration/validation manifest with tamper-evident lock.

    `dataset_fingerprint` should be a SHA-256 (or other stable content digest) of
    the source archive/file when available. Locking IDs alone prevents split
    drift; locking the source fingerprint additionally detects dataset changes.
    """
    dataset_name: str
    dataset_fingerprint: str
    calibration_subject_ids: tuple[str, ...]
    validation_subject_ids: tuple[str, ...]
    split_seed: int
    validation_lock_sha256: str

    @classmethod
    def create(
        cls,
        subject_ids: Iterable[str],
        dataset_name: str,
        calibration_fraction: float = 0.70,
        seed: int = 2020,
        dataset_fingerprint: str = "",
    ) -> "LockedCohortManifest":
        ids = sorted({str(x) for x in subject_ids})
        if len(ids) < 4:
            raise ValueError("At least 4 distinct subjects are required")
        if not (0.2 <= calibration_fraction <= 0.9):
            raise ValueError("calibration_fraction must be between 0.2 and 0.9")
        rng = np.random.default_rng(seed)
        order = np.asarray(ids, dtype=object)
        rng.shuffle(order)
        ncal = int(round(len(ids) * calibration_fraction))
        ncal = min(max(2, ncal), len(ids)-2)
        calibration = tuple(sorted(str(x) for x in order[:ncal]))
        validation = tuple(sorted(str(x) for x in order[ncal:]))
        digest = cls._hash_validation(
            dataset_name,
            dataset_fingerprint,
            calibration,
            validation,
            int(seed),
        )
        return cls(dataset_name, str(dataset_fingerprint), calibration, validation, int(seed), digest)

    @staticmethod
    def _hash_validation(
        dataset_name: str,
        dataset_fingerprint: str,
        calibration_ids: tuple[str, ...],
        validation_ids: tuple[str, ...],
        split_seed: int,
    ) -> str:
        payload = json.dumps({
            "dataset_name": str(dataset_name),
            "dataset_fingerprint": str(dataset_fingerprint),
            "calibration_subject_ids": list(calibration_ids),
            "validation_subject_ids": list(validation_ids),
            "split_seed": int(split_seed),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def verify_lock(self) -> bool:
        return self.validation_lock_sha256 == self._hash_validation(
            self.dataset_name,
            self.dataset_fingerprint,
            self.calibration_subject_ids,
            self.validation_subject_ids,
            self.split_seed,
        )

    def assert_no_leakage(self) -> None:
        overlap = set(self.calibration_subject_ids) & set(self.validation_subject_ids)
        if overlap:
            raise ValueError(f"Calibration/validation leakage: {sorted(overlap)}")
        if not self.verify_lock():
            raise ValueError("Validation cohort lock hash mismatch")

    def as_dict(self) -> dict:
        self.assert_no_leakage()
        return {
            "dataset_name": self.dataset_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "calibration_subject_ids": list(self.calibration_subject_ids),
            "validation_subject_ids": list(self.validation_subject_ids),
            "split_seed": self.split_seed,
            "validation_lock_sha256": self.validation_lock_sha256,
        }


def latin_hypercube(n: int, d: int, seed: int | None = None) -> np.ndarray:
    """Dependency-free Latin-hypercube design in [0,1]^d."""
    if n <= 0 or d <= 0:
        raise ValueError("n and d must be positive")
    rng = np.random.default_rng(seed)
    u = rng.random((n, d))
    out = np.empty_like(u)
    for j in range(d):
        perm = rng.permutation(n)
        out[:, j] = (perm + u[:, j]) / n
    return out


def _map_unit(u: float, spec: ParameterSpec) -> float:
    if spec.distribution == "uniform":
        return float(spec.low + u * (spec.high - spec.low))
    if spec.distribution == "loguniform":
        return float(np.exp(np.log(spec.low) + u * (np.log(spec.high) - np.log(spec.low))))
    raise ValueError(f"Unsupported distribution: {spec.distribution}")


def virtual_patient_from_unit_row(
    row: Iterable[float],
    patient_id: str,
    base_config: HumanConfig | None = None,
    specs: tuple[ParameterSpec, ...] = DEFAULT_PARAMETER_SPECS,
) -> VirtualPatient:
    base = base_config or HumanConfig()
    row = list(row)
    if len(row) != len(specs):
        raise ValueError(f"Expected {len(specs)} unit values, got {len(row)}")
    latent = {s.name: _map_unit(float(u), s) for u, s in zip(row, specs)}

    bw = latent["body_weight_kg"]
    blood_volume_ml = bw * latent["blood_volume_ml_per_kg"]
    tbw_l = bw * latent["tbw_fraction"]
    ecf_l = bw * latent["ecf_fraction"]
    # Plasma volume is derived from blood volume with a fixed nominal plasma fraction.
    plasma_l = 0.60 * blood_volume_ml / 1000.0

    bv_scale = blood_volume_ml / base.cv_baseline_blood_volume_ml
    cfg = replace(
        base,
        body_weight_kg=bw,
        cv_baseline_blood_volume_ml=blood_volume_ml,
        total_body_water_baseline_l=tbw_l,
        ecf_volume_baseline_l=ecf_l,
        plasma_volume_baseline_l=plasma_l,
        # Keep the lumped circulation geometrically coherent when blood volume varies.
        # Initial/unstressed volumes and vascular compliances scale together, so
        # pressure does not change merely because the virtual adult is larger.
        cv_v_la0_ml=base.cv_v_la0_ml * bv_scale,
        cv_v_lv0_ml=base.cv_v_lv0_ml * bv_scale,
        cv_v_sa0_ml=base.cv_v_sa0_ml * bv_scale,
        cv_v_sv0_ml=base.cv_v_sv0_ml * bv_scale,
        cv_v_ra0_ml=base.cv_v_ra0_ml * bv_scale,
        cv_v_rv0_ml=base.cv_v_rv0_ml * bv_scale,
        cv_v_pa0_ml=base.cv_v_pa0_ml * bv_scale,
        cv_v_pv0_ml=base.cv_v_pv0_ml * bv_scale,
        cv_v0_la_ml=base.cv_v0_la_ml * bv_scale,
        cv_v0_lv_ml=base.cv_v0_lv_ml * bv_scale,
        cv_v0_sa_ml=base.cv_v0_sa_ml * bv_scale,
        cv_v0_sv_ml=base.cv_v0_sv_ml * bv_scale,
        cv_v0_ra_ml=base.cv_v0_ra_ml * bv_scale,
        cv_v0_rv_ml=base.cv_v0_rv_ml * bv_scale,
        cv_v0_pa_ml=base.cv_v0_pa_ml * bv_scale,
        cv_v0_pv_ml=base.cv_v0_pv_ml * bv_scale,
        cv_c_la_ml_mmHg=base.cv_c_la_ml_mmHg * bv_scale,
        cv_c_sa_ml_mmHg=base.cv_c_sa_ml_mmHg * bv_scale,
        cv_c_sv_ml_mmHg=base.cv_c_sv_ml_mmHg * bv_scale,
        cv_c_ra_ml_mmHg=base.cv_c_ra_ml_mmHg * bv_scale,
        cv_c_pa_ml_mmHg=base.cv_c_pa_ml_mmHg * bv_scale,
        cv_c_pv_ml_mmHg=base.cv_c_pv_ml_mmHg * bv_scale,
        hemoglobin_g_dl=latent["hemoglobin_g_dl"],
        baseline_gfr_ml_min=latent["baseline_gfr_ml_min"],
        cv_resting_hr_bpm=latent["cv_resting_hr_bpm"],
        cv_r_systemic_mmHg_s_ml=(
            base.cv_r_systemic_mmHg_s_ml * latent["cv_r_systemic_scale"]
        ),
        cv_lv_emax=base.cv_lv_emax * latent["cv_lv_emax_scale"],
        pulmonary_baseline_vq_log_sd=latent["pulmonary_baseline_vq_log_sd"],
        dalla_insulin_sensitivity_scale=latent["dalla_insulin_sensitivity_scale"],
        dalla_gastric_absorption_scale=latent["dalla_gastric_absorption_scale"],
        respiratory_acidosis_efficiency=latent["respiratory_acidosis_efficiency"],
        pbpk_fraction_unbound=latent["pbpk_fraction_unbound"],
        pbpk_hepatic_clint_l_min=latent["pbpk_hepatic_clint_l_min"],
    )
    return VirtualPatient(patient_id=patient_id, config=cfg, latent=latent)


def sample_virtual_cohort(
    n: int,
    seed: int = 20260817,
    base_config: HumanConfig | None = None,
    specs: tuple[ParameterSpec, ...] = DEFAULT_PARAMETER_SPECS,
    correlated: bool = True,
    correlations: dict[tuple[str, str], float] | None = None,
) -> list[VirtualPatient]:
    design = (
        correlated_latin_hypercube(n, specs=specs, seed=seed, correlations=correlations)
        if correlated
        else latin_hypercube(n, len(specs), seed=seed)
    )
    return [
        virtual_patient_from_unit_row(row, f"VP-{i:04d}", base_config=base_config, specs=specs)
        for i, row in enumerate(design)
    ]
