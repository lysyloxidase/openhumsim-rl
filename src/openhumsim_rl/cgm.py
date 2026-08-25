from __future__ import annotations

from dataclasses import dataclass, asdict
from math import exp, isfinite
import numpy as np


@dataclass(frozen=True)
class CGMObservationConfig:
    """Reduced-order blood-to-interstitial CGM observation model.

    ``lag_tau_min`` is an effective first-order equilibration time constant.  The
    default (6 min) is centered on the 5.3--6.2 min physiological lag reported by
    Basu et al. (Diabetes 2013;62:4083-4087) after accounting for measurement-system
    delay.  It is an observation-model parameter, not part of the Dalla Man core.

    Noise is disabled by default so scientific regression tests remain deterministic.
    When enabled, ``relative_noise_sd`` is a Gaussian relative SD engineering model;
    it is *not* equivalent to MARD and must not be interpreted as a Dexcom accuracy
    specification.
    """

    lag_tau_min: float = 6.0
    additive_bias_mg_dl: float = 0.0
    relative_noise_sd: float = 0.0
    lower_reportable_mg_dl: float = 40.0
    upper_reportable_mg_dl: float = 400.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CGMObservationState:
    interstitial_glucose_mg_dl: float = 95.0
    sensor_glucose_mg_dl: float = 95.0


class CGMObservationModel:
    """First-order interstitial compartment plus optional sensor noise."""

    def __init__(self, config: CGMObservationConfig | None = None):
        self.config = config or CGMObservationConfig()
        if not isfinite(self.config.lag_tau_min) or self.config.lag_tau_min <= 0.0:
            raise ValueError("lag_tau_min must be finite and positive")
        if not isfinite(self.config.relative_noise_sd) or self.config.relative_noise_sd < 0.0:
            raise ValueError("relative_noise_sd must be finite and nonnegative")
        if not isfinite(self.config.additive_bias_mg_dl):
            raise ValueError("additive_bias_mg_dl must be finite")
        if (
            not isfinite(self.config.lower_reportable_mg_dl)
            or not isfinite(self.config.upper_reportable_mg_dl)
            or self.config.lower_reportable_mg_dl > self.config.upper_reportable_mg_dl
        ):
            raise ValueError("CGM reportable bounds must be finite and ordered")

    def initialize(
        self,
        blood_glucose_mg_dl: float,
        rng: np.random.Generator | None = None,
    ) -> CGMObservationState:
        g = float(blood_glucose_mg_dl)
        if not isfinite(g):
            raise ValueError("blood_glucose_mg_dl must be finite")
        return CGMObservationState(g, self._report(g, rng))

    def step(
        self,
        state: CGMObservationState,
        blood_glucose_mg_dl: float,
        dt_min: float,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Advance interstitial glucose and take one sensor reading.

        ``ClinicalMeasurementModel`` uses the two operations separately so the
        physiological lag can evolve at the integration cadence while sensor
        noise and dropout are realized only at the configured CGM cadence.
        Keeping ``step`` as their composition preserves the standalone API.
        """
        self.advance_interstitial(state, blood_glucose_mg_dl, dt_min)
        return self.sample(state, rng=rng)

    def advance_interstitial(
        self,
        state: CGMObservationState,
        blood_glucose_mg_dl: float,
        dt_min: float,
    ) -> float:
        """Advance only the blood-to-interstitial lag compartment."""
        if not isfinite(dt_min) or dt_min < 0.0:
            raise ValueError("dt_min must be finite and nonnegative")
        g_blood = float(blood_glucose_mg_dl)
        if not isfinite(g_blood):
            raise ValueError("blood_glucose_mg_dl must be finite")
        if dt_min > 0.0:
            alpha = 1.0 - exp(-float(dt_min) / self.config.lag_tau_min)
            state.interstitial_glucose_mg_dl += alpha * (
                g_blood - state.interstitial_glucose_mg_dl
            )
        return float(state.interstitial_glucose_mg_dl)

    def sample(
        self,
        state: CGMObservationState,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Realize one noisy, range-limited sensor reading."""
        if not isfinite(state.interstitial_glucose_mg_dl):
            raise ValueError("interstitial_glucose_mg_dl must be finite")
        state.sensor_glucose_mg_dl = self._report(
            state.interstitial_glucose_mg_dl, rng
        )
        return float(state.sensor_glucose_mg_dl)

    def _report(
        self,
        interstitial_glucose_mg_dl: float,
        rng: np.random.Generator | None,
    ) -> float:
        c = self.config
        value = float(interstitial_glucose_mg_dl) + c.additive_bias_mg_dl
        if c.relative_noise_sd > 0.0:
            if rng is None:
                rng = np.random.default_rng()
            value += float(rng.normal(0.0, c.relative_noise_sd * max(value, 1.0)))
        return float(np.clip(value, c.lower_reportable_mg_dl, c.upper_reportable_mg_dl))


def blood_to_cgm_trace(
    blood_glucose_mg_dl: np.ndarray | list[float],
    dt_min: float,
    config: CGMObservationConfig | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Transform a blood-glucose trace to a CGM-like interstitial trace."""
    values = np.asarray(blood_glucose_mg_dl, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("blood_glucose_mg_dl must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("blood_glucose_mg_dl must contain only finite values")
    if not np.isfinite(dt_min) or dt_min < 0.0:
        raise ValueError("dt_min must be finite and nonnegative")
    model = CGMObservationModel(config)
    rng = np.random.default_rng(seed)
    state = model.initialize(float(values[0]), rng=rng)
    out = np.empty_like(values, dtype=float)
    out[0] = state.sensor_glucose_mg_dl
    for i in range(1, len(values)):
        out[i] = model.step(state, float(values[i]), dt_min, rng=rng)
    return out
