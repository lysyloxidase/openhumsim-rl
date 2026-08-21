from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
import numpy as np

from .config import HumanConfig


@dataclass(frozen=True)
class OxygenBindingSnapshot:
    saturation_fraction: float
    p50_mmHg: float
    effective_standard_po2_mmHg: float
    bohr_shift_factor: float


class OxygenBindingModel:
    """Reduced adult-human Hb-O2 binding with a physiologic Bohr shift.

    The standard curve is the Severinghaus (1979) adult-human equation.  The
    actual curve is shifted using Severinghaus' PO2-dependent Bohr coefficient
    relative to pH 7.40, with a deliberately small independent PCO2 residual
    term so that CO2 can affect affinity beyond its effect on pH without double
    counting most of the carbonic-acid effect.

    This is still reduced-order: temperature, 2,3-DPG, dyshemoglobins and fetal
    hemoglobin are fixed/not represented. Affinity responds consistently to the
    model's pH and PCO2 state.
    """

    def __init__(self, config: HumanConfig):
        self.cfg = config

    @staticmethod
    def standard_severinghaus_saturation(po2_mmHg: float) -> float:
        po2 = max(0.0, float(po2_mmHg))
        den = po2**3 + 150.0 * po2
        if den <= 0.0:
            return 0.0
        return float(np.clip(1.0 / (1.0 + 23400.0 / den), 0.0, 1.0))

    @staticmethod
    def _severinghaus_bohr_lnpo2_per_ph(po2_mmHg: float) -> float:
        # Severinghaus 1979 Eq. 4: d ln(PO2) / d pH at constant saturation.
        po2 = max(1e-6, float(po2_mmHg))
        return float((po2 / 26.6) ** 0.184 - 2.2)

    def bohr_shift_factor(self, po2_mmHg: float, ph: float, pco2_mmHg: float) -> float:
        dph = float(ph) - self.cfg.o2_standard_ph
        coeff = self._severinghaus_bohr_lnpo2_per_ph(po2_mmHg)
        ph_factor = exp(coeff * dph)

        # Small residual direct CO2 effect.  The majority of the CO2 effect is
        # already mediated by the model's pH; keeping this coefficient small
        # avoids double-counting.  2x PCO2 changes P50 by only ~2% here.
        pco2_ratio = max(1e-6, float(pco2_mmHg) / self.cfg.o2_standard_pco2_mmHg)
        co2_factor = exp(self.cfg.o2_direct_co2_log_affinity_gain * log(pco2_ratio))
        return float(np.clip(ph_factor * co2_factor, 0.45, 2.2))

    def saturation(self, po2_mmHg: float, ph: float, pco2_mmHg: float) -> OxygenBindingSnapshot:
        po2 = max(0.0, float(po2_mmHg))
        shift = self.bohr_shift_factor(max(po2, 1.0), ph, pco2_mmHg)
        effective = po2 / max(1e-9, shift)
        sat = self.standard_severinghaus_saturation(effective)
        # P50 for this local condition, evaluated from the standard 26.6 mmHg anchor.
        p50_shift = self.bohr_shift_factor(26.6, ph, pco2_mmHg)
        return OxygenBindingSnapshot(
            saturation_fraction=float(sat),
            p50_mmHg=float(26.6 * p50_shift),
            effective_standard_po2_mmHg=float(effective),
            bohr_shift_factor=float(shift),
        )

    def content_ml_dl(
        self,
        po2_mmHg: float,
        ph: float,
        pco2_mmHg: float,
        hemoglobin_g_dl: float,
    ) -> float:
        sat = self.saturation(po2_mmHg, ph, pco2_mmHg).saturation_fraction
        c = self.cfg
        return float(
            c.hemoglobin_o2_capacity_ml_g * max(0.0, hemoglobin_g_dl) * sat
            + c.dissolved_o2_coeff_ml_dl_mmHg * max(0.0, po2_mmHg)
        )

    def po2_from_content(
        self,
        content_ml_dl: float,
        ph: float,
        pco2_mmHg: float,
        hemoglobin_g_dl: float,
    ) -> float:
        target = max(0.0, float(content_ml_dl))
        lo, hi = 0.0, 700.0
        for _ in range(42):
            mid = 0.5 * (lo + hi)
            if self.content_ml_dl(mid, ph, pco2_mmHg, hemoglobin_g_dl) < target:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))
