from __future__ import annotations

from dataclasses import dataclass
from math import log10

from .config import HumanConfig


@dataclass(frozen=True)
class AcidBaseDiagnostics:
    ph: float
    bicarbonate_mmol_l: float
    dissolved_co2_mmol_l: float
    carbonate_mmol_l: float
    total_co2_mmol_l: float
    sida_mEq_l: float
    side_mEq_l: float
    strong_ion_gap_mEq_l: float
    albumin_charge_mEq_l: float
    phosphate_charge_mEq_l: float
    charge_balance_residual_mEq_l: float
    henderson_hasselbalch_residual: float
    iterations: int


class PhysicochemicalAcidBaseModel:
    """Reduced Stewart-Figge plasma chemistry with explicit carbonate species.

    The reduced plasma charge equation includes explicit divalent carbonate
    (2*CO3--). It remains a reduced charge closure because Ca/Mg/
    sulfate and other minor ions are not explicit. CO2/HCO3 mass action is the
    plasma closure, but exposes a non-mutating snapshot solver so the whole-blood carbon
    module can solve PaCO2 from a conserved exchangeable-CO2 pool. RBC carbonate
    and carbamino content are included in ``blood_gas.py``; its hemoglobin proton
    term is diagnostic and is not fed back into this plasma pH root. Consequently
    this is not a closed whole-blood proton/charge-balance model.
    """

    def __init__(self, config: HumanConfig):
        self.cfg = config

    @staticmethod
    def albumin_charge_mEq_l(albumin_g_dl: float, ph: float) -> float:
        return max(0.0, 10.0 * albumin_g_dl * (0.123 * ph - 0.631))

    @staticmethod
    def phosphate_charge_mEq_l(phosphate_mmol_l: float, ph: float) -> float:
        return max(0.0, phosphate_mmol_l * (0.309 * ph - 0.469))

    def bicarbonate_from_ph_pco2(self, ph: float, paco2_mmHg: float) -> float:
        c = self.cfg
        dissolved = c.co2_solubility_mmol_l_mmHg * max(1e-12, paco2_mmHg)
        return dissolved * 10.0 ** (ph - c.carbonic_acid_pka)

    def carbonate_from_bicarbonate(self, ph: float, bicarbonate_mmol_l: float) -> float:
        return max(0.0, bicarbonate_mmol_l * 10.0 ** (ph - self.cfg.carbonate_pka2))

    def _weak_acid_concentrations(self, state):
        plasma_l = max(1e-9, float(state.plasma_volume_l))
        ecf_l = max(1e-9, float(state.ecf_volume_l))
        albumin = float(state.plasma_albumin_g / (10.0 * plasma_l))
        phosphate = float(state.ecf_phosphate_mmol / ecf_l)
        uma = float(state.nonvolatile_strong_anion_mEq / ecf_l)
        return albumin, phosphate, uma

    def apparent_sid(self, state, chloride_override: float | None = None) -> float:
        chloride = state.chloride_mmol_l if chloride_override is None else chloride_override
        return float(state.sodium_mmol_l + state.potassium_mmol_l - chloride - state.lactate_mmol_l)

    def snapshot_for_pco2(
        self, state, paco2_mmHg: float, *, chloride_override: float | None = None
    ) -> AcidBaseDiagnostics:
        c = self.cfg
        albumin, phosphate, uma = self._weak_acid_concentrations(state)
        sida = self.apparent_sid(state, chloride_override=chloride_override)
        effective_strong = sida - uma

        def residual(ph: float) -> float:
            hco3 = self.bicarbonate_from_ph_pco2(ph, paco2_mmHg)
            co3 = self.carbonate_from_bicarbonate(ph, hco3)
            alb = self.albumin_charge_mEq_l(albumin, ph)
            pi = self.phosphate_charge_mEq_l(phosphate, ph)
            # Carbonate is divalent and contributes 2 equivalents per mmol.
            return float(effective_strong - (hco3 + 2.0 * co3 + alb + pi))

        lo, hi = 6.50, 8.00
        flo, fhi = residual(lo), residual(hi)
        iterations = 0
        if flo == 0.0:
            ph = lo
        elif fhi == 0.0:
            ph = hi
        elif flo * fhi > 0.0:
            ph = lo if abs(flo) < abs(fhi) else hi
        else:
            for iterations in range(1, c.acid_base_max_iterations + 1):
                mid = 0.5 * (lo + hi)
                fm = residual(mid)
                if abs(fm) <= c.acid_base_charge_tolerance_mEq_l:
                    lo = hi = mid
                    break
                if flo * fm <= 0.0:
                    hi, fhi = mid, fm
                else:
                    lo, flo = mid, fm
            ph = 0.5 * (lo + hi)

        hco3 = self.bicarbonate_from_ph_pco2(ph, paco2_mmHg)
        dissolved = c.co2_solubility_mmol_l_mmHg * paco2_mmHg
        carbonate = self.carbonate_from_bicarbonate(ph, hco3)
        alb = self.albumin_charge_mEq_l(albumin, ph)
        pi = self.phosphate_charge_mEq_l(phosphate, ph)
        side = hco3 + 2.0 * carbonate + alb + pi
        sig = sida - side
        charge_residual = sida - uma - side
        hh_ph = c.carbonic_acid_pka + log10(max(1e-12, hco3) / max(1e-12, dissolved))

        return AcidBaseDiagnostics(
            ph=float(ph), bicarbonate_mmol_l=float(hco3),
            dissolved_co2_mmol_l=float(dissolved), carbonate_mmol_l=float(carbonate),
            total_co2_mmol_l=float(hco3 + dissolved + carbonate),
            sida_mEq_l=float(sida), side_mEq_l=float(side),
            strong_ion_gap_mEq_l=float(sig), albumin_charge_mEq_l=float(alb),
            phosphate_charge_mEq_l=float(pi),
            charge_balance_residual_mEq_l=float(charge_residual),
            henderson_hasselbalch_residual=float(ph - hh_ph), iterations=int(iterations),
        )

    def _apply_snapshot(self, state, d: AcidBaseDiagnostics):
        albumin, phosphate, uma = self._weak_acid_concentrations(state)
        state.albumin_g_dl = float(albumin)
        state.phosphate_mmol_l = float(phosphate)
        state.nonvolatile_strong_anion_mEq_l = float(uma)
        state.ph_arterial = d.ph
        state.bicarbonate_mmol_l = d.bicarbonate_mmol_l
        state.dissolved_co2_mmol_l = d.dissolved_co2_mmol_l
        state.carbonate_mmol_l = d.carbonate_mmol_l
        state.total_co2_mmol_l = d.total_co2_mmol_l
        state.strong_ion_difference_apparent_mEq_l = d.sida_mEq_l
        state.strong_ion_difference_effective_mEq_l = d.side_mEq_l
        state.strong_ion_gap_mEq_l = d.strong_ion_gap_mEq_l
        state.albumin_charge_mEq_l = d.albumin_charge_mEq_l
        state.phosphate_charge_mEq_l = d.phosphate_charge_mEq_l
        state.charge_balance_residual_mEq_l = d.charge_balance_residual_mEq_l
        state.henderson_hasselbalch_residual = d.henderson_hasselbalch_residual
        state.anion_gap_mEq_l = float(state.sodium_mmol_l - state.chloride_mmol_l - d.bicarbonate_mmol_l)
        state.albumin_corrected_anion_gap_mEq_l = float(
            state.anion_gap_mEq_l + 2.5 * (self.cfg.baseline_albumin_g_dl - albumin)
        )
        state.acid_base_solver_iterations = float(d.iterations)
        return state

    def initialize_state(self, state):
        c = self.cfg
        state.plasma_albumin_g = c.baseline_albumin_g_dl * max(1e-9, state.plasma_volume_l) * 10.0
        state.ecf_phosphate_mmol = c.baseline_phosphate_mmol_l * max(1e-9, state.ecf_volume_l)
        # Baseline UMA is calibrated so the nominal state closes at reference pH.
        albumin, phosphate, _ = self._weak_acid_concentrations(state)
        ph0 = c.baseline_ph_arterial
        hco3 = self.bicarbonate_from_ph_pco2(ph0, state.paco2_mmHg)
        alb = self.albumin_charge_mEq_l(albumin, ph0)
        pi = self.phosphate_charge_mEq_l(phosphate, ph0)
        sida = self.apparent_sid(state)
        co3 = self.carbonate_from_bicarbonate(ph0, hco3)
        uma0 = max(0.0, sida - (hco3 + 2.0 * co3 + alb + pi))
        state.nonvolatile_strong_anion_mEq = uma0 * max(1e-9, state.ecf_volume_l)
        state.initial_nonvolatile_strong_anion_mEq = state.nonvolatile_strong_anion_mEq
        state.nonvolatile_acid_generated_mEq = 0.0
        state.nonvolatile_acid_excreted_mEq = 0.0
        return self.solve(state)

    def solve(self, state) -> AcidBaseDiagnostics:
        d = self.snapshot_for_pco2(state, state.paco2_mmHg)
        self._apply_snapshot(state, d)
        return d
