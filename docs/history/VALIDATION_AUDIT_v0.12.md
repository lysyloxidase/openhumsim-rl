# OpenHumSim-RL v0.12 — credibility audit

## Scope

v0.12 targets the largest remaining chemistry/physics gap from v0.11: whole-blood CO2 carriage and coupling between plasma acid-base chemistry, erythrocytes, hemoglobin, ventilation and tissue CO2 production.

The release is a **reduced-order mechanistic research model**. It is not a clinical blood-gas analyzer, a validated ventilator model, or a patient-specific digital twin.

## Implemented changes

1. **Conserved rapidly exchangeable CO2 pool**
   - metabolic `VCO2` adds carbon;
   - pulmonary elimination removes carbon;
   - urinary bicarbonate loss removes carbon;
   - PaCO2 is solved from the resulting whole-blood CO2 content instead of being independently assigned in the integrated simulation.

2. **Explicit carbonate species**
   - dissolved CO2;
   - HCO3-;
   - CO3--;
   - separately represented in plasma and RBC water where applicable.

3. **Human erythrocyte Donnan equilibrium**
   - Funder & Wieth (1966) empirical relationships are used for H+ and Cl- distribution between plasma and RBCs near physiological conditions.

4. **Hemoglobin-related chemistry**
   - non-carbonic Hb buffer capacity is explicit;
   - carbamino CO2 is represented;
   - standard carbamino saturation is anchored to 13.1% at standard physiological conditions reported by Dash/Bassingthwaighte;
   - deoxygenation increases CO2 capacity as a reduced Haldane effect.

5. **Arteriovenous coupling**
   - mixed-venous CO2 content is constrained by Fick balance from VCO2/cardiac output;
   - venous PCO2/pH are solved from target venous content;
   - local chloride chemistry distinguishes plasma-to-RBC Hamburger exchange from oxygen-dependent Hb chloride binding;
- total local chloride (plasma + RBC free + Hb-bound increment) is exactly balanced.

## External checks

### O'Neill & Robbins 2017 — total arterial blood CO2

Published mechanistic reference condition:
- PCO2 = 40 mmHg
- PO2 = 100 mmHg
- total blood CO2 = 44.6 mL CO2 / 100 mL blood

OpenHumSim v0.12 standard-state result:
- total blood CO2 = **44.37 mL/100 mL**
- absolute difference = **0.23 mL/100 mL**

This is a scale benchmark. OpenHumSim does not reproduce all molecular equilibria or parameterization of the O'Neill-Robbins model.

### Funder & Wieth 1966 / Kummerow 2000 — RBC acid-base partition

At plasma pH 7.40:
- v0.12 RBC pH = **7.189**
- Kummerow reported human RBC intracellular pH = **7.19 +/- 0.04** at external pH 7.4.

The Funder-Wieth H+/Cl- ratio equations are transcribed directly into the model.

### Dash/Korman/Bassingthwaighte — carbamino anchor

At standard physiological conditions:
- reference Hb O2 saturation = 97.2%
- reference amino-group CO2 saturation = 13.1%
- v0.12 carbamino fraction at the anchor = **0.131 exactly by construction**.

This validates parameter transcription only. The v0.12 single-site reduced binding law is not the full Dash equation set.

### Arteriovenous behavior

Nominal seeded baseline at reset gives approximately:
- PaCO2 44.3 mmHg;
- mixed-venous PCO2 49.8 mmHg;
- arterial pH 7.380;
- mixed-venous pH 7.351;
- plasma chloride-shift diagnostic 1.07 mmol/L;
- RBC free-Cl arterial-to-venous fall 1.34 mmol/L;
- increased venous Hb-bound chloride is tracked separately;
- Fick CO2-content residual ~0.

After 60 min zero-action equilibration, PaCO2 approaches ~40.5 mmHg. The initial seed includes physiological initialization jitter; the standard-state external benchmark is evaluated separately at exact reference conditions.

## Physical and chemical conservation

The v0.12 scientific gate checks:
- exchangeable carbon ledger residual < 1e-8 mmol;
- whole-blood CO2 content root residual < 1e-5 mmol/L;
- arteriovenous Fick CO2 residual < 1e-5 mmol/L;
- local chloride redistribution residual < 1e-10 mmol/L blood;
- plasma electroneutrality residual < 1e-6 mEq/L.

Ten-seed random-intervention stress testing passed these gates without unexpected termination in the tested window.

## Numerical verification

The coupled whole-blood carbon solver is compared at physiology integration steps 0.25 vs 0.125 min. Release criteria are:
- |delta PaCO2| < 0.5 mmHg;
- |delta pH| < 0.005.

Both pass in the baseline convergence test.

## Biological / biochemical credibility classification

### Stronger components

- plasma charge closure and strong-ion accounting;
- explicit HCO3-/CO3--/dissolved CO2 speciation;
- externally anchored whole-blood CO2 content scale;
- human empirical RBC Donnan relationships;
- exact internal carbon, Fick and local chloride ledgers;
- correct qualitative Haldane direction;
- coupling of metabolic VCO2, ventilation and PaCO2.

### Still reduced / heuristic

- carbamino Hb binding uses a one-site effective law, not a full multi-site Hb model;
- Hb proton buffering is a reduced buffer-capacity representation;
- hematocrit and water fractions are fixed reference values in the blood-gas chemistry;
- the exchangeable CO2 volume is an effective capacitance, not a full anatomical body-carbon model;
- chloride shift is equilibrium redistribution, not Band-3 kinetic transport;
- oxygen-dependent Hb chloride binding is an empirical reduced term, not a molecular binding model;
- carbonic anhydrase kinetics are assumed effectively instantaneous.

### Missing before clinical blood-gas prediction is credible

- full RBC/plasma/tissue/interstitial CO2 compartments;
- temperature and 2,3-DPG effects on Hb chemistry;
- explicit Bohr/Haldane multi-site coupling rather than reduced terms;
- dynamic hematocrit/osmotic RBC water shifts;
- pulmonary capillary transit and diffusion;
- V/Q heterogeneity and shunt;
- external validation against raw paired arterial/mixed-venous blood-gas and content datasets across hypercapnia/hypocapnia/exercise states.

## Medical interpretation

The model must **not** be used for:
- ventilator settings;
- diagnosis of acid-base disorders in a patient;
- oxygen prescription;
- drug or insulin dosing;
- clinical triage.

Passing conservation and external scale checks means the simulator is more internally credible. It does not establish patient-level clinical validity.

## Release gates

Executed for v0.12:

- 35 legacy/core short tests: PASS
- 2 long 24-hour renal/acid-base legacy tests: PASS separately
- 6 event/replay tests: PASS
- 8 new whole-blood gas tests: PASS
- 10/10 v0.12 scientific checks: PASS

The combined monolithic test suite can exceed the local execution timeout; release gates were therefore run in modules. The optional Gymnasium checker requires the optional local dependency.

## Primary references

- O'Neill DP, Robbins PA. *A mechanistic physicochemical model of carbon dioxide transport in blood.* Journal of Applied Physiology. 2017;122(2):283-295. PMID 27881667.
- Funder J, Wieth JO. *Chloride and hydrogen ion distribution between human red cells and plasma.* Acta Physiologica Scandinavica. 1966;68:234-245.
- Dash RK, Korman B, Bassingthwaighte JB. *Simple accurate mathematical models of blood HbO2 and HbCO2 dissociation curves at varied physiological conditions.* European Journal of Applied Physiology. 2016;116(1):97-113. PMID 26298270.
- Kummerow D et al. *Variations of intracellular pH in human erythrocytes via K(+)(Na(+))/H(+) exchange under low K(+) conditions.* PMID 10931972.
- Prange HD et al. *Physiological consequences of oxygen-dependent chloride binding to hemoglobin.* Journal of Applied Physiology. 2001;91(1):33-38. PMID 11408410.

## Next highest-value step

v0.13 should prioritize **lung heterogeneity and capillary gas exchange**: multiple V/Q compartments, physiologic shunt, dead-space fractions and capillary transit/diffusion. In parallel, the whole-blood CO2 subsystem should be benchmarked against additional raw experimental CO2 dissociation data before its complexity is expanded further.
