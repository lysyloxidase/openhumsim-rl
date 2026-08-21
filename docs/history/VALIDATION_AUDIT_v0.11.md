# OpenHumSim-RL v0.11 — Physiologic / biochemical / physical / medical credibility audit

## Executive status

v0.11 is the **physicochemical acid–base release**. The key change is conceptual: arterial pH and bicarbonate are no longer independently tuned state variables. They are solved as dependent variables from a reduced Stewart–Figge charge balance coupled to Henderson–Hasselbalch CO2 chemistry.

This materially improves biochemical and physical credibility, especially for chloride loading, lactate accumulation, respiratory acidosis and renal acid handling. It does **not** make the simulator clinically validated.

## New acid–base closure

The implemented reduced plasma system uses:

- dynamic Na+, K+, Cl− and lactate−;
- PaCO2 from the respiratory model;
- conserved/dilutable plasma albumin mass;
- conserved/dilutable extracellular phosphate pool;
- an explicit nonvolatile strong-anion burden;
- renal NH4+, titratable-acid and bicarbonaturia readouts;
- numerical electrical-neutrality closure.

The solver enforces, in reduced form:

```text
SIDa − unmeasured strong anion burden
  = HCO3− + albumin charge + phosphate charge
```

with:

```text
HCO3− = alpha_CO2 * PaCO2 * 10^(pH − pKa)
```

and Figge-style pH-dependent albumin/phosphate charges.

### Important interpretation

`strong_ion_gap_mEq_l` is not a measured molecule. It is the residual between represented apparent and effective strong-ion difference. In v0.11 its baseline amount is calibrated to close the nominal plasma charge balance and then evolves through the explicit nonvolatile-acid ledger.

## External human benchmark: 0.9% saline

Primary external benchmark: Dell'Anna et al., 2025 randomized physiological trial in humans.

Published protocol/result relevant to this model:

- normal saline chloride: 154 mEq/L;
- saline SID: 0 mEq/L;
- total study dose: 30 mL/kg (10 + 20 mL/kg);
- median pH one hour after the second saline bolus: 7.34 [7.32–7.36];
- the reported mechanism was increased chloride / reduced SID with limited renal response.

OpenHumSim v0.11, nominal 70-kg challenge, seed 1:

```text
baseline pH                 7.4000
30 mL/kg saline pH          7.3423
baseline chloride           102.22 mmol/L
post-challenge chloride     108.96 mmol/L
baseline SIDa                41.60 mEq/L
post-challenge SIDa          36.05 mEq/L
albumin after dilution        3.65 g/dL
```

The pH magnitude is close to the published median, but the model collapses the two-bolus/timing protocol into an instantaneous distribution challenge. This is therefore an **external dose/composition response benchmark**, not a validated reproduction of the study time course.

## Lactate chemistry

The transient lactate challenge no longer manually sets pH or HCO3−. Raising lactate directly lowers represented SID and the coupled charge solver lowers pH.

Nominal seed-1 initial response:

```text
baseline lactate             ~0.99 mmol/L
challenge lactate             4.00 mmol/L
baseline pH                   7.400
challenge pH                  7.350
```

This is mechanistically preferable to hard-coding `HCO3 = 15` and `pH = 7.22`, but lactate production/clearance itself remains reduced-order.

## Respiratory–renal coupling

Respiratory mechanics controls PaCO2. The acid–base solver then derives pH/HCO3−. Renal acid handling responds on subsequent steps through:

```text
NH4+ excretion
+ titratable acid
− urinary HCO3−
= net acid excretion
```

Additional ammonium excretion is represented predominantly with chloride loss, allowing renal compensation to change strong-ion balance instead of editing bicarbonate directly.

For a persistent respiratory-acidosis challenge, v0.11 shows:

- acute fall in pH;
- increased urinary ammonium;
- progressive increase in HCO3−/pH over time;
- chloride/SID changes contributing to compensation.

This is a whole-kidney reduced model, not a nephron transport model.

## Renal-function credibility

Reduced renal function lowers acid excretion and increases the explicit nonvolatile strong-anion burden / SIG over time. The acid ledger closes numerically:

```text
current acid burden
= initial burden
+ generated nonvolatile acid
− renally excreted acid
```

The validation suite requires the residual to remain below `1e-8 mEq`.

## Numerical / physical verification

v0.11 scientific checks:

```text
9 / 9 PASS
0 FAIL
```

They cover:

1. frozen v0.10 regression status;
2. release version;
3. charge + Henderson–Hasselbalch closure;
4. external human saline benchmark;
5. lactate/SID/pH direction;
6. respiratory-acidosis renal compensation;
7. reduced-renal-function acid retention;
8. acid-base timestep convergence;
9. 10-seed random-intervention residual stress test.

Core tests run separately:

```text
tests/test_env.py          37 / 37 PASS
tests/test_event_replay.py  6 / 6 PASS
Gymnasium checker           SKIP (package unavailable in validation environment)
```

A combined full-suite invocation can exceed the available release-test runtime because several legacy scientific tests are computationally heavy; the modules above pass separately.

## What is now physically constrained

### Stronger

- Na/K/Cl/H2O mass ledgers;
- PBPK drug mass ledger;
- closed-loop blood-volume conservation;
- GI glucose mass balance;
- Fick O2 identity;
- plasma acid-base charge closure for represented species;
- Henderson–Hasselbalch consistency;
- nonvolatile-acid mass ledger.

### Still incomplete

The model does **not** yet conserve full whole-body inorganic carbon. PaCO2 is generated by respiratory gas-exchange dynamics rather than a conserved CO2/HCO3/carbonate body pool.

## What is not yet biochemically complete

Missing or reduced:

- dynamic ionized Ca2+ and Mg2+ pools;
- full carbonate species (including CO3--);
- hemoglobin proton buffering;
- erythrocyte chloride shift;
- 2,3-DPG / temperature-dependent acid–base/O2 coupling;
- full plasma protein chemistry beyond albumin approximation;
- urine electroneutrality;
- explicit glutamine ammoniagenesis;
- phosphate-buffer transport by nephron segments;
- Na+/H+ exchanger, H+-ATPase and H+/K+-ATPase transport;
- hepatic urea/glutamine acid-base coupling.

## Medical credibility classification

| Component | v0.11 classification |
|---|---|
| Dalla Man glucose/insulin core | literature-derived mechanistic core; external free-living calibration remains partial |
| 0D cardiovascular | mechanistic reduced-order; not patient-specific validated |
| Respiratory gas exchange | mechanistic reduced-order; no V/Q distribution or shunt |
| O2 transport | mass-flow consistent Fick diagnostics; simplified Hb chemistry |
| Renal water/electrolytes | mass-conserving reduced-order |
| Renal acid handling | improved mechanistic decomposition; not nephron-resolved |
| Acid-base chemistry | charge- and mass-action-constrained reduced Stewart–Figge model |
| PBPK | mass-conserving generic probe; not a validated drug |
| RL policies | research benchmark only; no therapeutic validity |
| Clinical decision use | **NOT SUPPORTED** |

## Primary scientific sources used for v0.11 design

- Stewart PA. *Modern quantitative acid-base chemistry*. Can J Physiol Pharmacol. 1983;61:1444–1461. PMID 6423247.
- Figge J, Rossing TH, Fencl V. *The role of serum proteins in acid-base equilibria*. J Lab Clin Med. 1991;117:453–467. PMID 2045713.
- Figge J, Mydosh T, Fencl V. *Serum proteins and acid-base equilibria: a follow-up*. J Lab Clin Med. 1992;120:713–719. PMID 1431499.
- Dell'Anna AM et al. *Stewart's theory and acid-base changes induced by crystalloid infusion in humans: a randomized physiological trial*. Ann Intensive Care. 2025;15:54. PMID 40263186.

## Recommended future work

Do **not** add another organ next. The highest-value v0.12 step is one of:

1. full CO2/HCO3/carbonate + erythrocyte/hemoglobin buffering and chloride shift, then external blood-gas validation; or
2. creatinine generation/clearance + urine-output time windows for clinically grounded kidney-state validation.

For overall physiologic/biochemical credibility, option 1 is the more direct continuation of v0.11.
