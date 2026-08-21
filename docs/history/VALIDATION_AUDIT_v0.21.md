# OpenHumSim-RL v0.21 — corrective integrity audit

## Decision

v0.21 materially improves numerical convergence, conservation ledgers and the
RL benchmark boundary. The corrected implementation passes its focused
regressions, but it is still research software and has not been externally
validated for clinical use.

## Corrected high-severity defects

1. **Cycle-window aliasing.** VT/work and cardiovascular summary variables were
   previously calculated from the most recent outer-step window. When that
   window was shorter than a breath or heartbeat, results changed sharply with
   `integration_step_min`. Persistent completed-cycle accumulators now decouple
   the diagnostics from outer-step phase.
2. **Cardiovascular explicit-step instability.** Low-resistance valves create
   time constants shorter than legacy 20–40 ms settings. The solver now derives
   a conservative internal cap from the fastest RC mode.
3. **HPV time counted inside fixed point.** Regional HPV kinetics could advance
   repeatedly during one algebraic solve. Fixed-point iterations now predict the
   algebraic target and commit one kinetic update.
4. **Renal acid-base double representation.** The alkalinizing effect of net
   acid excretion affected both UMA and chloride. The acid, carbonate and
   electrolyte ledgers are now separate and their signs are explicit.
5. **PBPK hepatic mass source.** Hepatic clearance was reported without removing
   the corresponding drug mass from the liver. Elimination is now withdrawn
   from that compartment and the total mass balance closes.
6. **Benchmark `info` leakage.** The blacklist omitted `blood_gas_carbon`, while
   reward terms and scenario metadata exposed hidden truth. A positive allowlist
   now exposes only static contract metadata and an optional terminal reason.
7. **Clock and safety semantics.** A final policy step could overshoot the
   horizon, and lethal transients inside a five-minute step were invisible.
   Duration is clipped to time-to-go and terminality is checked at integrator
   cadence. Private cardiac/respiratory runtime state participates in rollback,
   and `step()` after termination requires `reset()`.
   Time-to-go is observed explicitly. Instantaneous interventions retain their
   documented per-decision bolus size on a short final transition; only their
   downstream continuous evolution is shortened. Running utility and time-held
   control costs use a fixed reference duration rather than the configured
   number of decisions.
8. **Measurement timing and reproducibility.** One pending result per group lost
   samples when delay exceeded cadence; A–a/Enghoff used live hidden values; CGM
   initialization bypassed its seeded RNG. These now use queued endpoint
   samples, consistent timing and a single seeded random stream. A–a and Enghoff
   were removed from the clinical vector because no atomic measured source
   record is modeled yet.
9. **Inactive population dimension and incomplete lock.** The prior varied a
   legacy A–a parameter overwritten by the pulmonary solver, and the lock did
   not cover all split inputs. The prior now varies active V/Q heterogeneity and
   the digest covers dataset fingerprint, both ID sets and seed.
10. **Contradictory oxygen reserve.** The model limited VO2 to an extraction
    fraction of DO2 but reported reserve as total DO2 minus demand. Reserve is
    now relative to extractable delivery, so its sign agrees with supply
    limitation.

## Verification evidence

The v0.21 focused gate contains 13 executable checks covering the strict info
allowlist, exact horizon, outer-step convergence, completed-breath semantics,
renal and PBPK ledgers, oxygen reserve, delayed-result FIFO, CGM seeding, active pulmonary UQ,
split locking, invalid configuration and non-finite state rejection.

Three new regression modules contain 24 tests:

- `tests/test_physics_regressions_v021.py`: 5;
- `tests/test_biomedical_regressions_v021.py`: 9;
- `tests/test_algorithm_regressions_v021.py`: 10.

The gate writes `validation/validation_results_v0.21.json`. Its scope string
explicitly labels the result as focused verification rather than clinical
validation.

The complete repository suite was also run in an isolated Python 3.12
environment with Gymnasium installed and warnings promoted to errors: 145/145
tests passed, including the Gymnasium environment checker.

## Compatibility

- Respiratory and cardiovascular diagnostics are sample-and-hold until a full
  cycle completes. Consumers relying on fragment-derived values will change.
- Mechanical power now includes dynamic source work and therefore is not
  numerically comparable with the former elastic-only surrogate.
- Correct stability capping makes coarse cardiovascular settings slower.
- v0.7/v0.8 neural checkpoints use an older observation width and intervention
  semantics. They are archival and must not be loaded into v0.21 or compared
  with v0.21 returns. Retraining and a versioned evaluation protocol are needed.
- Historical JSON files remain evidence for their named release only. They were
  not overwritten by the v0.21 run.

## Important unresolved limitations

1. **No external clinical validation.** Most checks are invariants, qualitative
   directions or self-consistency tests. There is no bundled independent,
   protocol-matched clinical cohort validating joint trajectories.
2. **Partial observability.** Neither the clinical vector nor the current full
   debug vector contains every dynamic controller, cycle accumulator, pending
   measurement and RNG state. Feed-forward policies therefore do not receive a
   Markov state.
3. **Reduced energy/redox model.** `oxygen_debt_ml_min` is an instantaneous unmet
   O2-demand rate despite its historical name. VCO2 demand is prescribed rather
   than derived from achieved VO2/substrate oxidation, and lactate is a
   reduced concentration kinetic rather than a whole-body mass ledger. A proper
   repair requires a specified ATP/substrate/redox model and calibration, not a
   local algebraic patch.
4. **Acid-base scope.** Hemoglobin proton binding is reported diagnostically but
   is not part of the plasma electroneutrality pH solve. Adding whole-blood
   noncarbonate buffering requires a re-derived charge/capacitance formulation
   and revalidation.
5. **Respiratory-controller scope.** The hypoxic/CO2/pH controller represents a
   bounded awake-adult reflex. Sleep, sedation, CNS injury and mechanistic apnea
   are not modeled.
6. **Inference limits.** The correlated prior remains an engineering prior.
   Importance calibration is small-sample and does not establish identifiability,
   convergence, Monte Carlo error or patient-specific posterior validity.
7. **Sampling resolution.** A measurement cadence finer than the physiology
   integration step cannot reconstruct intermediate truth. The model now records
   such unresolved samples as skipped and takes an honestly timestamped endpoint
   sample; users needing the finer cadence must reduce `integration_step_min`.
8. **Residual operator splitting.** HR targets, vascular resistance and some
   cross-organ inputs are frozen over an outer substep. The large cycle-aliasing
   defect is gone, but small bounded differences remain when the same duration is
   partitioned differently; the v0.21 gate makes those tolerances explicit.

Those limitations are deliberately disclosed rather than hidden behind tuned
thresholds or a synthetic validation label.
