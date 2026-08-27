# OpenHumSim-RL v0.23.1 validation audit

## Decision

The focused v0.23.1 integrity gate passed 11/11 top-level checks on CPython
3.12.13, executing 103 regression cases. The full local repository suite
passed 315/315 tests with warnings treated as errors. A wheel and source
distribution passed `twine check`; the isolated wheel passed version,
diagnostic and baseline-simulation smoke tests outside the source checkout.

This is internal software verification and mechanistic consistency evidence.
It is not independent external clinical validation.

## Contract

- Package release: `0.23.1`.
- Persisted state schema: `0.22`.
- Debug reward: `latent_research_v0.23`.
- Benchmark reward: `observable_benchmark_v0.23`.
- Clinical/full observation widths: 54/138.
- Action width: 8.
- Historical policy compatibility is not inferred from matching tensor shapes.

## Focused evidence

`validation/run_validation_v23.py` writes
`validation/validation_results_v0.23.1.json` only after all checks execute. The
candidate record was produced on 2026-08-27 with source stability verified
before and after the gate. The final release manifest records the exact result
digest, source fingerprint, commit and CI run.

The focused gate covers:

1. exact package, state-schema and reward-profile contracts;
2. FiO2 handling, configured regional carbonate chemistry, pulmonary-unit
   configuration and terminal-reset behavior;
3. solver and configuration hardening;
4. respiratory-mechanics continuity and total-PEEP semantics;
5. temporal reward, measurement, RNG and rollback behavior;
6. transactional step failures;
7. policy-manifest and checkpoint integrity;
8. atomic environment snapshots, including measurement schedules and counters;
9. strict observation history and deterministic baseline harnesses;
10. source-fingerprint stability throughout execution.

## Biological and physical interpretation

The gas-exchange model now distinguishes legal hypoxic inspired fractions
instead of silently mapping them to 0.15. Regional Henderson-Hasselbalch
calculations use the same configurable pKa and CO2 solubility as the systemic
chemistry contract. These changes improve internal consistency but do not make
the reduced lung model patient-specific or device-calibrated.

## Medical interpretation

Scenario names and dashboard values describe synthetic model states. The
dashboard retains explicit research-only boundaries and keeps measurements
separate from latent mechanistic diagnostics. No protocol-matched cohort,
prospective controller study, alarm validation or treatment-effect evidence is
bundled.

## Algorithmic and numerical interpretation

Snapshot restoration now fails closed when panel members disagree on collection
or result timing, or when event counters cannot match the sampling grid.
Terminal initial states cannot enter the normal step lifecycle. These checks
protect determinism and runtime integrity; they do not establish policy quality
or safety.

## Remaining limitations

1. The simulator is reduced-order and retains empirical control laws, bounded
   engineering parameters and operator splitting.
2. Energy, respiratory-mechanics and PBPK subsystems are not tissue-, device-
   or compound-calibrated clinical models.
3. The clinical interface is a POMDP; the debug vector is not claimed to be a
   complete Markov state.
4. Reward functions can omit real-world harms and remain vulnerable to policy
   exploitation.
5. Virtual-subject variability is an engineering prior, not an identified
   clinical population or patient posterior.

The authoritative machine-readable record is `RELEASE_v0.23.1.json`.
