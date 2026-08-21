from copy import deepcopy

import pytest

from openhumsim_rl.physiology import HumanState, STATE_SCHEMA_VERSION


def test_versioned_human_state_round_trip_is_exact():
    state = HumanState(lactate_mmol_l=4.0, lactate_amount_mmol=168.0)
    payload = state.to_versioned_payload()
    restored = HumanState.from_versioned_payload(payload)

    assert payload["state_schema_version"] == STATE_SCHEMA_VERSION == "0.22"
    assert restored.as_dict() == state.as_dict()


def test_legacy_concentration_only_state_is_rejected_with_migration_reason():
    legacy = {
        "state_schema_version": "0.21",
        "state": {"lactate_mmol_l": 4.0, "total_body_water_l": 30.0},
    }

    with pytest.raises(ValueError, match="explicit migration.*lactate"):
        HumanState.from_versioned_payload(legacy)


def test_current_state_payload_requires_exact_finite_fields():
    payload = HumanState().to_versioned_payload()
    missing = deepcopy(payload)
    del missing["state"]["lactate_amount_mmol"]
    with pytest.raises(ValueError, match="missing=.*lactate_amount_mmol"):
        HumanState.from_versioned_payload(missing)

    nonfinite = deepcopy(payload)
    nonfinite["state"]["lactate_amount_mmol"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        HumanState.from_versioned_payload(nonfinite)
