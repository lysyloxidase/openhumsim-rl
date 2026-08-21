from __future__ import annotations

import json
from pathlib import Path
from openhumsim_rl.external_data import REFERENCE, JAEB_HEALTHY_CGM_URL
from historical_version_guard import require_exact_version

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "external_reference_shah2019_v0.8.json"


def main():
    require_exact_version("0.8.0")
    payload = {
        "version": "0.8.0",
        "dataset": "Jaeb Center Glucose Sensor Profiles in Healthy Non-Diabetic Subjects (2017)",
        "public_dataset_url": JAEB_HEALTHY_CGM_URL,
        "publication": REFERENCE.citation,
        "published_aggregate_metrics": REFERENCE.as_dict(),
        "validation_role": (
            "Independent free-living human CGM distribution reference. The raw archive is intentionally "
            "not bundled and can be downloaded locally with the CLI. These metrics are not protocol-matched "
            "to the OpenHumSim 75-g meal/glucose challenge."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["published_aggregate_metrics"], indent=2))


if __name__ == "__main__":
    main()
