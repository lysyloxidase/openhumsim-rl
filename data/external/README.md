# External human data

No individual-level human dataset is bundled with OpenHumSim-RL.

The Jaeb Center dataset **Glucose Sensor Profiles in Healthy Non-Diabetic Subjects** is available through the official Jaeb public-dataset page. That page asks the downloader for contact/institution/planned-use information and requires agreement to dataset terms.

OpenHumSim v0.10 therefore does **not** fetch the underlying S3 object directly.

Run:

```bash
uv run openhumsim data jaeb-download-instructions
```

After completing the official download, place the archive here:

```text
data/external/CGMND.zip
```

Then:

```bash
uv run openhumsim data inspect-jaeb-schema data/external/CGMND.zip
uv run openhumsim data extract-jaeb-events data/external/CGMND.zip
uv run openhumsim data evaluate-jaeb-events data/external/CGMND.zip
uv run openhumsim data fit-jaeb-event-model data/external/CGMND.zip --seed 2020
```

The corresponding publications are Shah et al. (JCEM 2019) for normative CGM distributions and DuBose et al. (JDST 2021; first published online 2020) for meal/exercise event analyses.

Important limitation: the diary recorded meal/snack timing but did not capture meal carbohydrate or fat quantity. Any inferred effective carbohydrate is therefore a latent replay parameter, not an observed dietary measurement.
