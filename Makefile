.PHONY: setup doctor demo test validate data events

setup:
	uv python install 3.12
	uv sync --extra local

doctor:
	uv run openhumsim doctor

demo:
	uv run openhumsim demo --scenario oral_glucose_75g --minutes 180 --seed 42

test:
	uv run pytest -q -ra

validate:
	uv run openhumsim validate

data:
	uv run openhumsim data jaeb-download-instructions

events:
	uv run openhumsim data inspect-jaeb-schema data/external/CGMND.zip
	uv run openhumsim data evaluate-jaeb-events data/external/CGMND.zip
	uv run openhumsim data fit-jaeb-event-model data/external/CGMND.zip --seed 2020
