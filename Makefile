.PHONY: setup data train dashboard validate sample api-smoke reproduce

PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) src/fetch_data.py

train: data
	$(PYTHON) src/train.py

dashboard:
	$(PYTHON) src/build_dashboard.py

validate:
	$(PYTHON) src/validate_outputs.py

sample:
	$(PYTHON) src/make_sample_input.py

api-smoke: sample
	$(PYTHON) src/smoke_test_api.py

reproduce: setup train validate api-smoke
