PY=python

install:
	$(PY) -m pip install -r requirements.txt

train:
	$(PY) Scripts/train_xgb.py

analyze:
	$(PY) Scripts/error_analysis.py


retrain:
	$(PY) Scripts/retrain.py

api:
	uvicorn api.main:app --reload --port 8000

drift:
	$(PY) Scripts/monitor_drift_simple.py

format:
	ruff check --fix .
	ruff format .
	black .

lint:
	ruff check .
