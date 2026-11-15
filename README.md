# Logistics ETA Prediction & Delay Detection

A reproducible synthetic logistics-data project that creates shipment master/transaction data,
validates relationships and timestamps, trains ETA and delay models, exposes predictions through
FastAPI, and displays operational KPIs in Streamlit.

## Stack
Python, Pandas, NumPy, scikit-learn, SQLite, FastAPI, Streamlit, pytest.

## Quick start (Windows PowerShell)
```powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.run_pipeline --shipments 5000 --seed 20261022
```

For the requested full synthetic volume:
```powershell
python -m src.run_pipeline --shipments 50000 --seed 20261022
```

Run the API (in terminal 1):
```powershell
uvicorn app.api:app --reload
```
Open http://127.0.0.1:8000/docs

Run the dashboard (in terminal 2):
```powershell
streamlit run app/dashboard.py
```

Run tests:
```powershell
pytest -q
```

## Reproducibility
The generator uses a fixed NumPy seed. Running twice with the same shipment count and seed
recreates the same CSV contents. `data/processed/validation_summary.json` records row counts
and validation results. `data/processed/checksums.json` contains SHA-256 checksums.

## Outputs
- `data/raw/`: source tables as CSV
- `data/processed/eta_snapshots.csv`: point-in-time model dataset
- `data/processed/ground_truth.csv`: separately stored labelled exception cases
- `data/processed/validation_summary.json`: validation report
- `data/processed/metrics.json`: model evaluation metrics
- `models/`: trained model pipelines

## Important modelling note
The baseline prediction features are point-in-time fields known at the prediction timestamp.
Actual delivery time and delay labels are targets only. `ground_truth.csv` is not loaded as a
training feature. This is a synthetic demonstration, not a production carrier integration.

## Data model and relationships
- Carrier 1:N Shipment
- Shipment 1:N ShipmentLeg
- Route 1:N ShipmentLeg (a route can be reused)
- Facility 1:N FacilityQueue
- Shipment 1:N TrackingEvent
- Shipment 1:N ETAEstimate
- WeatherEvent N:M Route through RouteWeather (bridge table)
- Shipment 1:1 ServiceLevel in this simplified dataset
- Shipment 0:1 DelayReason for delayed shipments

All tables have primary identifiers. Foreign keys are checked by the validator.

## Edge-case catalogue
The generator adds four explicitly labelled examples to `ground_truth.csv`:
1. Event ingestion time later than event time by an abnormal amount (late-arriving scan).
2. Impossible event sequence / route order.
3. A deliberately unsafe ETA snapshot marked as having future-event leakage.
4. Delivered shipment whose current-state field is still `IN_TRANSIT`.

These examples are isolated and labelled; the baseline model excludes the anomaly flags and
ground-truth table. The validator reports baseline integrity and anomaly examples separately.
"""

files["src/__init__.py"] = ""
files["src/config.py"] = """
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

TABLES = [
    "carrier", "facility", "route", "service_level", "shipment",
    "shipment_leg", "tracking_event", "facility_queue", "weather_event",
    "route_weather", "eta_estimate", "delay_reason"
]

def ensure_dirs():
    for path in (RAW_DIR, PROCESSED_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)
"""