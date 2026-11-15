# Logistics ETA Prediction & Delay Intelligence

An end-to-end Machine Learning solution that predicts shipment Estimated Time of Arrival (ETA), identifies potential delivery delays, and provides operational insights through an interactive dashboard.

The system combines carrier tracking events, route information, weather conditions, and facility queue data to estimate delivery times and highlight delay risks.

## Key Features

- Synthetic logistics data generation with reproducible results.
- Data validation and relationship integrity checks.
- Shipment ETA prediction using Machine Learning.
- Delivery delay-risk classification.
- Tracking event analysis and shipment-level search.
- What-if ETA prediction based on operational conditions.
- REST API using FastAPI.
- Interactive operational dashboard using Streamlit.
- Model evaluation metrics and data checksums.
- Automated tests using pytest.

## Technology Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Backend:** FastAPI, Uvicorn
- **Dashboard:** Streamlit
- **Database:** SQLite
- **Testing:** Pytest
- **Model Storage:** Joblib

## Project Architecture

```text
logistics-eta-prediction/
│
├── app/
│   ├── api.py
│   └── dashboard.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── eta_model.joblib
│   └── delay_model.joblib
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── generate.py
│   ├── validate.py
│   ├── model.py
│   └── run_pipeline.py
│
├── tests/
│   └── test_project.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```powershell
git clone https://github.com/UTTAM130/logistics-eta-prediction.git
```

```powershell
cd logistics-eta-prediction
```

### 2. Create a virtual environment

```powershell
py -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
```

```powershell
pip install -r requirements.txt
```

## Run the Complete ML Pipeline

Generate synthetic data, validate datasets, train models, evaluate performance, and create processed outputs:

```powershell
python -m src.run_pipeline --shipments 5000 --seed 20261022
```

For a larger synthetic dataset:

```powershell
python -m src.run_pipeline --shipments 50000 --seed 20261022
```

## Run the Application

### Start FastAPI Backend

Open the first terminal:

```powershell
cd C:\CandidateWorkspace\logistics-eta-prediction
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
uvicorn app.api:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

### Start Streamlit Dashboard

Open a second terminal:

```powershell
cd C:\CandidateWorkspace\logistics-eta-prediction
```

```powershell
.\.venv\Scripts\Activate.ps1
```

Set the Python import path:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

Start the dashboard:

```powershell
python -m streamlit run app/dashboard.py
```

Dashboard URL:

http://localhost:8501

## Model Performance

Results from the 5,000-shipment synthetic dataset:

### ETA Prediction Model

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 3.87 hours |
| Root Mean Squared Error (RMSE) | 5.01 hours |
| R² Score | 0.9081 |

### Delay Classification Model

| Metric | Result |
|---|---:|
| Accuracy | 99.10% |
| Precision | 99.10% |
| Recall | 100.00% |
| F1 Score | 99.55% |
| ROC-AUC | 0.7617 |

**Note:** These results are from synthetic demonstration data. Accuracy should not be interpreted alone; ROC-AUC indicates more moderate class-discrimination performance.

## Model Input Features

The prediction models use the following operational features:

| Feature | Description |
|---|---|
| planned_remaining_hours | Planned time remaining until delivery |
| queue_pressure | Facility queue pressure |
| weather_severity | Weather severity level |
| carrier_reliability | Carrier reliability score |

The ETA model predicts remaining delivery time, while the delay classifier estimates delivery delay risk.

## Data Outputs

| Directory / File | Description |
|---|---|
| `data/raw/` | Generated logistics source datasets |
| `data/processed/eta_snapshots.csv` | Point-in-time model dataset |
| `data/processed/ground_truth.csv` | Labelled exception examples |
| `data/processed/validation_summary.json` | Data validation results |
| `data/processed/metrics.json` | Model evaluation metrics |
| `data/processed/checksums.json` | SHA-256 checksums |
| `models/` | Trained Machine Learning models |

## Data Model Relationships

- Carrier → Shipments (one-to-many)
- Shipment → Shipment Legs (one-to-many)
- Route → Shipment Legs (one-to-many)
- Facility → Facility Queues (one-to-many)
- Shipment → Tracking Events (one-to-many)
- Shipment → ETA Estimates (one-to-many)
- Route ↔ Weather Events (many-to-many through RouteWeather)
- Service Level → Shipments (one-to-many)

Foreign-key relationships and timestamp integrity are checked by the validation pipeline.

## Reproducibility and Data Quality

The synthetic data generator uses a configurable random seed to make experiments reproducible.

The validation pipeline checks data relationships, identifiers, timestamps, and other data-quality conditions.

Generated validation reports and SHA-256 checksums help preserve traceability.

## Run Tests

Execute the project tests:

```powershell
python -m pytest -q
```

## Business Value

This solution demonstrates how logistics organizations can use Machine Learning to:

- Improve shipment ETA visibility.
- Identify potentially delayed shipments.
- Monitor operational bottlenecks.
- Understand the impact of weather and facility queues.
- Support proactive exception management.
- Improve logistics planning and decision-making.

## Project Limitations

This is a synthetic-data hackathon demonstration. It does not currently represent a live carrier integration or a production-validated ETA service. Model performance on real-world logistics data may differ.

## Author

**Uttam Rangisetti**

GitHub: https://github.com/UTTAM130

---

**Project:** Logistics ETA Prediction & Delay Intelligence  
**Category:** Machine Learning | Data Science | Supply Chain Analytics
