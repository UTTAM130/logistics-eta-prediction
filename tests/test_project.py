import pandas as pd

from src.generate import generate_data
from src.validate import validate_tables


def test_generator_and_relationships(tmp_path):
    tables, eta, truth = generate_data(200, 20261022)

    assert len(tables["shipment"]) == 200
    assert len(tables["tracking_event"]) == 1200

    assert tables["shipment"].shipment_id.is_unique
    assert tables["tracking_event"].event_id.is_unique

    assert set(tables["tracking_event"].shipment_id).issubset(
        set(tables["shipment"].shipment_id)
    )

    # Validate the actual ground-truth schema.
    assert len(truth) == 4
    assert "anomaly_type" in truth.columns

    assert set(truth.anomaly_type) == {
        "INGESTION_EVENT_TIME_CONFUSION",
        "HIGH_FACILITY_QUEUE",
        "SEVERE_WEATHER",
        "CARRIER_RELIABILITY_RISK",
    }


def test_baseline_validation():
    generate_data(200, 20261022)

    result = validate_tables()

    assert result["all_passed"], result["checks"]
