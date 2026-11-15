from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

START_DATE = pd.Timestamp("2026-01-01T00:00:00Z")
DEFAULT_SEED = 20261022
DEFAULT_SHIPMENTS = 5000

NUM_CARRIERS = 20
NUM_FACILITIES = 100
NUM_ROUTES = 250
NUM_SERVICE_LEVELS = 2
NUM_WEATHER_EVENTS = 500
EVENTS_PER_SHIPMENT = 6

CARRIERS_FILE = "carrier.csv"
FACILITIES_FILE = "facility.csv"
ROUTES_FILE = "route.csv"
SERVICE_LEVELS_FILE = "service_level.csv"
SHIPMENTS_FILE = "shipment.csv"
SHIPMENT_LEGS_FILE = "shipment_leg.csv"
TRACKING_EVENTS_FILE = "tracking_event.csv"
FACILITY_QUEUES_FILE = "facility_queue.csv"
WEATHER_EVENTS_FILE = "weather_event.csv"
ROUTE_WEATHER_FILE = "route_weather.csv"
ETA_ESTIMATES_FILE = "eta_estimate.csv"
DELAY_REASONS_FILE = "delay_reason.csv"


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def random_datetime(
    rng: np.random.Generator,
    start: pd.Timestamp,
    count: int,
    max_days: int,
) -> pd.Series:
    """Generate reproducible UTC timestamps."""
    offsets = rng.integers(
        0,
        max_days * 24 * 60,
        size=count,
    )
    return pd.Series(
        start + pd.to_timedelta(offsets, unit="m")
    )


def iso_utc(values: pd.Series) -> pd.Series:
    """Convert timestamps to consistent ISO-8601 UTC strings."""
    return pd.to_datetime(values, utc=True).map(
        lambda value: value.isoformat(
            timespec="microseconds"
        ).replace("+00:00", "Z")
    )


def write_csv(df: pd.DataFrame, filename: str, directory: Path) -> None:
    """Save a DataFrame as a CSV file."""
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / filename
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(df):,} rows)")


def make_ids(prefix: str, count: int, width: int = 7) -> list[str]:
    return [
        f"{prefix}-{number:0{width}d}"
        for number in range(1, count + 1)
    ]


# ---------------------------------------------------------
# Master data generation
# ---------------------------------------------------------

def generate_master_data(rng: np.random.Generator):
    # Carriers
    carrier_ids = make_ids("CAR", NUM_CARRIERS)

    carriers = pd.DataFrame({
        "carrier_id": carrier_ids,
        "carrier_name": [
            f"Carrier {i:02d}" for i in range(1, NUM_CARRIERS + 1)
        ],
        "reliability_score": np.round(
            rng.uniform(0.72, 0.99, NUM_CARRIERS), 3
        ),
        "on_time_rate": np.round(
            rng.uniform(0.70, 0.98, NUM_CARRIERS), 3
        ),
        "active": True,
    })

    # Facilities
    facility_ids = make_ids("FAC", NUM_FACILITIES)

    cities = [
        "Hyderabad", "Bengaluru", "Chennai", "Mumbai", "Pune",
        "Delhi", "Kolkata", "Ahmedabad", "Jaipur", "Kochi",
        "Vijayawada", "Visakhapatnam", "Nagpur", "Indore",
        "Coimbatore", "Lucknow", "Surat", "Bhopal",
    ]

    states = [
        "Telangana", "Karnataka", "Tamil Nadu", "Maharashtra",
        "Maharashtra", "Delhi", "West Bengal", "Gujarat",
        "Rajasthan", "Kerala", "Andhra Pradesh", "Andhra Pradesh",
        "Maharashtra", "Madhya Pradesh", "Tamil Nadu", "Uttar Pradesh",
        "Gujarat", "Madhya Pradesh",
    ]

    city_indices = rng.integers(0, len(cities), NUM_FACILITIES)

    facilities = pd.DataFrame({
        "facility_id": facility_ids,
        "facility_name": [
            f"{cities[index]} Distribution Center {i:03d}"
            for i, index in enumerate(city_indices, start=1)
        ],
        "city": [cities[index] for index in city_indices],
        "state": [states[index] for index in city_indices],
        "country": "India",
        "facility_type": rng.choice(
            ["WAREHOUSE", "HUB", "DISTRIBUTION_CENTER"],
            NUM_FACILITIES,
        ),
        "capacity": rng.integers(500, 5000, NUM_FACILITIES),
        "active": True,
    })

    # Routes
    route_ids = make_ids("RTE", NUM_ROUTES)
    route_origin_indices = rng.integers(
        0, NUM_FACILITIES, NUM_ROUTES
    )
    route_destination_indices = (
        route_origin_indices
        + rng.integers(1, NUM_FACILITIES, NUM_ROUTES)
    ) % NUM_FACILITIES

    route_distances = rng.integers(50, 2500, NUM_ROUTES)

    routes = pd.DataFrame({
        "route_id": route_ids,
        "origin_facility_id": [
            facility_ids[i] for i in route_origin_indices
        ],
        "destination_facility_id": [
            facility_ids[i] for i in route_destination_indices
        ],
        "distance_km": route_distances,
        "base_transit_hours": np.round(
            route_distances / rng.uniform(35, 65, NUM_ROUTES), 2
        ),
        "route_type": rng.choice(
            ["DOMESTIC", "REGIONAL", "LONG_HAUL"],
            NUM_ROUTES,
        ),
        "active": True,
    })

    # Service levels
    service_levels = pd.DataFrame({
        "service_level_id": ["SL-001", "SL-002"],
        "service_level_name": ["STANDARD", "EXPRESS"],
        "target_transit_hours": [72, 36],
        "priority": [1, 2],
    })

    return (
        carriers,
        facilities,
        routes,
        service_levels,
    )


# ---------------------------------------------------------
# Shipment generation
# ---------------------------------------------------------

def generate_shipments(
    rng: np.random.Generator,
    shipments_count: int,
    carriers: pd.DataFrame,
    facilities: pd.DataFrame,
    routes: pd.DataFrame,
    service_levels: pd.DataFrame,
):
    shipment_ids = make_ids("SHP-2026", shipments_count)

    route_indices = rng.integers(0, len(routes), shipments_count)
    carrier_indices = rng.integers(0, len(carriers), shipments_count)
    service_indices = rng.integers(
        0, len(service_levels), shipments_count
    )

    route_lookup = routes.reset_index(drop=True)
    carrier_lookup = carriers.reset_index(drop=True)
    service_lookup = service_levels.reset_index(drop=True)

    selected_routes = route_lookup.iloc[route_indices].reset_index(drop=True)
    selected_carriers = carrier_lookup.iloc[
        carrier_indices
    ].reset_index(drop=True)
    selected_services = service_lookup.iloc[
        service_indices
    ].reset_index(drop=True)

    created_at = random_datetime(
        rng,
        START_DATE,
        shipments_count,
        max_days=90,
    )

    planned_transit_hours = (
        selected_routes["base_transit_hours"].to_numpy()
        * np.where(
            selected_services["service_level_name"].to_numpy()
            == "EXPRESS",
            0.75,
            1.0,
        )
    )

    planned_transit_hours = np.maximum(
        planned_transit_hours, 2
    )

    # Carrier performance and random operational disruption
    carrier_reliability = selected_carriers[
        "reliability_score"
    ].to_numpy()

    disruption_hours = rng.gamma(
        shape=1.8,
        scale=3.0,
        size=shipments_count,
    )

    reliability_delay = (
        (1.0 - carrier_reliability) * 35
    )

    actual_transit_hours = (
        planned_transit_hours
        + reliability_delay
        + disruption_hours
        + rng.normal(0, 2.5, shipments_count)
    )

    actual_transit_hours = np.maximum(
        actual_transit_hours, 1
    )

    planned_delivery_at = (
        created_at
        + pd.to_timedelta(planned_transit_hours, unit="h")
    )

    actual_delivery_at = (
        created_at
        + pd.to_timedelta(actual_transit_hours, unit="h")
    )

    is_delayed = (
        actual_delivery_at > planned_delivery_at
    ).astype(int)

    # A shipment is marked delivered after its actual delivery time.
    simulation_time = START_DATE + pd.Timedelta(days=180)

    current_status = np.where(
        actual_delivery_at <= simulation_time,
        "DELIVERED",
        "IN_TRANSIT",
    )

    shipments = pd.DataFrame({
        "shipment_id": shipment_ids,
        "carrier_id": selected_carriers["carrier_id"],
        "route_id": selected_routes["route_id"],
        "service_level_id": selected_services["service_level_id"],
        "origin_facility_id": selected_routes["origin_facility_id"],
        "destination_facility_id": selected_routes[
            "destination_facility_id"
        ],
        "created_at": iso_utc(created_at),
        "planned_delivery_at": iso_utc(planned_delivery_at),
        "actual_delivery_at": iso_utc(actual_delivery_at),
        "planned_transit_hours": np.round(
            planned_transit_hours, 2
        ),
        "actual_transit_hours": np.round(
            actual_transit_hours, 2
        ),
        "current_status": current_status,
        "is_delayed": is_delayed,
    })

    shipment_metadata = pd.DataFrame({
        "shipment_id": shipment_ids,
        "created_at": created_at,
        "planned_delivery_at": planned_delivery_at,
        "actual_delivery_at": actual_delivery_at,
        "planned_transit_hours": planned_transit_hours,
        "actual_transit_hours": actual_transit_hours,
        "is_delayed": is_delayed,
        "carrier_reliability": carrier_reliability,
        "carrier_id": selected_carriers["carrier_id"].to_numpy(),
        "route_id": selected_routes["route_id"].to_numpy(),
        "origin_facility_id": selected_routes[
            "origin_facility_id"
        ].to_numpy(),
        "destination_facility_id": selected_routes[
            "destination_facility_id"
        ].to_numpy(),
    })

    return shipments, shipment_metadata


# ---------------------------------------------------------
# Shipment legs
# ---------------------------------------------------------


def generate_shipment_legs(
    rng: np.random.Generator,
    shipment_metadata: pd.DataFrame,
):
    leg_rows = []
    leg_number = 1

    for shipment in shipment_metadata.itertuples(index=False):
        number_of_legs = int(
            rng.choice([1, 2], p=[0.6666, 0.3334])
        )

        created = shipment.created_at
        total_hours = shipment.actual_transit_hours

        for sequence in range(1, number_of_legs + 1):
            fraction_start = (sequence - 1) / number_of_legs
            fraction_end = sequence / number_of_legs

            leg_start = created + pd.Timedelta(
                hours=total_hours * fraction_start
            )
            leg_end = created + pd.Timedelta(
                hours=total_hours * fraction_end
            )

            leg_rows.append({
                "leg_id": f"LEG-{leg_number:08d}",
                "shipment_id": shipment.shipment_id,
                "route_id": shipment.route_id,
                "leg_sequence": sequence,
                "origin_facility_id": shipment.origin_facility_id,
                "destination_facility_id": shipment.destination_facility_id,
                "planned_departure_at": iso_utc(
                    pd.Series([leg_start])
                ).iloc[0],
                "planned_arrival_at": iso_utc(
                    pd.Series([leg_end])
                ).iloc[0],
                "actual_departure_at": iso_utc(
                    pd.Series([leg_start])
                ).iloc[0],
                "actual_arrival_at": iso_utc(
                    pd.Series([leg_end])
                ).iloc[0],
                "leg_status": "COMPLETED",
            })

            leg_number += 1

    return pd.DataFrame(leg_rows)



# ---------------------------------------------------------
# Tracking events
# ---------------------------------------------------------

def generate_tracking_events(
    rng: np.random.Generator,
    shipment_metadata: pd.DataFrame,
):
    event_types = [
        "SHIPMENT_CREATED",
        "PICKED_UP",
        "IN_TRANSIT",
        "ARRIVED_AT_FACILITY",
        "OUT_FOR_DELIVERY",
        "DELIVERED",
    ]

    event_rows = []
    event_number = 1

    for shipment in shipment_metadata.itertuples(index=False):
        created = shipment.created_at
        actual_transit_hours = shipment.actual_transit_hours

        for event_index, event_type in enumerate(event_types):
            fraction = event_index / (len(event_types) - 1)

            event_time = (
                created
                + pd.Timedelta(
                    hours=actual_transit_hours * fraction
                )
            )

            # Positive ingestion lag: ingestion always occurs at or
            # after the operational event timestamp.
            ingestion_lag_minutes = int(
                rng.integers(1, 91)
            )

            ingestion_time = (
                event_time
                + pd.Timedelta(minutes=ingestion_lag_minutes)
            )

            event_rows.append({
                "event_id": f"EVT-{event_number:09d}",
                "shipment_id": shipment.shipment_id,
                "event_type": event_type,
                "event_time": event_time,
                "ingestion_time": ingestion_time,
                "facility_id": (
                    shipment.destination_facility_id
                    if event_index >= 3
                    else shipment.origin_facility_id
                ),
                "event_description": (
                    f"{event_type.replace('_', ' ').title()} "
                    f"for {shipment.shipment_id}"
                ),
            })

            event_number += 1

    events = pd.DataFrame(event_rows)

    # Validate timestamps before saving.
    event_time = pd.to_datetime(
        events["event_time"], utc=True, errors="coerce"
    )
    ingestion_time = pd.to_datetime(
        events["ingestion_time"], utc=True, errors="coerce"
    )

    invalid = (
        event_time.isna()
        | ingestion_time.isna()
        | (ingestion_time < event_time)
    )

    if invalid.any():
        raise ValueError(
            "Invalid tracking events found: ingestion_time "
            "must be greater than or equal to event_time.\n"
            f"{events.loc[invalid].head(10)}"
        )

    events["event_time"] = iso_utc(events["event_time"])
    events["ingestion_time"] = iso_utc(events["ingestion_time"])

    return events.sort_values(
        ["shipment_id", "event_time"]
    ).reset_index(drop=True)


# ---------------------------------------------------------
# Facility queue generation
# ---------------------------------------------------------

def generate_facility_queues(
    rng: np.random.Generator,
    shipment_metadata: pd.DataFrame,
    facilities: pd.DataFrame,
):
    count = len(shipment_metadata)

    facility_ids = facilities["facility_id"].to_numpy()
    selected_facilities = rng.choice(facility_ids, count)

    queue_length = rng.integers(0, 250, count)
    processing_capacity = rng.integers(20, 150, count)

    queue_pressure = np.round(
        queue_length / np.maximum(processing_capacity, 1),
        3,
    )

    snapshot_time = (
        shipment_metadata["created_at"]
        + pd.to_timedelta(1, unit="m")
    )

    queues = pd.DataFrame({
        "queue_id": make_ids("QUE", count),
        "facility_id": selected_facilities,
        "shipment_id": shipment_metadata["shipment_id"].to_numpy(),
        "snapshot_time": iso_utc(snapshot_time),
        "queue_length": queue_length,
        "processing_capacity": processing_capacity,
        "avg_processing_minutes": np.round(
            rng.uniform(10, 180, count), 2
        ),
        "queue_pressure": queue_pressure,
    })

    return queues


# ---------------------------------------------------------
# Weather generation
# ---------------------------------------------------------

def generate_weather_events(
    rng: np.random.Generator,
):
    weather_ids = make_ids("WTH", NUM_WEATHER_EVENTS)

    start_at = random_datetime(
        rng,
        START_DATE,
        NUM_WEATHER_EVENTS,
        max_days=120,
    )

    duration_hours = rng.integers(
        2, 73, NUM_WEATHER_EVENTS
    )

    end_at = (
        start_at
        + pd.to_timedelta(duration_hours, unit="h")
    )

    severity = rng.choice(
        ["LOW", "MODERATE", "HIGH", "SEVERE"],
        NUM_WEATHER_EVENTS,
        p=[0.30, 0.35, 0.25, 0.10],
    )

    severity_score = pd.Series(severity).map({
        "LOW": 1,
        "MODERATE": 2,
        "HIGH": 3,
        "SEVERE": 4,
    }).to_numpy()

    weather = pd.DataFrame({
        "weather_event_id": weather_ids,
        "region": rng.choice(
            [
                "SOUTH", "NORTH", "EAST",
                "WEST", "CENTRAL",
            ],
            NUM_WEATHER_EVENTS,
        ),
        "weather_type": rng.choice(
            [
                "RAIN", "HEAVY_RAIN", "STORM",
                "FOG", "HEATWAVE", "WIND",
            ],
            NUM_WEATHER_EVENTS,
        ),
        "severity": severity,
        "severity_score": severity_score,
        "precipitation_mm": np.round(
            rng.uniform(0, 120, NUM_WEATHER_EVENTS), 2
        ),
        "wind_speed_kph": np.round(
            rng.uniform(0, 100, NUM_WEATHER_EVENTS), 2
        ),
        "temperature_c": np.round(
            rng.uniform(10, 45, NUM_WEATHER_EVENTS), 2
        ),
        "start_at": iso_utc(start_at),
        "end_at": iso_utc(end_at),
    })

    return weather


def generate_route_weather(
    rng: np.random.Generator,
    routes: pd.DataFrame,
    weather_events: pd.DataFrame,
):
    count = min(
        len(routes) * 2,
        len(weather_events),
    )

    route_indices = rng.choice(
        len(routes), count, replace=True
    )
    weather_indices = rng.choice(
        len(weather_events), count, replace=False
    )

    route_weather = pd.DataFrame({
        "route_weather_id": make_ids("RW", count),
        "route_id": routes.iloc[
            route_indices
        ]["route_id"].to_numpy(),
        "weather_event_id": weather_events.iloc[
            weather_indices
        ]["weather_event_id"].to_numpy(),
        "impact_score": np.round(
            rng.uniform(0.1, 1.0, count), 3
        ),
    })

    return route_weather


# ---------------------------------------------------------
# ETA estimates and ML snapshots
# ---------------------------------------------------------

def generate_eta_estimates(
    rng: np.random.Generator,
    shipment_metadata: pd.DataFrame,
    facility_queues: pd.DataFrame,
    weather_events: pd.DataFrame,
):
    count = len(shipment_metadata)

    queue_pressure = facility_queues[
        "queue_pressure"
    ].to_numpy()

    weather_severity = rng.choice(
        [1, 2, 3, 4],
        count,
        p=[0.30, 0.35, 0.25, 0.10],
    )

    carrier_reliability = shipment_metadata[
        "carrier_reliability"
    ].to_numpy()

    planned_remaining_hours = shipment_metadata[
        "planned_transit_hours"
    ].to_numpy()

    # This is a simple baseline estimate, not the trained ML model.
    estimated_extra_hours = (
        queue_pressure * 1.5
        + weather_severity * 0.8
        + (1.0 - carrier_reliability) * 10
    )

    predicted_remaining_hours = (
        planned_remaining_hours
        + estimated_extra_hours
    )

    delay_probability = 1 / (
        1 + np.exp(
            -(
                -2.0
                + queue_pressure * 0.7
                + weather_severity * 0.45
                + (1.0 - carrier_reliability) * 4
            )
        )
    )

    risk_level = np.where(
        delay_probability >= 0.70,
        "HIGH",
        np.where(
            delay_probability >= 0.40,
            "MEDIUM",
            "LOW",
        ),
    )

    estimate_time = (
        shipment_metadata["created_at"]
        + pd.to_timedelta(1, unit="m")
    )

    predicted_delivery_at = (
        estimate_time
        + pd.to_timedelta(
            predicted_remaining_hours,
            unit="h",
        )
    )

    estimates = pd.DataFrame({
        "eta_id": make_ids("ETA", count),
        "shipment_id": shipment_metadata["shipment_id"].to_numpy(),
        "estimate_time": iso_utc(estimate_time),
        "predicted_delivery_at": iso_utc(
            predicted_delivery_at
        ),
        "predicted_remaining_hours": np.round(
            predicted_remaining_hours, 2
        ),
        "delay_probability": np.round(
            delay_probability, 4
        ),
        "risk_level": risk_level,
        "model_version": "baseline-v1",
        "future_event_leakage": 0,
    })

    # Training snapshot. Features are calculated from the shipment's
    # initial operational context. Actual outcome columns are labels.
    snapshots = pd.DataFrame({
        "shipment_id": shipment_metadata["shipment_id"].to_numpy(),
        
        "planned_remaining_hours": np.round(
            planned_remaining_hours, 2
        ),
        
        "queue_pressure": np.round(queue_pressure, 3),
        
        "weather_severity": weather_severity,
        
        "carrier_reliability": np.round(
            carrier_reliability, 3
        ),
        
        "actual_remaining_hours": np.round(
            shipment_metadata["actual_transit_hours"].to_numpy(),
            2,
        ),
        
        "is_delayed": shipment_metadata["is_delayed"].to_numpy(),
        
        "delay_label": shipment_metadata["is_delayed"].to_numpy(),
    })

    return estimates, snapshots


# ---------------------------------------------------------
# Delay reasons
# ---------------------------------------------------------

def generate_delay_reasons(
    rng: np.random.Generator,
    shipment_metadata: pd.DataFrame,
):
    delayed = shipment_metadata[
        shipment_metadata["is_delayed"] == 1
    ].copy()

    reason_options = [
        ("CARRIER_DELAY", "Carrier operational delay"),
        ("WEATHER", "Weather-related disruption"),
        ("FACILITY_CONGESTION", "Facility queue congestion"),
        ("ROUTE_DISRUPTION", "Route disruption"),
        ("CAPACITY_SHORTAGE", "Transport capacity shortage"),
    ]

    rows = []

    for index, shipment in enumerate(
        delayed.itertuples(index=False),
        start=1,
    ):
        reason_index = int(
            rng.integers(0, len(reason_options))
        )

        reason_code, description = reason_options[reason_index]

        delay_hours = max(
            shipment.actual_transit_hours
            - shipment.planned_transit_hours,
            0,
        )

        rows.append({
            "delay_reason_id": f"DLY-{index:08d}",
            "shipment_id": shipment.shipment_id,
            "reason_code": reason_code,
            "reason_description": description,
            "delay_hours": round(float(delay_hours), 2),
            "severity": (
                "HIGH" if delay_hours >= 12
                else "MEDIUM" if delay_hours >= 5
                else "LOW"
            ),
        })

    return pd.DataFrame(
        rows,
        columns=[
            "delay_reason_id",
            "shipment_id",
            "reason_code",
            "reason_description",
            "delay_hours",
            "severity",
        ],
    )


# ---------------------------------------------------------
# Ground-truth anomaly register
# ---------------------------------------------------------

def generate_ground_truth(
    shipment_metadata: pd.DataFrame,
    tracking_events: pd.DataFrame,
):
    first_event_id = (
        tracking_events["event_id"].iloc[0]
        if not tracking_events.empty
        else None
    )

    first_shipment_id = (
        shipment_metadata["shipment_id"].iloc[0]
        if not shipment_metadata.empty
        else None
    )

    records = [
        {
            "anomaly_id": "ANOM-001",
            "anomaly_type": "INGESTION_EVENT_TIME_CONFUSION",
            "entity_type": "tracking_event",
            "entity_id": first_event_id,
            "description": (
                "Timestamp-ordering test case. Operational event_time "
                "and ingestion_time are separate concepts. The clean "
                "generated event data retains ingestion_time at or "
                "after event_time."
            ),
            "is_injected_into_raw_data": False,
        },
        {
            "anomaly_id": "ANOM-002",
            "anomaly_type": "HIGH_FACILITY_QUEUE",
            "entity_type": "shipment",
            "entity_id": first_shipment_id,
            "description": (
                "Reference scenario for identifying shipment risk "
                "caused by high facility queue pressure."
            ),
            "is_injected_into_raw_data": False,
        },
        {
            "anomaly_id": "ANOM-003",
            "anomaly_type": "SEVERE_WEATHER",
            "entity_type": "shipment",
            "entity_id": first_shipment_id,
            "description": (
                "Reference scenario for identifying shipment risk "
                "caused by severe weather."
            ),
            "is_injected_into_raw_data": False,
        },
        {
            "anomaly_id": "ANOM-004",
            "anomaly_type": "CARRIER_RELIABILITY_RISK",
            "entity_type": "shipment",
            "entity_id": first_shipment_id,
            "description": (
                "Reference scenario for identifying shipment risk "
                "associated with low carrier reliability."
            ),
            "is_injected_into_raw_data": False,
        },
    ]

    return pd.DataFrame(records)


# ---------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------

def generate_data(
    shipments_count: int = DEFAULT_SHIPMENTS,
    seed: int = DEFAULT_SEED,
):
    if shipments_count < 1:
        raise ValueError("shipments_count must be greater than zero")

    rng = np.random.default_rng(seed)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating logistics ETA dataset...")
    print(f"Shipments: {shipments_count:,}")
    print(f"Random seed: {seed}")

    # 1. Master data
    (
        carriers,
        facilities,
        routes,
        service_levels,
    ) = generate_master_data(rng)

    # 2. Shipments
    shipments, shipment_metadata = generate_shipments(
        rng,
        shipments_count,
        carriers,
        facilities,
        routes,
        service_levels,
    )

    # 3. Shipment legs
    shipment_legs = generate_shipment_legs(
        rng,
        shipment_metadata,
    )

    # 4. Tracking events
    tracking_events = generate_tracking_events(
        rng,
        shipment_metadata,
    )

    # 5. Facility queues
    facility_queues = generate_facility_queues(
        rng,
        shipment_metadata,
        facilities,
    )

    # 6. Weather events
    weather_events = generate_weather_events(rng)

    # 7. Route-weather relationships
    route_weather = generate_route_weather(
        rng,
        routes,
        weather_events,
    )

    # 8. ETA estimates and ML snapshots
    eta_estimates, eta_snapshots = generate_eta_estimates(
        rng,
        shipment_metadata,
        facility_queues,
        weather_events,
    )

    # 9. Delay reasons
    delay_reasons = generate_delay_reasons(
        rng,
        shipment_metadata,
    )

    # 10. Ground truth / anomaly register
    ground_truth = generate_ground_truth(
        shipment_metadata,
        tracking_events,
    )

    # -----------------------------------------------------
    # Save raw tables
    # -----------------------------------------------------

    tables = {
        CARRIERS_FILE: carriers,
        FACILITIES_FILE: facilities,
        ROUTES_FILE: routes,
        SERVICE_LEVELS_FILE: service_levels,
        SHIPMENTS_FILE: shipments,
        SHIPMENT_LEGS_FILE: shipment_legs,
        TRACKING_EVENTS_FILE: tracking_events,
        FACILITY_QUEUES_FILE: facility_queues,
        WEATHER_EVENTS_FILE: weather_events,
        ROUTE_WEATHER_FILE: route_weather,
        ETA_ESTIMATES_FILE: eta_estimates,
        DELAY_REASONS_FILE: delay_reasons,
    }

    for filename, dataframe in tables.items():
        write_csv(dataframe, filename, RAW_DIR)

    # Processed datasets
    write_csv(
        eta_snapshots,
        "eta_snapshots.csv",
        PROCESSED_DIR,
    )

    write_csv(
        ground_truth,
        "ground_truth.csv",
        PROCESSED_DIR,
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    summary = {
        "carrier": len(carriers),
        "facility": len(facilities),
        "route": len(routes),
        "service_level": len(service_levels),
        "shipment": len(shipments),
        "shipment_leg": len(shipment_legs),
        "tracking_event": len(tracking_events),
        "facility_queue": len(facility_queues),
        "weather_event": len(weather_events),
        "route_weather": len(route_weather),
        "eta_estimate": len(eta_estimates),
        "delay_reason": len(delay_reasons),
    }

    print("\nDataset generation completed.")
    print("\nGenerated row counts:")

    for table_name, row_count in summary.items():
        print(f"  {table_name}: {row_count:,}")

    print("\nOutput directories:")
    print(f"  Raw data:       {RAW_DIR}")
    print(f"  Processed data: {PROCESSED_DIR}")

        # Return generated datasets for pipeline and tests.
    # Convert filenames such as "shipment.csv" into keys such as "shipment".
    table_data = {
        Path(filename).stem: dataframe
        for filename, dataframe in tables.items()
    }

    return table_data, eta_snapshots, ground_truth


# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic logistics ETA prediction data."
    )

    parser.add_argument(
        "--shipments",
        type=int,
        default=DEFAULT_SHIPMENTS,
        help="Number of shipments to generate (default: 5000).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible data generation.",
    )

    args = parser.parse_args()

    generate_data(
        shipments_count=args.shipments,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
