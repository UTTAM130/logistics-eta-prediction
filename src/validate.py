import argparse
import json
import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR


# --------------------------------------------------
# REQUIRED TABLES
# --------------------------------------------------

TABLES = [
    "carrier",
    "facility",
    "route",
    "service_level",
    "shipment",
    "shipment_leg",
    "tracking_event",
    "facility_queue",
    "weather_event",
    "route_weather",
    "eta_estimate",
    "delay_reason",
]


# --------------------------------------------------
# PRIMARY KEYS
# --------------------------------------------------

PRIMARY_KEYS = {
    "carrier": "carrier_id",
    "facility": "facility_id",
    "route": "route_id",
    "service_level": "service_level_id",
    "shipment": "shipment_id",
    "shipment_leg": "leg_id",
    "tracking_event": "event_id",
    "facility_queue": "queue_id",
    "weather_event": "weather_event_id",
    "route_weather": "route_weather_id",
    "eta_estimate": "eta_id",
    "delay_reason": "delay_reason_id",
}


# --------------------------------------------------
# VALIDATION FUNCTION
# --------------------------------------------------

def validate_tables():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    tables = {}
    checks = {}

    # ==============================================
    # 1. LOAD DATASETS
    # ==============================================

    print("\nLoading datasets...")

    for name in TABLES:

        file_path = RAW_DIR / f"{name}.csv"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {file_path}"
            )

        try:
            tables[name] = pd.read_csv(file_path)

        except pd.errors.EmptyDataError:
            raise ValueError(
                f"Dataset is empty: {file_path}"
            )

        print(
            f"Loaded {name}: {len(tables[name])} rows"
        )

    # ==============================================
    # 2. PRIMARY KEY VALIDATION
    # ==============================================

    print("\nChecking primary keys...")

    for table, column in PRIMARY_KEYS.items():

        df = tables[table]

        exists = column in df.columns

        checks[f"{table}.{column}.exists"] = bool(exists)

        if not exists:
            checks[f"{table}.{column}.unique"] = False
            continue

        checks[f"{table}.{column}.unique"] = bool(
            df[column].notna().all()
            and df[column].is_unique
        )

    # ==============================================
    # 3. FOREIGN KEY VALIDATION
    # ==============================================

    print("\nChecking foreign keys...")

    def check_fk(
        source,
        source_col,
        target,
        target_col
    ):

        source_df = tables[source]
        target_df = tables[target]

        check_name = f"{source}.{source_col}_fk"

        if source_col not in source_df.columns:
            checks[check_name] = False
            return

        if target_col not in target_df.columns:
            checks[check_name] = False
            return

        valid = (
            source_df[source_col].notna().all()
            and source_df[source_col].isin(
                target_df[target_col]
            ).all()
        )

        checks[check_name] = bool(valid)

    foreign_keys = [

        ("shipment", "carrier_id", "carrier", "carrier_id"),

        ("shipment", "route_id", "route", "route_id"),

        (
            "shipment",
            "service_level_id",
            "service_level",
            "service_level_id"
        ),

        ("shipment_leg", "shipment_id", "shipment", "shipment_id"),

        ("shipment_leg", "route_id", "route", "route_id"),

        ("tracking_event", "shipment_id", "shipment", "shipment_id"),

        ("tracking_event", "facility_id", "facility", "facility_id"),

        ("facility_queue", "facility_id", "facility", "facility_id"),

        ("route_weather", "route_id", "route", "route_id"),

        (
            "route_weather",
            "weather_event_id",
            "weather_event",
            "weather_event_id"
        ),

        ("eta_estimate", "shipment_id", "shipment", "shipment_id"),

        ("delay_reason", "shipment_id", "shipment", "shipment_id"),
    ]

    for source, source_col, target, target_col in foreign_keys:

        check_fk(
            source,
            source_col,
            target,
            target_col
        )

    # ==============================================
    # 4. TIMESTAMP VALIDATION
    # ==============================================

    print("\nChecking timestamps...")

    def check_timestamp_order(
        table,
        earlier_column,
        later_column,
        check_name
    ):

        df = tables[table]

        if (
            earlier_column not in df.columns
            or later_column not in df.columns
        ):
            checks[check_name] = False
            return

        earlier = pd.to_datetime(
            df[earlier_column],
            utc=True,
            errors="coerce"
        )

        later = pd.to_datetime(
            df[later_column],
            utc=True,
            errors="coerce"
        )

        valid = (
            earlier.notna().all()
            and later.notna().all()
            and (later >= earlier).all()
        )

        checks[check_name] = bool(valid)

    # Event ingestion must happen after event creation

    check_timestamp_order(
        "tracking_event",
        "event_time",
        "ingestion_time",
        "events.ingestion_not_before_event"
    )

    # Delivery must happen after shipment creation

    check_timestamp_order(
        "shipment",
        "created_at",
        "actual_delivery_at",
        "shipment.delivery_after_created"
    )

    # ==============================================
    # 5. ETA DATA VALIDATION
    # ==============================================

    print("\nChecking ETA data...")

    eta = tables["eta_estimate"]

    if "future_event_leakage" in eta.columns:

        checks["eta.no_future_leakage_baseline"] = bool(
            eta["future_event_leakage"].notna().all()
            and (eta["future_event_leakage"] == 0).all()
        )

    else:

        checks["eta.no_future_leakage_baseline"] = False

    # ==============================================
    # 6. DELAY LABEL VALIDATION
    # ==============================================

    print("\nChecking delay labels...")

    shipment = tables["shipment"]

    if "is_delayed" in shipment.columns:

        checks["shipment.delay_label_binary"] = bool(
            shipment["is_delayed"].notna().all()
            and shipment["is_delayed"].isin([0, 1]).all()
        )

    else:

        checks["shipment.delay_label_binary"] = False

    # ==============================================
    # 7. DATA QUALITY SUMMARY
    # ==============================================

    checks = {
        name: bool(value)
        for name, value in checks.items()
    }

    total_checks = int(len(checks))

    passed_checks = int(
        sum(checks.values())
    )

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    # ==============================================
    # 8. FINAL VALIDATION REPORT
    # ==============================================

    result = {

        "all_passed": bool(
            all(checks.values())
        ),

        "total_checks": total_checks,

        "passed_checks": passed_checks,

        "failed_checks_count": int(
            len(failed_checks)
        ),

        "failed_checks": failed_checks,

        "checks": checks,

        "row_counts": {
            name: int(len(df))
            for name, df in tables.items()
        }

    }

    # ==============================================
    # 9. SAVE JSON REPORT
    # ==============================================

    output_path = (
        PROCESSED_DIR / "validation_summary.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=4,
            allow_nan=False
        ),
        encoding="utf-8"
    )

    return result


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Logistics ETA Dataset Validation"
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if validation fails."
    )

    args = parser.parse_args()

    try:

        print("\n======================================")
        print(" LOGISTICS ETA DATA VALIDATION")
        print("======================================")

        result = validate_tables()

        print("\n======================================")
        print(" VALIDATION SUMMARY")
        print("======================================")

        print(
            f"Total checks  : {result['total_checks']}"
        )

        print(
            f"Passed checks : {result['passed_checks']}"
        )

        print(
            f"Failed checks : {result['failed_checks_count']}"
        )

        print(
            f"Overall status: "
            f"{'PASSED' if result['all_passed'] else 'FAILED'}"
        )

        if result["failed_checks"]:

            print("\nFailed validations:")

            for check in result["failed_checks"]:
                print(f" - {check}")

        print(
            "\nReport saved to: "
            "data/processed/validation_summary.json"
        )

        if result["all_passed"]:

            print("\nALL VALIDATION CHECKS PASSED")

        else:

            print("\nVALIDATION FAILED")

            if args.strict:
                raise SystemExit(1)

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:

        print(f"\nERROR: {exc}")

        raise SystemExit(1)


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
