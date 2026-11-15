import argparse
import hashlib
import json

from src.config import RAW_DIR, PROCESSED_DIR, ensure_dirs
from src.generate import generate_data
from src.validate import validate_tables
from src.model import train_models


def generate_checksums():
    checksums = {}

    files = list(RAW_DIR.glob("*.csv"))
    files += list(PROCESSED_DIR.glob("*.csv"))

    for file in sorted(files):
        checksums[file.name] = hashlib.sha256(
            file.read_bytes()
        ).hexdigest()

    output = PROCESSED_DIR / "checksums.json"

    output.write_text(
        json.dumps(checksums, indent=4)
    )

    return checksums


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shipments",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261022
    )

    args = parser.parse_args()

    ensure_dirs()

    print("STEP 1: Generating synthetic data...")

    tables, eta, truth = generate_data(
        args.shipments,
        args.seed
    )

    print("STEP 2: Validating data...")

    validation = validate_tables()

    if not validation["all_passed"]:
        raise SystemExit("Data validation failed!")

    print("STEP 3: Training ML models...")

    metrics = train_models()

    print("STEP 4: Generating checksums...")

    checksums = generate_checksums()

    print("\nPROJECT PIPELINE COMPLETED")

    print("Generated records:")

    for name, df in tables.items():
        print(f"{name}: {len(df)}")

    print("\nValidation:", validation["all_passed"])

    print("\nModel metrics:")

    print(json.dumps(metrics, indent=4))

    print("\nChecksums:", len(checksums))


if __name__ == "__main__":
    main()

