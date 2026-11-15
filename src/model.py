import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    PROCESSED_DIR,
    MODEL_DIR,
    ensure_dirs
)


FEATURES = [
    "planned_remaining_hours",
    "queue_pressure",
    "weather_severity",
    "carrier_reliability"
]


def train_models():

    ensure_dirs()

    data_path = PROCESSED_DIR / "eta_snapshots.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {data_path}. "
            "Run the data preparation pipeline first."
        )

    df = pd.read_csv(data_path)

    required_columns = FEATURES + [
        "actual_remaining_hours",
        "delay_label"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if len(df) < 10:
        raise ValueError(
            "Insufficient training records. "
            "At least 10 records are required."
        )

    # Prepare features and targets
    X = df[FEATURES].copy()

    for column in FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    y_eta = pd.to_numeric(
        df["actual_remaining_hours"],
        errors="coerce"
    )

    y_delay = pd.to_numeric(
        df["delay_label"],
        errors="coerce"
    )

    # Remove rows with invalid target values
    valid_rows = (
        y_eta.notna()
        & y_delay.notna()
        & y_eta.ge(0)
        & y_delay.isin([0, 1])
    )

    X = X.loc[valid_rows].reset_index(drop=True)
    y_eta = y_eta.loc[valid_rows].reset_index(drop=True)
    y_delay = y_delay.loc[valid_rows].astype(int).reset_index(drop=True)

    if len(X) < 10:
        raise ValueError(
            "Not enough valid training records after cleaning."
        )

    if y_delay.nunique() != 2:
        raise ValueError(
            "Delay classification requires both classes: "
            "0 (on time) and 1 (delayed)."
        )

    class_counts = y_delay.value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Each delay class must contain at least 2 records."
        )

    # Split once so both models use identical records
    X_train, X_test, y_eta_train, y_eta_test, \
        y_delay_train, y_delay_test = train_test_split(
            X,
            y_eta,
            y_delay,
            test_size=0.2,
            random_state=42,
            stratify=y_delay
        )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    )
                ]),
                FEATURES
            )
        ]
    )

    # ETA regression model
    eta_model = Pipeline([
        ("prep", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=120,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    # Delay classification model
    delay_model = Pipeline([
        ("prep", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=120,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    # Train
    eta_model.fit(X_train, y_eta_train)

    delay_model.fit(X_train, y_delay_train)

    # Predictions
    eta_pred = eta_model.predict(X_test)

    delay_pred = delay_model.predict(X_test)

    classifier = delay_model.named_steps["model"]

    positive_class_index = list(
        classifier.classes_
    ).index(1)

    delay_prob = delay_model.predict_proba(
        X_test
    )[:, positive_class_index]

    # Evaluation
    metrics = {
        "eta": {
            "mae_hours": float(
                mean_absolute_error(
                    y_eta_test,
                    eta_pred
                )
            ),
            "rmse_hours": float(
                np.sqrt(
                    mean_squared_error(
                        y_eta_test,
                        eta_pred
                    )
                )
            ),
            "r2": float(
                r2_score(
                    y_eta_test,
                    eta_pred
                )
            )
        },

        "delay": {
            "accuracy": float(
                accuracy_score(
                    y_delay_test,
                    delay_pred
                )
            ),
            "precision": float(
                precision_score(
                    y_delay_test,
                    delay_pred,
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_delay_test,
                    delay_pred,
                    zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y_delay_test,
                    delay_pred,
                    zero_division=0
                )
            ),
            "roc_auc": float(
                roc_auc_score(
                    y_delay_test,
                    delay_prob
                )
            )
        },

        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES
    }

    # Save models
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        eta_model,
        MODEL_DIR / "eta_model.joblib"
    )

    joblib.dump(
        delay_model,
        MODEL_DIR / "delay_model.joblib"
    )

    # Save metrics
    metrics_path = PROCESSED_DIR / "metrics.json"

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    print("\nMODEL TRAINING COMPLETED")
    print(json.dumps(metrics, indent=2))

    return metrics


def predict_one(payload):

    eta_path = MODEL_DIR / "eta_model.joblib"
    delay_path = MODEL_DIR / "delay_model.joblib"

    if not eta_path.exists() or not delay_path.exists():
        raise FileNotFoundError(
            "Trained models not found. Run train_models() first."
        )

    missing_features = [
        feature for feature in FEATURES
        if feature not in payload
    ]

    if missing_features:
        raise ValueError(
            f"Missing prediction features: {missing_features}"
        )

    row = pd.DataFrame([
        {
            feature: payload[feature]
            for feature in FEATURES
        }
    ])

    for feature in FEATURES:
        row[feature] = pd.to_numeric(
            row[feature],
            errors="coerce"
        )

    if row.isna().any().any():
        raise ValueError(
            "Prediction features must contain valid numeric values."
        )

    eta_model = joblib.load(eta_path)

    delay_model = joblib.load(delay_path)

    eta_hours = max(
        0.0,
        float(eta_model.predict(row)[0])
    )

    classifier = delay_model.named_steps["model"]

    positive_class_index = list(
        classifier.classes_
    ).index(1)

    probability = float(
        delay_model.predict_proba(row)[0, positive_class_index]
    )

    if probability >= 0.70:
        risk = "HIGH"
    elif probability >= 0.40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "predicted_remaining_hours": round(
            eta_hours, 2
        ),
        "delay_probability": round(
            probability, 4
        ),
        "delay_risk": risk
    }


if __name__ == "__main__":

    train_models()
