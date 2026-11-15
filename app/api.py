from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_DIR
from src.model import predict_one

app = FastAPI(
    title="Logistics ETA Prediction API",
    description="Shipment ETA and delay-risk prediction system",
    version="1.0.0"
)


class PredictionInput(BaseModel):
    planned_remaining_hours: float = Field(ge=0)
    queue_pressure: float = Field(ge=0)
    weather_severity: int = Field(ge=0, le=3)
    carrier_reliability: float = Field(ge=0, le=1)


@app.get("/")
def home():
    return {
        "message": "Logistics ETA Prediction API is running",
        "status": "success"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: PredictionInput):

    if not (MODEL_DIR / "eta_model.joblib").exists():
        raise HTTPException(
            status_code=503,
            detail="Models not found. Run the training pipeline first."
        )

    try:
        result = predict_one(data.model_dump())
        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
