from sqlalchemy import func
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import Base, engine, get_db
from backend.models import Customer, Prediction

# Create tables if they do not exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Online Retail ML API")

# Load saved model files from the main project's models folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

model = joblib.load(MODEL_DIR / "repeat_purchase_model.pkl")
feature_names = joblib.load(
    MODEL_DIR / "repeat_purchase_features.pkl"
)


class CustomerInput(BaseModel):
    customer_id: int | None = None
    Recency: int = Field(ge=0)
    Frequency: int = Field(ge=0)
    Monetary: float = Field(ge=0)
    TotalQuantity: int = Field(ge=0)
    UniqueProducts: int = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "Online Retail ML API is running",
        "docs": "/docs"
    }


@app.get("/customers")
def get_customers(db: Session = Depends(get_db)):
    customers = db.query(Customer).limit(100).all()

    return [
        {
            "customer_id": c.customer_id,
            "recency": c.recency,
            "frequency": c.frequency,
            "monetary": c.monetary,
            "repeat_purchase": c.repeat_purchase,
            "cluster": c.cluster,
        }
        for c in customers
    ]


@app.post("/predict")
def predict(
    data: CustomerInput,
    db: Session = Depends(get_db)
):
    try:
        # Build input using the exact feature order used in training
        input_data = pd.DataFrame(
            [[
                data.Recency,
                data.Frequency,
                data.Monetary,
                data.TotalQuantity,
                data.UniqueProducts,
            ]],
            columns=[
                "Recency",
                "Frequency",
                "Monetary",
                "TotalQuantity",
                "UniqueProducts",
            ],
        )

        input_data = input_data[feature_names]

        prediction = int(model.predict(input_data)[0])
        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        # Save prediction to PostgreSQL
        record = Prediction(
            customer_id=data.customer_id,
            predicted_repeat_purchase=prediction,
            probability=probability,
            created_at=datetime.now(),
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "customer_id": data.customer_id,
            "predicted_repeat_purchase": prediction,
            "probability": round(probability, 4),
            "prediction_id": record.id,
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/predictions")
def get_predictions(
    customer_id: int | None = None,
    db: Session = Depends(get_db)
):
    query = db.query(Prediction)

    # Filter by customer ID when one is provided
    if customer_id is not None:
        query = query.filter(
            Prediction.customer_id == customer_id
        )

    predictions = (
        query
        .order_by(Prediction.id.desc())
        .limit(100)
        .all()
    )

    return [
        {
            "prediction_id": p.id,
            "customer_id": p.customer_id,
            "predicted_repeat_purchase": p.predicted_repeat_purchase,
            "probability": p.probability,
            "created_at": (
                p.created_at.isoformat()
                if p.created_at else None
            ),
        }
        for p in predictions
    ]

@app.get("/prediction-stats")
def get_prediction_stats(db: Session = Depends(get_db)):

    total_predictions = (
        db.query(func.count(Prediction.id)).scalar() or 0
    )

    repeat_predictions = (
        db.query(func.count(Prediction.id))
        .filter(Prediction.predicted_repeat_purchase == 1)
        .scalar() or 0
    )

    non_repeat_predictions = (
        db.query(func.count(Prediction.id))
        .filter(Prediction.predicted_repeat_purchase == 0)
        .scalar() or 0
    )

    recent_predictions = (
        db.query(Prediction)
        .order_by(Prediction.id.desc())
        .limit(10)
        .all()
    )

    return {
        "total_predictions": total_predictions,
        "repeat_predictions": repeat_predictions,
        "non_repeat_predictions": non_repeat_predictions,
        "recent_predictions": [
            {
                "prediction_id": p.id,
                "customer_id": p.customer_id,
                "predicted_repeat_purchase": (
                    p.predicted_repeat_purchase
                ),
                "probability": p.probability,
                "created_at": (
                    p.created_at.isoformat()
                    if p.created_at else None
                ),
            }
            for p in recent_predictions
        ],
    }
