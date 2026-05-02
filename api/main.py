from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_pipeline
from src.utils import setup_logging


app = FastAPI(title="ML Cloud Security API", version="1.0.0")
logger = setup_logging("api")
pipeline = load_pipeline()


class PredictionRequest(BaseModel):
    rows: list[dict]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "classes": pipeline.metadata["classes"],
        "feature_count": pipeline.metadata["feature_count"],
        "trained_at": pipeline.metadata.get("trained_at"),
    }


@app.get("/schema")
def schema():
    return pipeline.input_schema()


@app.post("/predict")
def predict(payload: PredictionRequest):
    if not payload.rows:
        raise HTTPException(status_code=400, detail="rows cannot be empty")
    try:
        frame = pd.DataFrame(payload.rows)
        result = pipeline.predict_dataframe(frame)
        return {"predictions": result.to_dict(orient="records")}
    except Exception as exc:
        logger.exception("Prediction request failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
