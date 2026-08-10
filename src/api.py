from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
MODEL_ARTIFACT = ROOT / "artifacts" / "best_model.joblib"


class PredictRequest(BaseModel):
    sensors: list[float | None] = Field(
        ...,
        description="Sensor values ordered as sensor_000 ... sensor_589. Use null for missing sensor values.",
    )
    threshold_override: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional operating threshold override for what-if checks.",
    )


class PredictResponse(BaseModel):
    model_name: str
    score_fail: float
    threshold: float
    prediction: str
    decision: str
    sensor_count: int


def create_app() -> FastAPI:
    app = FastAPI(
        title="SECOM FDC Fault Detection API",
        version="0.1.0",
        description="Inference API for fail-risk screening using the trained SECOM decision-support model.",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        if not MODEL_ARTIFACT.exists():
            return {
                "status": "degraded",
                "artifact_loaded": False,
                "message": "Run `python src/fetch_data.py` and `python src/train.py` first.",
            }
        artifact = load_artifact()
        return {
            "status": "ok",
            "artifact_loaded": True,
            "model_name": artifact["model_name"],
            "feature_count": len(artifact["feature_names"]),
        }

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        artifact = require_artifact()
        return {
            "model_name": artifact["model_name"],
            "threshold": artifact["threshold"],
            "positive_class": artifact["positive_class"],
            "score_name": artifact["score_name"],
            "feature_count": len(artifact["feature_names"]),
            "metrics": artifact["metrics"],
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        artifact = require_artifact()
        feature_names = artifact["feature_names"]
        if len(request.sensors) != len(feature_names):
            raise HTTPException(
                status_code=422,
                detail=f"Expected {len(feature_names)} sensor values, received {len(request.sensors)}.",
            )

        threshold = request.threshold_override if request.threshold_override is not None else artifact["threshold"]
        frame = pd.DataFrame([request.sensors], columns=feature_names, dtype=float)
        score = fail_score(artifact["model"], frame)
        predicted_fail = score >= threshold
        return PredictResponse(
            model_name=artifact["model_name"],
            score_fail=round(score, 6),
            threshold=round(float(threshold), 6),
            prediction="fail_risk" if predicted_fail else "pass",
            decision="review_required" if predicted_fail else "no_alarm",
            sensor_count=len(feature_names),
        )

    return app


@lru_cache(maxsize=1)
def load_artifact() -> dict[str, Any]:
    return joblib.load(MODEL_ARTIFACT)


def require_artifact() -> dict[str, Any]:
    if not MODEL_ARTIFACT.exists():
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found. Run `python src/fetch_data.py` and `python src/train.py` first.",
        )
    return load_artifact()


def fail_score(model: Any, frame: pd.DataFrame) -> float:
    estimator = model[-1]
    if hasattr(estimator, "predict_proba"):
        return float(model.predict_proba(frame)[:, 1][0])
    if hasattr(estimator, "decision_function"):
        scores = model.decision_function(frame)
        return float((scores - scores.min()) / (scores.max() - scores.min() + 1e-9))
    return float(model.predict(frame)[0])


app = create_app()
