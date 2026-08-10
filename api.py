"""FastAPI backend for Language Identification.

Endpoints:

* ``GET  /``          - API status
* ``GET  /health``    - health probe (also loads the model)
* ``GET  /languages`` - list of supported languages
* ``POST /predict``   - predict the language of one text
* ``POST /batch``     - predict many texts at once

Run locally::

    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import PredictionError, get_predictor
from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger("api")

#: App metadata
APP_TITLE = "Language Identification API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "REST API for the multilingual language identification model. "
    "Send a JSON body {\"text\": \"Bonjour tout le monde\"} to /predict."
)

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS: allow the Streamlit UI and any local frontend to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    """Body of a single prediction request."""

    text: str = Field(..., min_length=1, max_length=100_000, description="Input text.")
    top_k: int = Field(5, ge=1, le=30, description="Number of ranked predictions to return.")


class BatchPredictRequest(BaseModel):
    """Body of a batch prediction request."""

    texts: list[str] = Field(..., min_length=1, max_length=1000, description="List of input texts.")
    top_k: int = Field(5, ge=1, le=30, description="Number of ranked predictions per text.")


# --------------------------------------------------------------------------- #
# Lifespan: load the model once at startup (fail fast with a clear message).
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def _load_model() -> None:
    try:
        get_predictor()
        logger.info("Language model loaded successfully.")
    except FileNotFoundError as exc:
        logger.error("Model not found: %s. Run `python -m src.train` first.", exc)
        raise


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/", tags=["meta"])
def root() -> dict[str, Any]:
    """Return basic API status."""
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "endpoints": ["/predict", "/batch", "/languages", "/health"],
    }


@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    """Health probe. Verifies the model artifacts are loadable."""
    predictor = get_predictor()
    return {
        "status": "ok",
        "model": predictor.meta.get("model", "unknown"),
        "n_languages": len(predictor.languages),
        "accuracy": predictor.meta.get("accuracy"),
    }


@app.get("/languages", tags=["meta"])
def languages() -> dict[str, Any]:
    """List supported languages."""
    predictor = get_predictor()
    return {"count": len(predictor.languages), "languages": sorted(predictor.languages)}


@app.post("/predict", tags=["predict"])
def predict(request: PredictRequest) -> dict[str, Any]:
    """Predict the language of a single text.

    Example body::

        {"text": "Hello, how are you?"}

    Returns the predicted language, confidence and top-k probabilities.
    """
    predictor = get_predictor()
    try:
        return predictor.predict(request.text, top_k=request.top_k)
    except PredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/batch", tags=["predict"])
def batch_predict(request: BatchPredictRequest) -> dict[str, Any]:
    """Predict languages for many texts in one call."""
    predictor = get_predictor()
    start = time.perf_counter()
    results = predictor.predict_many(request.texts, top_k=request.top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {"count": len(results), "total_time_ms": round(elapsed_ms, 2), "results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
