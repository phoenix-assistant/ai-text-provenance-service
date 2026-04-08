"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings

from ai_text_provenance import __version__
from ai_text_provenance.inference.engine import InferenceEngine
from ai_text_provenance.models.schemas import (
    ClassifyRequest,
    ClassifyBatchRequest,
    ClassifyResponse,
    ClassifyBatchResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings."""

    model_path: Optional[str] = None
    use_onnx: bool = True
    device: Optional[str] = None
    max_batch_size: int = 32
    cors_origins: list[str] = ["*"]
    log_level: str = "INFO"

    class Config:
        env_prefix = "PROVENANCE_"


# Global engine instance (initialized in lifespan)
_engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    """Get the inference engine instance."""
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _engine

    # Startup
    settings = Settings()

    logging.basicConfig(level=settings.log_level)
    logger.info(f"Starting AI Text Provenance Service v{__version__}")

    # Initialize engine
    _engine = InferenceEngine(
        model_path=settings.model_path,
        use_onnx=settings.use_onnx,
        device=settings.device,
        max_batch_size=settings.max_batch_size,
    )

    logger.info("Service ready")

    yield

    # Shutdown
    logger.info("Shutting down...")
    if _engine:
        _engine.shutdown()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI Text Provenance Service",
        description=(
            "4-way text classification API distinguishing: "
            "human-written, AI-generated, polished human, humanized AI."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    # CORS
    settings = Settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = str(round(process_time, 2))
        return response

    # Routes
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        engine = get_engine()
        health_status = engine.health_check()

        return HealthResponse(
            status=health_status.get("status", "unknown"),
            version=__version__,
            model_loaded=health_status.get("model_loaded", False),
            spacy_loaded=True,  # Loaded during engine init
        )

    @app.post("/classify", response_model=ClassifyResponse)
    async def classify(request: ClassifyRequest):
        """Classify a single text.

        Returns the predicted provenance class (human, ai, polished_human, humanized_ai)
        along with confidence scores and per-class probabilities.
        """
        engine = get_engine()

        try:
            result = await engine.classify_async(
                request.text,
                include_features=request.include_features,
            )

            return ClassifyResponse(
                prediction=result.prediction.value if hasattr(result.prediction, 'value') else result.prediction,
                confidence=result.confidence,
                probabilities=result.probabilities,
                features=result.features.model_dump() if result.features else None,
            )
        except Exception as e:
            logger.exception("Classification error")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/classify/batch", response_model=ClassifyBatchResponse)
    async def classify_batch(request: ClassifyBatchRequest):
        """Classify multiple texts in a batch.

        More efficient than multiple single requests for bulk processing.
        Maximum batch size is 100 texts.
        """
        engine = get_engine()

        if len(request.texts) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum batch size is 100 texts",
            )

        try:
            start_time = time.time()

            results = await engine.classify_batch_async(
                request.texts,
                include_features=request.include_features,
            )

            processing_time = (time.time() - start_time) * 1000

            return ClassifyBatchResponse(
                results=[
                    ClassifyResponse(
                        prediction=r.prediction.value if hasattr(r.prediction, 'value') else r.prediction,
                        confidence=r.confidence,
                        probabilities=r.probabilities,
                        features=r.features.model_dump() if r.features else None,
                    )
                    for r in results
                ],
                total=len(results),
                processing_time_ms=round(processing_time, 2),
            )
        except Exception as e:
            logger.exception("Batch classification error")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """Root endpoint with API info."""
        return {
            "service": "AI Text Provenance Service",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Create app instance for uvicorn
app = create_app()
