"""Inference engine for production serving.

Optimized for:
- Low latency single requests
- High throughput batch processing
- Memory efficiency
- Optional GPU acceleration
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from ai_text_provenance.models.classifier import ProvenanceClassifier
from ai_text_provenance.models.schemas import (
    ClassificationResult,
)

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Production inference engine with optimizations.

    Features:
    - Model warming
    - Request batching
    - Async execution
    - Memory management
    """

    def __init__(
        self,
        model_path: str | None = None,
        use_onnx: bool = True,
        device: str | None = None,
        max_batch_size: int = 32,
        max_workers: int = 4,
    ):
        """Initialize the inference engine.

        Args:
            model_path: Path to model weights or ONNX file.
            use_onnx: Use ONNX Runtime for faster inference.
            device: Device to use ('cpu', 'cuda', 'mps').
            max_batch_size: Maximum batch size for processing.
            max_workers: Thread pool size for concurrent requests.
        """
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers

        # Initialize classifier
        self.classifier = ProvenanceClassifier(
            model_path=model_path,
            use_onnx=use_onnx and model_path is not None,
            device=device,
        )

        # Thread pool for async execution
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Warm up the model
        self._warm_up()

        logger.info(
            f"Inference engine initialized (ONNX: {use_onnx}, device: {self.classifier.device})"
        )

    def _warm_up(self) -> None:
        """Warm up the model with dummy inference."""
        logger.info("Warming up model...")

        dummy_text = (
            "This is a sample text for warming up the model. "
            "It needs to be long enough to exercise all code paths. "
            "The quick brown fox jumps over the lazy dog. "
            "Machine learning models often need warm-up because "
            "the first inference may be slower due to JIT compilation "
            "and memory allocation. This warm-up ensures that actual "
            "user requests have consistent latency."
        )

        # Run a few warm-up inferences
        for _ in range(3):
            self.classifier.classify(dummy_text)

        logger.info("Model warmed up")

    def classify(self, text: str, include_features: bool = False) -> ClassificationResult:
        """Synchronous single text classification.

        Args:
            text: Text to classify.
            include_features: Include extracted features in response.

        Returns:
            ClassificationResult with prediction and probabilities.
        """
        return self.classifier.classify(text, include_features=include_features)

    def classify_batch(
        self, texts: list[str], include_features: bool = False
    ) -> list[ClassificationResult]:
        """Synchronous batch classification.

        Args:
            texts: List of texts to classify.
            include_features: Include extracted features in responses.

        Returns:
            List of ClassificationResult objects.
        """
        # Process in chunks if batch is too large
        if len(texts) <= self.max_batch_size:
            return self.classifier.classify_batch(texts, include_features)

        results = []
        for i in range(0, len(texts), self.max_batch_size):
            chunk = texts[i : i + self.max_batch_size]
            chunk_results = self.classifier.classify_batch(chunk, include_features)
            results.extend(chunk_results)

        return results

    async def classify_async(
        self, text: str, include_features: bool = False
    ) -> ClassificationResult:
        """Async single text classification.

        Args:
            text: Text to classify.
            include_features: Include extracted features in response.

        Returns:
            ClassificationResult with prediction and probabilities.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.classify(text, include_features),
        )

    async def classify_batch_async(
        self, texts: list[str], include_features: bool = False
    ) -> list[ClassificationResult]:
        """Async batch classification.

        Args:
            texts: List of texts to classify.
            include_features: Include extracted features in responses.

        Returns:
            List of ClassificationResult objects.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.classify_batch(texts, include_features),
        )

    def health_check(self) -> dict:
        """Check if the engine is healthy.

        Returns:
            Health status dict.
        """
        try:
            # Quick inference test
            start = time.time()
            self.classify("This is a health check test sentence.")
            latency_ms = (time.time() - start) * 1000

            return {
                "status": "healthy",
                "model_loaded": True,
                "latency_ms": round(latency_ms, 2),
                "device": self.classifier.device,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_loaded": False,
                "error": str(e),
            }

    def shutdown(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        logger.info("Inference engine shut down")
