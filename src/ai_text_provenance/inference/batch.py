"""Batch processing utilities for high-throughput scenarios.

Handles:
- File-based batch processing
- Streaming results
- Progress tracking
- Error handling
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional
import logging
import time

from ai_text_provenance.models.classifier import ProvenanceClassifier
from ai_text_provenance.models.schemas import ClassificationResult

logger = logging.getLogger(__name__)


class BatchProcessor:
    """High-throughput batch processor for text classification."""

    def __init__(
        self,
        classifier: ProvenanceClassifier,
        batch_size: int = 32,
        progress_callback: Optional[callable] = None,
    ):
        """Initialize the batch processor.

        Args:
            classifier: ProvenanceClassifier instance.
            batch_size: Number of texts to process at once.
            progress_callback: Optional callback for progress updates.
                               Receives (processed, total, elapsed_seconds).
        """
        self.classifier = classifier
        self.batch_size = batch_size
        self.progress_callback = progress_callback

    def process_texts(
        self,
        texts: list[str],
        include_features: bool = False,
    ) -> list[ClassificationResult]:
        """Process a list of texts.

        Args:
            texts: List of texts to classify.
            include_features: Include extracted features in results.

        Returns:
            List of ClassificationResult objects.
        """
        results = []
        total = len(texts)
        start_time = time.time()

        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_results = self.classifier.classify_batch(batch, include_features)
            results.extend(batch_results)

            if self.progress_callback:
                elapsed = time.time() - start_time
                self.progress_callback(len(results), total, elapsed)

        return results

    def process_file(
        self,
        input_path: str,
        output_path: str,
        text_field: str = "text",
        include_features: bool = False,
    ) -> dict:
        """Process texts from a JSONL file.

        Args:
            input_path: Path to input JSONL file.
            output_path: Path to output JSONL file.
            text_field: Field name containing text in each JSON object.
            include_features: Include extracted features in output.

        Returns:
            Processing statistics.
        """
        input_file = Path(input_path)
        output_file = Path(output_path)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Read all texts
        texts = []
        records = []

        with open(input_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                if text_field not in record:
                    logger.warning(f"Missing field '{text_field}' in record, skipping")
                    continue

                texts.append(record[text_field])
                records.append(record)

        if not texts:
            raise ValueError(f"No texts found in input file with field '{text_field}'")

        # Process
        start_time = time.time()
        results = self.process_texts(texts, include_features)
        elapsed = time.time() - start_time

        # Write output
        with open(output_file, "w") as f:
            for record, result in zip(records, results):
                output_record = {
                    **record,
                    "provenance": {
                        "prediction": result.prediction.value if hasattr(result.prediction, 'value') else result.prediction,
                        "confidence": result.confidence,
                        "probabilities": result.probabilities,
                    },
                }

                if include_features and result.features:
                    output_record["provenance"]["features"] = result.features.model_dump()

                f.write(json.dumps(output_record) + "\n")

        stats = {
            "total_processed": len(results),
            "elapsed_seconds": round(elapsed, 2),
            "texts_per_second": round(len(results) / elapsed, 2) if elapsed > 0 else 0,
            "input_file": str(input_path),
            "output_file": str(output_path),
        }

        logger.info(f"Batch processing complete: {stats}")
        return stats

    def stream_results(
        self,
        texts: Iterator[str],
        include_features: bool = False,
    ) -> Iterator[ClassificationResult]:
        """Stream results for an iterator of texts.

        Yields results as they are processed, useful for large datasets.

        Args:
            texts: Iterator of texts to classify.
            include_features: Include extracted features in results.

        Yields:
            ClassificationResult objects.
        """
        batch = []

        for text in texts:
            batch.append(text)

            if len(batch) >= self.batch_size:
                results = self.classifier.classify_batch(batch, include_features)
                for result in results:
                    yield result
                batch = []

        # Process remaining texts
        if batch:
            results = self.classifier.classify_batch(batch, include_features)
            for result in results:
                yield result

    async def stream_results_async(
        self,
        texts: AsyncIterator[str],
        include_features: bool = False,
    ) -> AsyncIterator[ClassificationResult]:
        """Async stream results.

        Args:
            texts: Async iterator of texts to classify.
            include_features: Include extracted features in results.

        Yields:
            ClassificationResult objects.
        """
        batch = []

        async for text in texts:
            batch.append(text)

            if len(batch) >= self.batch_size:
                # Run classification in thread pool
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda b=batch: self.classifier.classify_batch(b, include_features),
                )
                for result in results:
                    yield result
                batch = []

        # Process remaining texts
        if batch:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.classifier.classify_batch(batch, include_features),
            )
            for result in results:
                yield result
