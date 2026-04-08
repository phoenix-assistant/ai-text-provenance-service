#!/usr/bin/env python3
"""Export trained model to ONNX format for production serving.

Usage:
    python scripts/export_onnx.py --model outputs/best_model.pt --output models/model.onnx
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to PyTorch model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/model.onnx",
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for export",
    )

    args = parser.parse_args()

    from ai_text_provenance.models.classifier import ProvenanceClassifier

    # Load model
    logger.info(f"Loading model from {args.model}")
    classifier = ProvenanceClassifier(
        model_path=args.model,
        device=args.device,
        use_onnx=False,
    )

    # Export to ONNX
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting to {args.output}")
    classifier.export_onnx(str(output_path))

    logger.info("Export complete!")
    logger.info(f"ONNX model saved to: {args.output}")
    logger.info(f"Model size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
