"""Command-line interface for the AI Text Provenance Service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ai_text_provenance import __version__


def classify_command(args: argparse.Namespace) -> None:
    """Run classification on text input."""
    from ai_text_provenance.models.classifier import ProvenanceClassifier

    # Initialize classifier
    classifier = ProvenanceClassifier(
        model_path=args.model,
        device=args.device,
    )

    # Get text input
    if args.file:
        text = Path(args.file).read_text()
    elif args.text:
        text = args.text
    else:
        # Read from stdin
        print("Enter text to classify (Ctrl+D to finish):", file=sys.stderr)
        text = sys.stdin.read()

    if not text.strip():
        print("Error: No text provided", file=sys.stderr)
        sys.exit(1)

    # Classify
    result = classifier.classify(text, include_features=args.features)

    # Output
    if args.json:
        output = {
            "prediction": result.prediction.value if hasattr(result.prediction, 'value') else result.prediction,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
        }
        if result.features:
            output["features"] = result.features.model_dump()
        print(json.dumps(output, indent=2))
    else:
        print(f"\nPrediction: {result.prediction.value if hasattr(result.prediction, 'value') else result.prediction}")
        print(f"Confidence: {result.confidence:.2%}")
        print("\nProbabilities:")
        for cls, prob in sorted(result.probabilities.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            print(f"  {cls:15s} {prob:.2%} {bar}")


def batch_command(args: argparse.Namespace) -> None:
    """Run batch processing on a file."""
    from ai_text_provenance.models.classifier import ProvenanceClassifier
    from ai_text_provenance.inference.batch import BatchProcessor

    # Initialize
    classifier = ProvenanceClassifier(
        model_path=args.model,
        device=args.device,
    )

    def progress_callback(processed: int, total: int, elapsed: float) -> None:
        rate = processed / elapsed if elapsed > 0 else 0
        print(
            f"\rProcessed {processed}/{total} ({processed/total:.1%}) - "
            f"{rate:.1f} texts/sec",
            end="",
            file=sys.stderr,
        )

    processor = BatchProcessor(
        classifier=classifier,
        batch_size=args.batch_size,
        progress_callback=progress_callback if not args.quiet else None,
    )

    # Process
    stats = processor.process_file(
        input_path=args.input,
        output_path=args.output,
        text_field=args.text_field,
        include_features=args.features,
    )

    if not args.quiet:
        print(file=sys.stderr)  # New line after progress
        print(f"Processed {stats['total_processed']} texts in {stats['elapsed_seconds']:.2f}s")
        print(f"Throughput: {stats['texts_per_second']:.1f} texts/sec")
        print(f"Output written to: {stats['output_file']}")


def server_command(args: argparse.Namespace) -> None:
    """Start the API server."""
    from ai_text_provenance.api.server import run

    print(f"Starting AI Text Provenance Service v{__version__}")
    print(f"Listening on http://{args.host}:{args.port}")

    run(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


def features_command(args: argparse.Namespace) -> None:
    """Extract and display features from text."""
    from ai_text_provenance.features.extractor import FeatureExtractor

    extractor = FeatureExtractor()

    # Get text input
    if args.file:
        text = Path(args.file).read_text()
    elif args.text:
        text = args.text
    else:
        print("Enter text to analyze (Ctrl+D to finish):", file=sys.stderr)
        text = sys.stdin.read()

    if not text.strip():
        print("Error: No text provided", file=sys.stderr)
        sys.exit(1)

    # Extract features
    features = extractor.extract(text)

    # Output
    if args.json:
        print(json.dumps(features.model_dump(), indent=2))
    else:
        print("\n=== RST Features ===")
        for name, value in features.rst.model_dump().items():
            print(f"  {name}: {value}")

        print("\n=== Linguistic Features ===")
        for name, value in features.linguistic.model_dump().items():
            print(f"  {name}: {value}")

        print("\n=== Statistical Features ===")
        for name, value in features.statistical.model_dump().items():
            print(f"  {name}: {value}")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="provenance",
        description="AI Text Provenance Service - 4-way text classification",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Classify command
    classify_parser = subparsers.add_parser(
        "classify",
        help="Classify a single text",
    )
    classify_parser.add_argument(
        "text",
        nargs="?",
        help="Text to classify (or use --file or stdin)",
    )
    classify_parser.add_argument(
        "-f", "--file",
        help="Read text from file",
    )
    classify_parser.add_argument(
        "-m", "--model",
        help="Path to model weights",
    )
    classify_parser.add_argument(
        "-d", "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device to use",
    )
    classify_parser.add_argument(
        "--features",
        action="store_true",
        help="Include extracted features in output",
    )
    classify_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    classify_parser.set_defaults(func=classify_command)

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Process texts from a JSONL file",
    )
    batch_parser.add_argument(
        "input",
        help="Input JSONL file",
    )
    batch_parser.add_argument(
        "output",
        help="Output JSONL file",
    )
    batch_parser.add_argument(
        "-m", "--model",
        help="Path to model weights",
    )
    batch_parser.add_argument(
        "-d", "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device to use",
    )
    batch_parser.add_argument(
        "--text-field",
        default="text",
        help="Field name containing text (default: text)",
    )
    batch_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)",
    )
    batch_parser.add_argument(
        "--features",
        action="store_true",
        help="Include extracted features in output",
    )
    batch_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    batch_parser.set_defaults(func=batch_command)

    # Server command
    server_parser = subparsers.add_parser(
        "server",
        help="Start the API server",
    )
    server_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    server_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    server_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    server_parser.set_defaults(func=server_command)

    # Features command
    features_parser = subparsers.add_parser(
        "features",
        help="Extract features from text (without classification)",
    )
    features_parser.add_argument(
        "text",
        nargs="?",
        help="Text to analyze (or use --file or stdin)",
    )
    features_parser.add_argument(
        "-f", "--file",
        help="Read text from file",
    )
    features_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    features_parser.set_defaults(func=features_command)

    return parser


def app() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    app()
