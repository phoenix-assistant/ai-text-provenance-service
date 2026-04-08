#!/usr/bin/env python3
"""Training script for the AI Text Provenance classifier.

Usage:
    python scripts/train.py --train data/train.jsonl --val data/val.jsonl

Or to create sample data first:
    python scripts/train.py --create-sample-data
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train the provenance classifier")
    parser.add_argument("--train", type=str, help="Path to training JSONL file")
    parser.add_argument("--val", type=str, help="Path to validation JSONL file")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--device", type=str, help="Device (cpu, cuda, mps)")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample training data",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Samples per class for sample data",
    )

    args = parser.parse_args()

    if args.create_sample_data:
        from ai_text_provenance.training.dataset import ProvenanceDataset

        # Create output directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        # Create train and val splits
        ProvenanceDataset.create_sample_dataset(
            str(data_dir / "train.jsonl"),
            num_samples=args.sample_size,
        )
        ProvenanceDataset.create_sample_dataset(
            str(data_dir / "val.jsonl"),
            num_samples=args.sample_size // 5,
        )

        logger.info("Sample data created in data/ directory")
        logger.info("Run training with: python scripts/train.py --train data/train.jsonl --val data/val.jsonl")
        return

    if not args.train:
        parser.error("--train is required (or use --create-sample-data first)")

    # Import training modules
    from ai_text_provenance.models.classifier import EnsembleClassifier
    from ai_text_provenance.training.dataset import ProvenanceDataset
    from ai_text_provenance.training.trainer import Trainer

    # Load datasets
    logger.info(f"Loading training data from {args.train}")
    train_dataset = ProvenanceDataset(
        args.train,
        precompute_features=True,
        cache_path=str(Path(args.output) / "train_features.json"),
    )

    val_dataset = None
    if args.val:
        logger.info(f"Loading validation data from {args.val}")
        val_dataset = ProvenanceDataset(
            args.val,
            precompute_features=True,
            cache_path=str(Path(args.output) / "val_features.json"),
        )

    # Create model
    logger.info("Creating model...")
    model = EnsembleClassifier(
        transformer_model="distilbert-base-uncased",
        feature_dim=train_dataset.feature_extractor.num_features,
        num_classes=4,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        device=args.device,
        output_dir=args.output,
        use_wandb=args.wandb,
    )

    # Train
    logger.info("Starting training...")
    stats = trainer.train()

    logger.info(f"Training complete! Best val accuracy: {stats.get('best_val_acc', 'N/A')}")
    logger.info(f"Model saved to {args.output}/best_model.pt")


if __name__ == "__main__":
    main()
