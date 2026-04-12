"""Training loop and utilities for the provenance classifier.

Supports:
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Checkpointing
- Wandb logging
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from ai_text_provenance.models.classifier import EnsembleClassifier
from ai_text_provenance.training.dataset import ProvenanceDataset

logger = logging.getLogger(__name__)

# Label names for reporting
LABEL_NAMES = ["human", "ai", "polished_human", "humanized_ai"]


class Trainer:
    """Trainer for the provenance classifier."""

    def __init__(
        self,
        model: EnsembleClassifier,
        train_dataset: ProvenanceDataset,
        val_dataset: ProvenanceDataset | None = None,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        device: str | None = None,
        output_dir: str = "outputs",
        use_wandb: bool = False,
        wandb_project: str = "ai-text-provenance",
    ):
        """Initialize the trainer.

        Args:
            model: EnsembleClassifier to train.
            train_dataset: Training dataset.
            val_dataset: Optional validation dataset.
            batch_size: Batch size for training.
            learning_rate: Peak learning rate.
            weight_decay: Weight decay for AdamW.
            num_epochs: Number of training epochs.
            warmup_ratio: Fraction of steps for warmup.
            gradient_accumulation_steps: Gradient accumulation steps.
            max_grad_norm: Maximum gradient norm for clipping.
            device: Device to train on.
            output_dir: Directory for checkpoints and logs.
            use_wandb: Enable Weights & Biases logging.
            wandb_project: W&B project name.
        """
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues with spacy
            pin_memory=self.device == "cuda",
        )

        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        total_steps = len(self.train_loader) * num_epochs
        int(total_steps * warmup_ratio)

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=warmup_ratio,
            anneal_strategy="cos",
        )

        # Loss function with class weights (optional)
        self.criterion = nn.CrossEntropyLoss()

        # Tracking
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.global_step = 0

        # W&B
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb

                wandb.init(
                    project=wandb_project,
                    config={
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "num_epochs": num_epochs,
                        "weight_decay": weight_decay,
                    },
                )
            except ImportError:
                logger.warning("wandb not installed, disabling logging")
                self.use_wandb = False

    def train(self) -> dict:
        """Run the training loop.

        Returns:
            Training statistics.
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train samples: {len(self.train_dataset)}")
        if self.val_dataset:
            logger.info(f"Val samples: {len(self.val_dataset)}")

        train_losses = []
        val_losses = []
        val_accs = []

        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            # Training
            train_loss = self._train_epoch(epoch)
            train_losses.append(train_loss)

            # Validation
            val_loss = None
            val_acc = None
            if self.val_loader:
                val_loss, val_acc, val_report = self._validate(epoch)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                # Save best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self._save_checkpoint("best_model.pt")

            epoch_time = time.time() - epoch_start

            # Log
            log_msg = f"Epoch {epoch + 1}/{self.num_epochs} - "
            log_msg += f"Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2%}"
            log_msg += f" - Time: {epoch_time:.1f}s"
            logger.info(log_msg)

            # Save checkpoint
            self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        # Final evaluation
        final_stats = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "best_val_acc": self.best_val_acc,
        }

        # Save training stats
        with open(self.output_dir / "training_stats.json", "w") as f:
            json.dump(final_stats, f, indent=2)

        return final_stats

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            features = batch["features"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward
            logits = self.model(input_ids, attention_mask, features)
            loss = self.criterion(logits, labels)
            loss = loss / self.gradient_accumulation_steps

            # Backward
            loss.backward()

            # Update
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / num_batches
                logger.info(
                    f"  Epoch {epoch + 1} - Batch {batch_idx + 1}/{len(self.train_loader)} - "
                    f"Loss: {avg_loss:.4f}"
                )

        return total_loss / num_batches

    def _validate(self, epoch: int) -> tuple[float, float, str]:
        """Validate the model.

        Returns:
            Tuple of (loss, accuracy, classification_report)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask, features)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

        # Generate report
        report = classification_report(
            all_labels,
            all_preds,
            target_names=LABEL_NAMES,
        )

        # Log confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info(f"\nClassification Report:\n{report}")

        # W&B logging
        if self.use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "val_loss": avg_loss,
                    "val_accuracy": accuracy,
                }
            )

        return avg_loss, accuracy, report

    def _save_checkpoint(self, filename: str) -> None:
        """Save a checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_acc": self.best_val_acc,
        }

        path = self.output_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_acc = checkpoint["best_val_acc"]

        logger.info(f"Loaded checkpoint from {path}")
