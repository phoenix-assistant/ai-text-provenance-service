"""Provenance classifier ensemble model.

Combines:
1. Transformer encoder (DistilBERT/RoBERTa) for semantic understanding
2. MLP on handcrafted features (RST, linguistic, statistical)
3. Ensemble layer that fuses both

The key insight: RST features are robust to humanization because
they capture discourse structure, not surface text.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import logging

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ai_text_provenance.features.extractor import FeatureExtractor
from ai_text_provenance.models.schemas import (
    ClassificationResult,
    ProvenanceClass,
    AllFeatures,
)

logger = logging.getLogger(__name__)


class FeatureMLP(nn.Module):
    """MLP classifier for handcrafted features."""

    def __init__(
        self,
        input_dim: int = 47,  # Total features from FeatureExtractor
        hidden_dims: list[int] = [128, 64],
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            Tuple of (logits, feature_embedding)
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits, features


class TransformerEncoder(nn.Module):
    """Transformer-based text encoder."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.transformer.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            Tuple of (logits, pooled_output)
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(pooled_output)
        return logits, pooled_output


class EnsembleClassifier(nn.Module):
    """Ensemble model combining transformer and feature MLP."""

    def __init__(
        self,
        transformer_model: str = "distilbert-base-uncased",
        feature_dim: int = 47,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.transformer = TransformerEncoder(
            model_name=transformer_model,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.feature_mlp = FeatureMLP(
            input_dim=feature_dim,
            hidden_dims=[128, 64],
            num_classes=num_classes,
            dropout=dropout,
        )

        # Fusion layer
        transformer_hidden = self.transformer.hidden_size
        mlp_hidden = 64  # Last hidden dim of MLP

        self.fusion = nn.Sequential(
            nn.Linear(transformer_hidden + mlp_hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Learnable weights for ensemble
        self.ensemble_weights = nn.Parameter(torch.ones(3))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Tokenized text [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            features: Handcrafted features [batch_size, feature_dim]

        Returns:
            Logits [batch_size, num_classes]
        """
        # Get transformer predictions and embedding
        transformer_logits, transformer_embed = self.transformer(
            input_ids, attention_mask
        )

        # Get MLP predictions and embedding
        mlp_logits, mlp_embed = self.feature_mlp(features)

        # Fusion prediction
        combined_embed = torch.cat([transformer_embed, mlp_embed], dim=1)
        fusion_logits = self.fusion(combined_embed)

        # Weighted ensemble
        weights = torch.softmax(self.ensemble_weights, dim=0)
        ensemble_logits = (
            weights[0] * transformer_logits
            + weights[1] * mlp_logits
            + weights[2] * fusion_logits
        )

        return ensemble_logits


class ProvenanceClassifier:
    """Main classifier interface for text provenance detection.

    This is the primary class users interact with. It handles:
    - Model loading and initialization
    - Feature extraction
    - Inference (single and batch)
    - ONNX export for production
    """

    # Class labels
    CLASSES = [
        ProvenanceClass.HUMAN,
        ProvenanceClass.AI,
        ProvenanceClass.POLISHED_HUMAN,
        ProvenanceClass.HUMANIZED_AI,
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        transformer_model: str = "distilbert-base-uncased",
        device: Optional[str] = None,
        use_onnx: bool = False,
    ):
        """Initialize the classifier.

        Args:
            model_path: Path to saved model weights. If None, uses untrained model.
            transformer_model: HuggingFace model name for transformer encoder.
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.
            use_onnx: Use ONNX runtime for inference (faster).
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

        logger.info(f"Using device: {self.device}")

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)

        self.use_onnx = use_onnx
        self.onnx_session = None

        if use_onnx and model_path:
            self._load_onnx(model_path)
        else:
            # Initialize PyTorch model
            self.model = EnsembleClassifier(
                transformer_model=transformer_model,
                feature_dim=self.feature_extractor.num_features,
                num_classes=len(self.CLASSES),
            )

            if model_path:
                self._load_weights(model_path)

            self.model = self.model.to(self.device)
            self.model.eval()

    def _load_weights(self, model_path: str) -> None:
        """Load model weights from file."""
        path = Path(model_path)

        if not path.exists():
            logger.warning(f"Model path {model_path} not found, using untrained model")
            return

        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        logger.info(f"Loaded model weights from {model_path}")

    def _load_onnx(self, model_path: str) -> None:
        """Load ONNX model for inference."""
        try:
            import onnxruntime as ort

            # Check for GPU provider
            providers = ["CPUExecutionProvider"]
            if self.device == "cuda":
                providers.insert(0, "CUDAExecutionProvider")

            self.onnx_session = ort.InferenceSession(model_path, providers=providers)
            logger.info(f"Loaded ONNX model from {model_path}")
        except ImportError:
            logger.error("ONNX Runtime not installed. Install with: pip install onnxruntime")
            raise

    def classify(
        self, text: str, include_features: bool = False
    ) -> ClassificationResult:
        """Classify a single text.

        Args:
            text: Input text to classify.
            include_features: Whether to include extracted features in result.

        Returns:
            ClassificationResult with prediction and probabilities.
        """
        # Extract features
        features = self.feature_extractor.extract(text)
        feature_vector = self.feature_extractor.to_vector(features)

        # Tokenize text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if self.use_onnx and self.onnx_session:
            logits = self._inference_onnx(encoding, feature_vector)
        else:
            logits = self._inference_pytorch(encoding, feature_vector)

        # Convert to probabilities
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

        # Get prediction
        pred_idx = np.argmax(probs)
        prediction = self.CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        # Build probability dict
        probabilities = {
            cls.value: float(probs[i]) for i, cls in enumerate(self.CLASSES)
        }

        return ClassificationResult(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            features=features if include_features else None,
        )

    def classify_batch(
        self, texts: list[str], include_features: bool = False
    ) -> list[ClassificationResult]:
        """Classify multiple texts.

        Args:
            texts: List of texts to classify.
            include_features: Whether to include extracted features.

        Returns:
            List of ClassificationResult objects.
        """
        results = []

        # Extract features for all texts
        all_features = [self.feature_extractor.extract(text) for text in texts]
        feature_vectors = [
            self.feature_extractor.to_vector(f) for f in all_features
        ]

        # Tokenize all texts
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if self.use_onnx and self.onnx_session:
            logits = self._inference_onnx_batch(encodings, feature_vectors)
        else:
            logits = self._inference_pytorch_batch(encodings, feature_vectors)

        # Process each result
        for i, (text, features) in enumerate(zip(texts, all_features)):
            probs = torch.softmax(torch.tensor(logits[i]), dim=-1).numpy()

            pred_idx = np.argmax(probs)
            prediction = self.CLASSES[pred_idx]
            confidence = float(probs[pred_idx])

            probabilities = {
                cls.value: float(probs[j]) for j, cls in enumerate(self.CLASSES)
            }

            results.append(
                ClassificationResult(
                    prediction=prediction,
                    confidence=confidence,
                    probabilities=probabilities,
                    features=features if include_features else None,
                )
            )

        return results

    def _inference_pytorch(
        self, encoding: dict, feature_vector: list[float]
    ) -> np.ndarray:
        """Run inference with PyTorch model."""
        with torch.no_grad():
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            features = torch.tensor([feature_vector], dtype=torch.float32).to(
                self.device
            )

            logits = self.model(input_ids, attention_mask, features)
            return logits.cpu().numpy()[0]

    def _inference_pytorch_batch(
        self, encodings: dict, feature_vectors: list[list[float]]
    ) -> np.ndarray:
        """Run batch inference with PyTorch model."""
        with torch.no_grad():
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            features = torch.tensor(feature_vectors, dtype=torch.float32).to(
                self.device
            )

            logits = self.model(input_ids, attention_mask, features)
            return logits.cpu().numpy()

    def _inference_onnx(
        self, encoding: dict, feature_vector: list[float]
    ) -> np.ndarray:
        """Run inference with ONNX Runtime."""
        inputs = {
            "input_ids": encoding["input_ids"].numpy(),
            "attention_mask": encoding["attention_mask"].numpy(),
            "features": np.array([feature_vector], dtype=np.float32),
        }

        outputs = self.onnx_session.run(None, inputs)
        return outputs[0][0]

    def _inference_onnx_batch(
        self, encodings: dict, feature_vectors: list[list[float]]
    ) -> np.ndarray:
        """Run batch inference with ONNX Runtime."""
        inputs = {
            "input_ids": encodings["input_ids"].numpy(),
            "attention_mask": encodings["attention_mask"].numpy(),
            "features": np.array(feature_vectors, dtype=np.float32),
        }

        outputs = self.onnx_session.run(None, inputs)
        return outputs[0]

    def export_onnx(self, output_path: str) -> None:
        """Export model to ONNX format for production inference.

        Args:
            output_path: Path to save ONNX model.
        """
        self.model.eval()

        # Create dummy inputs
        dummy_input_ids = torch.randint(0, 1000, (1, 512)).to(self.device)
        dummy_attention_mask = torch.ones(1, 512).to(self.device)
        dummy_features = torch.randn(1, self.feature_extractor.num_features).to(
            self.device
        )

        # Export
        torch.onnx.export(
            self.model,
            (dummy_input_ids, dummy_attention_mask, dummy_features),
            output_path,
            input_names=["input_ids", "attention_mask", "features"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "features": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=14,
        )

        logger.info(f"Exported ONNX model to {output_path}")

    def save(self, output_path: str) -> None:
        """Save model weights.

        Args:
            output_path: Path to save model weights.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved model weights to {output_path}")
