"""Tests for the classifier model."""

import pytest
import torch

from ai_text_provenance.models.classifier import (
    ProvenanceClassifier,
    EnsembleClassifier,
    FeatureMLP,
    TransformerEncoder,
)
from ai_text_provenance.models.schemas import ProvenanceClass


@pytest.fixture
def classifier():
    """Create an untrained classifier."""
    return ProvenanceClassifier(device="cpu")


class TestFeatureMLP:
    """Test the feature MLP component."""

    def test_forward_pass(self):
        """Test MLP forward pass."""
        model = FeatureMLP(input_dim=47, hidden_dims=[64, 32], num_classes=4)

        # Batch of 4 samples
        x = torch.randn(4, 47)
        logits, features = model(x)

        assert logits.shape == (4, 4)
        assert features.shape == (4, 32)

    def test_different_configs(self):
        """Test MLP with different configurations."""
        # Deeper network
        model = FeatureMLP(input_dim=50, hidden_dims=[128, 64, 32], num_classes=4)
        x = torch.randn(2, 50)
        logits, _ = model(x)
        assert logits.shape == (2, 4)

        # Wider network
        model = FeatureMLP(input_dim=50, hidden_dims=[256], num_classes=4)
        logits, _ = model(x)
        assert logits.shape == (2, 4)


class TestTransformerEncoder:
    """Test the transformer encoder component."""

    def test_forward_pass(self):
        """Test transformer forward pass."""
        model = TransformerEncoder(model_name="distilbert-base-uncased", num_classes=4)

        # Batch of 2 samples
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128)

        logits, pooled = model(input_ids, attention_mask)

        assert logits.shape == (2, 4)
        assert pooled.shape[0] == 2


class TestEnsembleClassifier:
    """Test the ensemble classifier."""

    def test_forward_pass(self):
        """Test ensemble forward pass."""
        model = EnsembleClassifier(
            transformer_model="distilbert-base-uncased",
            feature_dim=47,
            num_classes=4,
        )

        # Batch of 2 samples
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128)
        features = torch.randn(2, 47)

        logits = model(input_ids, attention_mask, features)

        assert logits.shape == (2, 4)

    def test_ensemble_weights(self):
        """Test that ensemble weights are learnable."""
        model = EnsembleClassifier(
            transformer_model="distilbert-base-uncased",
            feature_dim=47,
            num_classes=4,
        )

        # Check weights exist and are parameters
        assert hasattr(model, "ensemble_weights")
        assert model.ensemble_weights.requires_grad


class TestProvenanceClassifier:
    """Test the main classifier interface."""

    def test_classify_single_text(self, classifier):
        """Test single text classification."""
        text = """
        Machine learning is a branch of artificial intelligence.
        It enables computers to learn from data without explicit programming.
        Many applications use machine learning today.
        """

        result = classifier.classify(text)

        assert result.prediction in list(ProvenanceClass)
        assert 0 <= result.confidence <= 1
        assert len(result.probabilities) == 4
        assert sum(result.probabilities.values()) == pytest.approx(1.0, rel=0.01)

    def test_classify_with_features(self, classifier):
        """Test classification with feature extraction."""
        text = """
        This is a sample text for testing feature extraction.
        It includes multiple sentences to ensure proper analysis.
        The features should be computed and returned.
        """

        result = classifier.classify(text, include_features=True)

        assert result.features is not None
        assert result.features.rst is not None
        assert result.features.linguistic is not None
        assert result.features.statistical is not None

    def test_classify_batch(self, classifier):
        """Test batch classification."""
        texts = [
            "First text for batch classification testing with enough content.",
            "Second text that is also long enough for the classifier.",
            "Third text to test batch processing functionality properly.",
        ]

        results = classifier.classify_batch(texts)

        assert len(results) == 3
        for result in results:
            assert result.prediction in list(ProvenanceClass)
            assert 0 <= result.confidence <= 1

    def test_classification_consistency(self, classifier):
        """Test that same text gives same result."""
        text = "This is a consistent test text that should give the same classification result each time it is processed."

        result1 = classifier.classify(text)
        result2 = classifier.classify(text)

        assert result1.prediction == result2.prediction
        assert result1.confidence == pytest.approx(result2.confidence, rel=0.01)

    def test_probabilities_sum_to_one(self, classifier):
        """Test that probabilities sum to 1."""
        text = "Testing that the probability distribution is valid and sums to one."

        result = classifier.classify(text)

        total = sum(result.probabilities.values())
        assert total == pytest.approx(1.0, rel=0.01)

    def test_long_text_handling(self, classifier):
        """Test handling of long text (truncation)."""
        # Generate text longer than max_length (512 tokens)
        text = " ".join(["This is a test sentence."] * 200)

        # Should not crash
        result = classifier.classify(text)

        assert result.prediction in list(ProvenanceClass)
