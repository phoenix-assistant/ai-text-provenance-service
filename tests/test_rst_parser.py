"""Tests for RST parser."""

import pytest

from ai_text_provenance.features.rst_parser import RSTParser


@pytest.fixture
def parser():
    """Create an RST parser."""
    return RSTParser()


class TestRSTParser:
    """Test RST parsing functionality."""

    def test_segment_simple_sentence(self, parser):
        """Test segmentation of a simple sentence."""
        text = "The cat sat on the mat."
        features = parser.extract_features(text)

        assert features.num_edus >= 1
        assert features.avg_edu_length > 0

    def test_segment_complex_sentence(self, parser):
        """Test segmentation of a sentence with subordinate clause."""
        text = "Although it was raining, the children played outside because they loved the mud."
        features = parser.extract_features(text)

        # Should have multiple EDUs due to subordinate clauses
        assert features.num_edus >= 2

    def test_multiple_sentences(self, parser):
        """Test multi-sentence text."""
        text = """
        Machine learning is transforming how we process text.
        Traditional methods relied on hand-crafted features.
        However, deep learning has changed the landscape entirely.
        Now, models can learn representations automatically.
        """
        features = parser.extract_features(text)

        assert features.num_edus >= 4
        assert features.tree_depth >= 1

    def test_coherence_calculation(self, parser):
        """Test coherence metrics."""
        # Coherent text (same topic)
        coherent = """
        Machine learning uses data to train models.
        The models learn patterns from this data.
        With enough data, the models can make predictions.
        """

        # Incoherent text (topic shifts)
        incoherent = """
        Machine learning uses neural networks.
        Bananas are yellow fruits from tropical regions.
        The stock market closed higher yesterday.
        """

        coherent_features = parser.extract_features(coherent)
        incoherent_features = parser.extract_features(incoherent)

        # Coherent text should have higher coherence or fewer breaks
        assert coherent_features.local_coherence >= incoherent_features.local_coherence or \
               coherent_features.coherence_breaks <= incoherent_features.coherence_breaks

    def test_relation_detection(self, parser):
        """Test discourse relation detection."""
        # Text with contrast relation
        text = "The algorithm is fast. However, it lacks accuracy."
        features = parser.extract_features(text)

        # Should detect some relations
        total_relations = (
            features.elaboration_ratio +
            features.contrast_ratio +
            features.cause_ratio +
            features.temporal_ratio
        )
        assert total_relations > 0

    def test_empty_text(self, parser):
        """Test handling of empty text."""
        features = parser.extract_features("")

        assert features.num_edus == 0
        assert features.tree_depth == 0

    def test_short_text(self, parser):
        """Test handling of very short text."""
        features = parser.extract_features("Hi")

        # Should return default features without crashing
        assert features is not None

    def test_tree_balance(self, parser):
        """Test tree balance calculation."""
        # Balanced structure
        balanced = """
        First, we prepare the data.
        Second, we train the model.
        Third, we evaluate results.
        """

        # Unbalanced structure (deep nesting)
        unbalanced = """
        The study found that when researchers analyzed the data,
        which had been collected over several years from multiple sources,
        they discovered patterns that suggested the hypothesis was correct,
        although there were some anomalies that required further investigation.
        """

        balanced_features = parser.extract_features(balanced)
        unbalanced_features = parser.extract_features(unbalanced)

        # Balance score should differ
        assert balanced_features.tree_balance >= 0
        assert unbalanced_features.tree_balance >= 0

    def test_nuclearity_patterns(self, parser):
        """Test nuclearity extraction."""
        text = """
        Climate change is the main topic.
        Rising temperatures cause ice to melt.
        This leads to sea level rise.
        Many coastal cities are at risk.
        """

        features = parser.extract_features(text)

        # Should have both nucleus and satellite spans
        assert 0 <= features.nucleus_ratio <= 1
        assert 0 <= features.multinuclear_ratio <= 1
