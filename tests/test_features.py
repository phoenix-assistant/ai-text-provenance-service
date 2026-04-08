"""Tests for feature extraction."""

import pytest

from ai_text_provenance.features.extractor import FeatureExtractor
from ai_text_provenance.features.linguistic import LinguisticExtractor
from ai_text_provenance.features.statistical import StatisticalExtractor


@pytest.fixture
def extractor():
    """Create a full feature extractor."""
    return FeatureExtractor()


@pytest.fixture
def linguistic():
    """Create a linguistic extractor."""
    return LinguisticExtractor()


@pytest.fixture
def statistical():
    """Create a statistical extractor."""
    return StatisticalExtractor()


class TestFeatureExtractor:
    """Test the unified feature extractor."""

    def test_extract_all_features(self, extractor):
        """Test extracting all feature types."""
        text = """
        Machine learning has revolutionized many fields.
        From healthcare to finance, its applications are vast.
        However, with great power comes great responsibility.
        We must ensure these systems are fair and transparent.
        """

        features = extractor.extract(text)

        # Check all components present
        assert features.rst is not None
        assert features.linguistic is not None
        assert features.statistical is not None

    def test_to_vector(self, extractor):
        """Test conversion to feature vector."""
        text = "This is a test sentence for feature extraction."

        features = extractor.extract(text)
        vector = extractor.to_vector(features)

        # Check vector length matches expected
        assert len(vector) == extractor.num_features
        assert len(vector) == len(extractor.feature_names)

        # All values should be numeric
        for v in vector:
            assert isinstance(v, (int, float))

    def test_feature_names(self, extractor):
        """Test feature names are correct."""
        names = extractor.feature_names

        # Check some expected names
        assert "rst_num_edus" in names
        assert "ling_num_sentences" in names
        assert "stat_perplexity" in names


class TestLinguisticExtractor:
    """Test linguistic feature extraction."""

    def test_sentence_statistics(self, linguistic):
        """Test sentence-level statistics."""
        text = """
        Short sentence. This is a medium length sentence here.
        This sentence is considerably longer and contains more words than the others.
        """

        features = linguistic.extract_features(text)

        assert features.num_sentences == 3
        assert features.avg_sentence_length > 0
        assert features.sentence_length_variance > 0

    def test_vocabulary_diversity(self, linguistic):
        """Test vocabulary diversity metrics."""
        # Low diversity (repetitive)
        repetitive = "The the the the cat cat cat sat sat sat."

        # High diversity (varied)
        varied = "The quick brown fox jumps over the lazy sleeping dog."

        rep_features = linguistic.extract_features(repetitive)
        var_features = linguistic.extract_features(varied)

        # Varied text should have higher TTR
        assert var_features.type_token_ratio > rep_features.type_token_ratio

    def test_transition_words(self, linguistic):
        """Test transition word detection."""
        # Text with transitions
        with_transitions = """
        First, we analyze the data. However, the results were unexpected.
        Therefore, we revised our approach. Consequently, the model improved.
        """

        # Text without transitions
        without_transitions = """
        We analyze the data. The results came back. We revised the approach.
        The model showed changes.
        """

        with_features = linguistic.extract_features(with_transitions)
        without_features = linguistic.extract_features(without_transitions)

        assert with_features.transition_words_ratio > without_features.transition_words_ratio

    def test_pos_distribution(self, linguistic):
        """Test POS tag distribution."""
        text = "The big red dog ran quickly through the beautiful garden."

        features = linguistic.extract_features(text)

        # Check ratios sum to approximately 1
        total = (
            features.noun_ratio +
            features.verb_ratio +
            features.adjective_ratio +
            features.adverb_ratio
        )
        # Won't exactly equal 1 due to other POS tags
        assert 0 < total < 1

    def test_readability_scores(self, linguistic):
        """Test readability calculations."""
        # Simple text
        simple = "I like cats. Cats are nice. They are soft."

        # Complex text
        complex_text = """
        The epistemological implications of quantum mechanical phenomena
        necessitate a fundamental reconsideration of classical deterministic
        paradigms in contemporary philosophical discourse.
        """

        simple_features = linguistic.extract_features(simple)
        complex_features = linguistic.extract_features(complex_text)

        # Simple should have higher Flesch score (easier)
        assert simple_features.flesch_reading_ease > complex_features.flesch_reading_ease

        # Complex should have higher grade level
        assert complex_features.flesch_kincaid_grade > simple_features.flesch_kincaid_grade


class TestStatisticalExtractor:
    """Test statistical feature extraction."""

    def test_entropy_measures(self, statistical):
        """Test entropy calculations."""
        text = """
        The algorithm processes data efficiently.
        Machine learning models require substantial training data.
        Neural networks have multiple layers of computation.
        """

        features = statistical.extract_features(text)

        assert features.word_entropy > 0
        assert features.bigram_entropy > 0
        assert features.trigram_entropy > 0

    def test_repetition_detection(self, statistical):
        """Test repetition score calculation."""
        # Highly repetitive
        repetitive = """
        The model works well. The model performs well.
        The model functions well. The model operates well.
        """

        # Non-repetitive
        varied = """
        Machine learning transforms industries.
        Natural language processing enables communication.
        Computer vision recognizes objects.
        Deep learning powers modern AI.
        """

        rep_features = statistical.extract_features(repetitive)
        var_features = statistical.extract_features(varied)

        assert rep_features.exact_repetition_score > var_features.exact_repetition_score

    def test_zipf_coefficient(self, statistical):
        """Test Zipf's law coefficient."""
        text = """
        The quick brown fox jumps over the lazy dog.
        A wizard's job is to vex chumps quickly in fog.
        Pack my box with five dozen liquor jugs.
        """

        features = statistical.extract_features(text)

        # Zipf coefficient typically around 1.0 for natural language
        assert 0.5 < features.zipf_coefficient < 2.0

    def test_heaps_coefficient(self, statistical):
        """Test Heaps' law coefficient."""
        text = " ".join([
            "Machine learning is transforming how we work.",
            "Algorithms process vast amounts of data daily.",
            "Natural language understanding has improved dramatically.",
            "Computer vision systems can identify objects accurately.",
            "Deep neural networks power modern artificial intelligence.",
        ] * 3)

        features = statistical.extract_features(text)

        # Heaps coefficient typically between 0.4 and 0.6
        assert 0.3 < features.heaps_coefficient < 0.8

    def test_predictability(self, statistical):
        """Test predictability scores."""
        text = """
        The sun rises in the east and sets in the west.
        Water flows downhill due to gravity.
        Trees produce oxygen through photosynthesis.
        """

        features = statistical.extract_features(text)

        assert 0 <= features.predictability_score <= 1
        assert 0 <= features.surprise_score <= 1
        # They should be roughly inversely related
        assert abs(features.predictability_score + features.surprise_score - 1.0) < 0.5

    def test_short_text_handling(self, statistical):
        """Test handling of very short text."""
        features = statistical.extract_features("Hi there")

        # Should return defaults without crashing
        assert features is not None
        assert features.perplexity > 0
