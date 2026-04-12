"""Unified feature extractor combining all feature types.

This module orchestrates RST, linguistic, and statistical feature extraction
into a single interface used by the classifier.
"""

from __future__ import annotations

import spacy

from ai_text_provenance.features.linguistic import LinguisticExtractor
from ai_text_provenance.features.rst_parser import RSTParser
from ai_text_provenance.features.statistical import StatisticalExtractor
from ai_text_provenance.models.schemas import (
    AllFeatures,
    LinguisticFeatures,
    RSTFeatures,
    StatisticalFeatures,
)


class FeatureExtractor:
    """Unified feature extraction for text provenance classification.

    Combines RST discourse analysis, linguistic patterns, and
    statistical features into a comprehensive feature set.
    """

    def __init__(self, nlp: spacy.Language | None = None):
        """Initialize feature extractors.

        Args:
            nlp: SpaCy language model. If None, loads en_core_web_sm.
        """
        if nlp is None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess

                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp

        # Initialize individual extractors with shared NLP model
        self.rst_parser = RSTParser(nlp=self.nlp)
        self.linguistic_extractor = LinguisticExtractor(nlp=self.nlp)
        self.statistical_extractor = StatisticalExtractor(nlp=self.nlp)

    def extract(self, text: str) -> AllFeatures:
        """Extract all features from text.

        Args:
            text: Input text to analyze.

        Returns:
            AllFeatures containing RST, linguistic, and statistical features.
        """
        # Extract each feature type
        rst_features = self.rst_parser.extract_features(text)
        linguistic_features = self.linguistic_extractor.extract_features(text)
        statistical_features = self.statistical_extractor.extract_features(text)

        return AllFeatures(
            rst=rst_features,
            linguistic=linguistic_features,
            statistical=statistical_features,
        )

    def extract_rst(self, text: str) -> RSTFeatures:
        """Extract only RST features."""
        return self.rst_parser.extract_features(text)

    def extract_linguistic(self, text: str) -> LinguisticFeatures:
        """Extract only linguistic features."""
        return self.linguistic_extractor.extract_features(text)

    def extract_statistical(self, text: str) -> StatisticalFeatures:
        """Extract only statistical features."""
        return self.statistical_extractor.extract_features(text)

    def to_vector(self, features: AllFeatures) -> list[float]:
        """Convert features to a flat feature vector for the classifier.

        Args:
            features: All extracted features.

        Returns:
            List of float values suitable for model input.
        """
        vector = []

        # RST features (19 values)
        rst = features.rst
        vector.extend(
            [
                float(rst.num_edus),
                rst.avg_edu_length,
                rst.edu_length_variance,
                float(rst.tree_depth),
                rst.tree_depth_avg,
                rst.tree_balance,
                rst.nucleus_ratio,
                rst.multinuclear_ratio,
                rst.elaboration_ratio,
                rst.contrast_ratio,
                rst.cause_ratio,
                rst.temporal_ratio,
                rst.attribution_ratio,
                rst.condition_ratio,
                rst.local_coherence,
                rst.global_coherence,
                float(rst.coherence_breaks),
            ]
        )

        # Linguistic features (18 values)
        ling = features.linguistic
        vector.extend(
            [
                float(ling.num_sentences),
                ling.avg_sentence_length,
                ling.sentence_length_variance,
                ling.sentence_length_entropy,
                ling.type_token_ratio,
                ling.hapax_ratio,
                ling.vocabulary_richness,
                ling.avg_tree_depth,
                ling.subordination_ratio,
                ling.coordination_ratio,
                ling.transition_words_ratio,
                ling.transition_variety,
                ling.noun_ratio,
                ling.verb_ratio,
                ling.adjective_ratio,
                ling.adverb_ratio,
                ling.flesch_reading_ease,
                ling.flesch_kincaid_grade,
            ]
        )

        # Statistical features (12 values)
        stat = features.statistical
        vector.extend(
            [
                stat.perplexity,
                stat.perplexity_variance,
                stat.perplexity_burstiness,
                stat.word_entropy,
                stat.bigram_entropy,
                stat.trigram_entropy,
                stat.zipf_coefficient,
                stat.heaps_coefficient,
                stat.exact_repetition_score,
                stat.semantic_repetition_score,
                stat.predictability_score,
                stat.surprise_score,
            ]
        )

        return vector

    @property
    def feature_names(self) -> list[str]:
        """Get names of all features in vector order."""
        return [
            # RST features
            "rst_num_edus",
            "rst_avg_edu_length",
            "rst_edu_length_variance",
            "rst_tree_depth",
            "rst_tree_depth_avg",
            "rst_tree_balance",
            "rst_nucleus_ratio",
            "rst_multinuclear_ratio",
            "rst_elaboration_ratio",
            "rst_contrast_ratio",
            "rst_cause_ratio",
            "rst_temporal_ratio",
            "rst_attribution_ratio",
            "rst_condition_ratio",
            "rst_local_coherence",
            "rst_global_coherence",
            "rst_coherence_breaks",
            # Linguistic features
            "ling_num_sentences",
            "ling_avg_sentence_length",
            "ling_sentence_length_variance",
            "ling_sentence_length_entropy",
            "ling_type_token_ratio",
            "ling_hapax_ratio",
            "ling_vocabulary_richness",
            "ling_avg_tree_depth",
            "ling_subordination_ratio",
            "ling_coordination_ratio",
            "ling_transition_words_ratio",
            "ling_transition_variety",
            "ling_noun_ratio",
            "ling_verb_ratio",
            "ling_adjective_ratio",
            "ling_adverb_ratio",
            "ling_flesch_reading_ease",
            "ling_flesch_kincaid_grade",
            # Statistical features
            "stat_perplexity",
            "stat_perplexity_variance",
            "stat_perplexity_burstiness",
            "stat_word_entropy",
            "stat_bigram_entropy",
            "stat_trigram_entropy",
            "stat_zipf_coefficient",
            "stat_heaps_coefficient",
            "stat_exact_repetition_score",
            "stat_semantic_repetition_score",
            "stat_predictability_score",
            "stat_surprise_score",
        ]

    @property
    def num_features(self) -> int:
        """Total number of features in the vector."""
        return len(self.feature_names)
