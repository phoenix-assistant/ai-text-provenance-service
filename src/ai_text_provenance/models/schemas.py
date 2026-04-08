"""Data models and schemas for the provenance service."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ProvenanceClass(str, Enum):
    """Classification categories for text provenance."""

    HUMAN = "human"
    AI = "ai"
    POLISHED_HUMAN = "polished_human"
    HUMANIZED_AI = "humanized_ai"


class RSTFeatures(BaseModel):
    """Rhetorical Structure Theory features.

    These capture discourse-level patterns that are difficult for
    humanizer tools to modify without fundamentally changing the text.
    """

    # Discourse unit statistics
    num_edus: int = Field(description="Number of Elementary Discourse Units")
    avg_edu_length: float = Field(description="Average EDU length in tokens")
    edu_length_variance: float = Field(description="Variance in EDU lengths")

    # Tree structure metrics
    tree_depth: int = Field(description="Maximum depth of RST tree")
    tree_depth_avg: float = Field(description="Average depth across branches")
    tree_balance: float = Field(description="Balance score (0=unbalanced, 1=balanced)")

    # Nuclearity patterns
    nucleus_ratio: float = Field(description="Ratio of nucleus to satellite spans")
    multinuclear_ratio: float = Field(description="Ratio of multinuclear relations")

    # Relation type distribution
    elaboration_ratio: float = Field(description="Proportion of elaboration relations")
    contrast_ratio: float = Field(description="Proportion of contrast relations")
    cause_ratio: float = Field(description="Proportion of cause/result relations")
    temporal_ratio: float = Field(description="Proportion of temporal relations")
    attribution_ratio: float = Field(description="Proportion of attribution relations")
    condition_ratio: float = Field(description="Proportion of condition relations")

    # Coherence metrics
    local_coherence: float = Field(description="Adjacent sentence coherence score")
    global_coherence: float = Field(description="Document-level coherence score")
    coherence_breaks: int = Field(description="Number of coherence discontinuities")


class LinguisticFeatures(BaseModel):
    """Linguistic and stylistic features."""

    # Sentence-level statistics
    num_sentences: int
    avg_sentence_length: float
    sentence_length_variance: float
    sentence_length_entropy: float = Field(description="Entropy of sentence lengths (burstiness)")

    # Vocabulary diversity
    type_token_ratio: float
    hapax_ratio: float = Field(description="Ratio of words appearing only once")
    vocabulary_richness: float = Field(description="MATTR moving average TTR")

    # Syntactic complexity
    avg_tree_depth: float = Field(description="Average parse tree depth")
    subordination_ratio: float = Field(description="Ratio of subordinate clauses")
    coordination_ratio: float = Field(description="Ratio of coordinate structures")

    # Transition patterns
    transition_words_ratio: float
    transition_variety: float = Field(description="Entropy of transition word types")

    # POS distribution
    noun_ratio: float
    verb_ratio: float
    adjective_ratio: float
    adverb_ratio: float

    # Readability
    flesch_reading_ease: float
    flesch_kincaid_grade: float


class StatisticalFeatures(BaseModel):
    """Statistical and information-theoretic features."""

    # Perplexity-based (traditional AI detection)
    perplexity: float = Field(description="GPT-2 perplexity score")
    perplexity_variance: float = Field(description="Variance in per-sentence perplexity")
    perplexity_burstiness: float = Field(description="Burstiness of perplexity scores")

    # Entropy measures
    word_entropy: float
    bigram_entropy: float
    trigram_entropy: float

    # Distribution characteristics
    zipf_coefficient: float = Field(description="Zipf's law coefficient")
    heaps_coefficient: float = Field(description="Heaps' law coefficient")

    # Repetition patterns
    exact_repetition_score: float = Field(description="Exact phrase repetition")
    semantic_repetition_score: float = Field(description="Semantic similarity repetition")

    # Predictability
    predictability_score: float = Field(description="How predictable next words are")
    surprise_score: float = Field(description="Average surprisal per word")


class AllFeatures(BaseModel):
    """Combined features from all extractors."""

    rst: RSTFeatures
    linguistic: LinguisticFeatures
    statistical: StatisticalFeatures


class ClassificationResult(BaseModel):
    """Result of a text classification."""

    prediction: ProvenanceClass
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[ProvenanceClass, float]
    features: Optional[AllFeatures] = None

    class Config:
        use_enum_values = True


class ClassifyRequest(BaseModel):
    """Request to classify a single text."""

    text: str = Field(min_length=50, max_length=100000)
    include_features: bool = False


class ClassifyBatchRequest(BaseModel):
    """Request to classify multiple texts."""

    texts: list[str] = Field(min_length=1, max_length=100)
    include_features: bool = False


class ClassifyResponse(BaseModel):
    """Response for single text classification."""

    prediction: str
    confidence: float
    probabilities: dict[str, float]
    features: Optional[dict] = None


class ClassifyBatchResponse(BaseModel):
    """Response for batch classification."""

    results: list[ClassifyResponse]
    total: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    model_loaded: bool
    spacy_loaded: bool
