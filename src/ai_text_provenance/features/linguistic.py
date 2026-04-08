"""Linguistic and stylistic feature extraction.

Captures writing style patterns that differ between human and AI text:
- Sentence structure variety
- Vocabulary diversity
- Syntactic complexity
- Transition patterns
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import spacy
from spacy.tokens import Doc

from ai_text_provenance.models.schemas import LinguisticFeatures


# Common transition words by category
TRANSITION_WORDS = {
    "addition": ["also", "furthermore", "moreover", "additionally", "besides", "too"],
    "contrast": ["however", "but", "although", "nevertheless", "yet", "whereas", "while"],
    "cause": ["because", "since", "therefore", "thus", "consequently", "hence", "so"],
    "example": ["for example", "for instance", "such as", "specifically", "namely"],
    "time": ["then", "next", "finally", "first", "second", "meanwhile", "afterward"],
    "emphasis": ["indeed", "certainly", "clearly", "obviously", "importantly"],
    "conclusion": ["finally", "in conclusion", "ultimately", "overall", "in summary"],
}

# Flatten for quick lookup
ALL_TRANSITIONS = set()
for words in TRANSITION_WORDS.values():
    ALL_TRANSITIONS.update(words)


class LinguisticExtractor:
    """Extract linguistic and stylistic features from text."""

    def __init__(self, nlp: Optional[spacy.Language] = None):
        """Initialize the linguistic extractor.

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

    def extract_features(self, text: str) -> LinguisticFeatures:
        """Extract linguistic features from text.

        Args:
            text: Input text to analyze.

        Returns:
            LinguisticFeatures with all metrics.
        """
        doc = self.nlp(text)

        # Sentence statistics
        sentences = list(doc.sents)
        num_sentences = len(sentences)

        if num_sentences == 0:
            return self._default_features()

        sentence_lengths = [len([t for t in s if not t.is_punct]) for s in sentences]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
        sentence_length_variance = (
            sum((l - avg_sentence_length) ** 2 for l in sentence_lengths)
            / len(sentence_lengths)
            if len(sentence_lengths) > 1
            else 0.0
        )
        sentence_length_entropy = self._calculate_entropy(sentence_lengths)

        # Vocabulary diversity
        tokens = [t.text.lower() for t in doc if t.is_alpha]
        if not tokens:
            return self._default_features()

        type_token_ratio = len(set(tokens)) / len(tokens) if tokens else 0.0
        hapax_ratio = self._calculate_hapax_ratio(tokens)
        vocabulary_richness = self._calculate_mattr(tokens)

        # Syntactic complexity
        avg_tree_depth = self._calculate_avg_tree_depth(doc)
        subordination_ratio = self._calculate_subordination_ratio(doc)
        coordination_ratio = self._calculate_coordination_ratio(doc)

        # Transition patterns
        text_lower = text.lower()
        transition_count = sum(1 for t in ALL_TRANSITIONS if t in text_lower)
        word_count = len(tokens)
        transition_words_ratio = transition_count / word_count if word_count > 0 else 0.0
        transition_variety = self._calculate_transition_variety(text_lower)

        # POS distribution
        pos_counts = Counter(t.pos_ for t in doc if t.is_alpha)
        total_pos = sum(pos_counts.values()) or 1

        noun_ratio = pos_counts.get("NOUN", 0) / total_pos
        verb_ratio = pos_counts.get("VERB", 0) / total_pos
        adjective_ratio = pos_counts.get("ADJ", 0) / total_pos
        adverb_ratio = pos_counts.get("ADV", 0) / total_pos

        # Readability scores
        flesch_reading_ease = self._flesch_reading_ease(doc, sentences)
        flesch_kincaid_grade = self._flesch_kincaid_grade(doc, sentences)

        return LinguisticFeatures(
            num_sentences=num_sentences,
            avg_sentence_length=avg_sentence_length,
            sentence_length_variance=sentence_length_variance,
            sentence_length_entropy=sentence_length_entropy,
            type_token_ratio=type_token_ratio,
            hapax_ratio=hapax_ratio,
            vocabulary_richness=vocabulary_richness,
            avg_tree_depth=avg_tree_depth,
            subordination_ratio=subordination_ratio,
            coordination_ratio=coordination_ratio,
            transition_words_ratio=transition_words_ratio,
            transition_variety=transition_variety,
            noun_ratio=noun_ratio,
            verb_ratio=verb_ratio,
            adjective_ratio=adjective_ratio,
            adverb_ratio=adverb_ratio,
            flesch_reading_ease=flesch_reading_ease,
            flesch_kincaid_grade=flesch_kincaid_grade,
        )

    def _calculate_entropy(self, values: list[int]) -> float:
        """Calculate Shannon entropy of a distribution."""
        if not values:
            return 0.0

        # Bin values for entropy calculation
        counter = Counter(values)
        total = sum(counter.values())

        entropy = 0.0
        for count in counter.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)

        return entropy

    def _calculate_hapax_ratio(self, tokens: list[str]) -> float:
        """Calculate ratio of words appearing only once (hapax legomena)."""
        if not tokens:
            return 0.0

        counter = Counter(tokens)
        hapax_count = sum(1 for count in counter.values() if count == 1)
        return hapax_count / len(set(tokens)) if tokens else 0.0

    def _calculate_mattr(self, tokens: list[str], window_size: int = 50) -> float:
        """Calculate Moving Average Type-Token Ratio (MATTR).

        More robust vocabulary diversity measure than simple TTR.
        """
        if len(tokens) <= window_size:
            return len(set(tokens)) / len(tokens) if tokens else 0.0

        ttrs = []
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i : i + window_size]
            ttr = len(set(window)) / len(window)
            ttrs.append(ttr)

        return sum(ttrs) / len(ttrs) if ttrs else 0.0

    def _calculate_avg_tree_depth(self, doc: Doc) -> float:
        """Calculate average syntactic tree depth."""
        depths = []

        for sent in doc.sents:
            for token in sent:
                # Calculate depth to root
                depth = 0
                current = token
                while current.head != current:
                    depth += 1
                    current = current.head
                    if depth > 50:  # Safety limit
                        break
                depths.append(depth)

        return sum(depths) / len(depths) if depths else 0.0

    def _calculate_subordination_ratio(self, doc: Doc) -> float:
        """Calculate ratio of subordinate clauses.

        Subordinate clauses include:
        - advcl (adverbial clause modifier)
        - ccomp (clausal complement)
        - xcomp (open clausal complement)
        - acl (clausal modifier of noun)
        """
        subordinate_deps = {"advcl", "ccomp", "xcomp", "acl", "relcl"}

        subordinate_count = sum(
            1 for t in doc if t.dep_ in subordinate_deps
        )
        clause_count = sum(1 for t in doc if t.pos_ == "VERB")

        return subordinate_count / clause_count if clause_count > 0 else 0.0

    def _calculate_coordination_ratio(self, doc: Doc) -> float:
        """Calculate ratio of coordinate structures.

        Looks for conjunctions connecting similar elements.
        """
        coord_count = sum(1 for t in doc if t.dep_ == "conj")
        clause_count = sum(1 for t in doc if t.pos_ == "VERB")

        return coord_count / clause_count if clause_count > 0 else 0.0

    def _calculate_transition_variety(self, text_lower: str) -> float:
        """Calculate entropy of transition word types.

        Higher entropy = more varied use of transition types.
        """
        category_counts = {}

        for category, words in TRANSITION_WORDS.items():
            count = sum(1 for word in words if word in text_lower)
            if count > 0:
                category_counts[category] = count

        if not category_counts:
            return 0.0

        total = sum(category_counts.values())
        entropy = 0.0
        for count in category_counts.values():
            prob = count / total
            entropy -= prob * math.log2(prob)

        # Normalize by max possible entropy
        max_entropy = math.log2(len(TRANSITION_WORDS))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count for a word."""
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # Adjust for silent e
        if word.endswith("e") and count > 1:
            count -= 1

        return max(count, 1)

    def _flesch_reading_ease(self, doc: Doc, sentences: list) -> float:
        """Calculate Flesch Reading Ease score.

        Higher scores = easier to read.
        Range: 0-100 (can go negative for very complex text)
        """
        words = [t for t in doc if t.is_alpha]
        if not words or not sentences:
            return 50.0  # Default middle score

        total_words = len(words)
        total_sentences = len(sentences)
        total_syllables = sum(self._count_syllables(t.text) for t in words)

        # Flesch formula
        score = (
            206.835
            - 1.015 * (total_words / total_sentences)
            - 84.6 * (total_syllables / total_words)
        )

        return max(0.0, min(100.0, score))

    def _flesch_kincaid_grade(self, doc: Doc, sentences: list) -> float:
        """Calculate Flesch-Kincaid Grade Level.

        Returns approximate US grade level needed to understand the text.
        """
        words = [t for t in doc if t.is_alpha]
        if not words or not sentences:
            return 8.0  # Default grade level

        total_words = len(words)
        total_sentences = len(sentences)
        total_syllables = sum(self._count_syllables(t.text) for t in words)

        # Flesch-Kincaid formula
        grade = (
            0.39 * (total_words / total_sentences)
            + 11.8 * (total_syllables / total_words)
            - 15.59
        )

        return max(0.0, min(20.0, grade))

    def _default_features(self) -> LinguisticFeatures:
        """Return default features for edge cases."""
        return LinguisticFeatures(
            num_sentences=0,
            avg_sentence_length=0.0,
            sentence_length_variance=0.0,
            sentence_length_entropy=0.0,
            type_token_ratio=0.0,
            hapax_ratio=0.0,
            vocabulary_richness=0.0,
            avg_tree_depth=0.0,
            subordination_ratio=0.0,
            coordination_ratio=0.0,
            transition_words_ratio=0.0,
            transition_variety=0.0,
            noun_ratio=0.0,
            verb_ratio=0.0,
            adjective_ratio=0.0,
            adverb_ratio=0.0,
            flesch_reading_ease=50.0,
            flesch_kincaid_grade=8.0,
        )
