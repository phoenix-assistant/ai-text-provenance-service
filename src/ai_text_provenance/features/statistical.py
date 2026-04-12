"""Statistical and information-theoretic feature extraction.

Captures distributional patterns in text:
- Perplexity-based features (traditional AI detection)
- Entropy measures
- Zipf/Heaps law coefficients
- Repetition patterns
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import spacy
from spacy.tokens import Doc

from ai_text_provenance.models.schemas import StatisticalFeatures


class StatisticalExtractor:
    """Extract statistical and information-theoretic features."""

    def __init__(self, nlp: spacy.Language | None = None):
        """Initialize the statistical extractor.

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

        # GPT-2 model for perplexity (lazy loaded)
        self._perplexity_model = None
        self._perplexity_tokenizer = None

    def _load_perplexity_model(self) -> None:
        """Lazy load GPT-2 for perplexity calculation."""
        if self._perplexity_model is not None:
            return

        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2Tokenizer

            self._perplexity_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self._perplexity_model = GPT2LMHeadModel.from_pretrained("gpt2")
            self._perplexity_model.eval()

            # Use GPU if available
            if torch.cuda.is_available():
                self._perplexity_model = self._perplexity_model.cuda()
        except ImportError:
            # Transformers not installed, perplexity features will use fallback
            pass

    def extract_features(self, text: str) -> StatisticalFeatures:
        """Extract statistical features from text.

        Args:
            text: Input text to analyze.

        Returns:
            StatisticalFeatures with all metrics.
        """
        doc = self.nlp(text)
        tokens = [t.text.lower() for t in doc if t.is_alpha]

        if len(tokens) < 10:
            return self._default_features()

        # Perplexity features
        perplexity, perplexity_variance, perplexity_burstiness = (
            self._calculate_perplexity_features(text, doc)
        )

        # Entropy measures
        word_entropy = self._calculate_word_entropy(tokens)
        bigram_entropy = self._calculate_ngram_entropy(tokens, 2)
        trigram_entropy = self._calculate_ngram_entropy(tokens, 3)

        # Distribution coefficients
        zipf_coefficient = self._calculate_zipf_coefficient(tokens)
        heaps_coefficient = self._calculate_heaps_coefficient(tokens)

        # Repetition patterns
        exact_repetition_score = self._calculate_exact_repetition(tokens)
        semantic_repetition_score = self._calculate_semantic_repetition(doc)

        # Predictability
        predictability_score, surprise_score = self._calculate_predictability(tokens)

        return StatisticalFeatures(
            perplexity=perplexity,
            perplexity_variance=perplexity_variance,
            perplexity_burstiness=perplexity_burstiness,
            word_entropy=word_entropy,
            bigram_entropy=bigram_entropy,
            trigram_entropy=trigram_entropy,
            zipf_coefficient=zipf_coefficient,
            heaps_coefficient=heaps_coefficient,
            exact_repetition_score=exact_repetition_score,
            semantic_repetition_score=semantic_repetition_score,
            predictability_score=predictability_score,
            surprise_score=surprise_score,
        )

    def _calculate_perplexity_features(self, _text: str, doc: Doc) -> tuple[float, float, float]:
        """Calculate perplexity-based features.

        Returns:
            Tuple of (perplexity, variance, burstiness)
        """
        self._load_perplexity_model()

        if self._perplexity_model is None:
            # Fallback: use statistical approximation
            return self._approximate_perplexity(doc)

        try:
            import torch

            sentences = list(doc.sents)
            sentence_perplexities = []

            for sent in sentences:
                sent_text = sent.text.strip()
                if len(sent_text) < 5:
                    continue

                inputs = self._perplexity_tokenizer(
                    sent_text, return_tensors="pt", truncation=True, max_length=512
                )

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._perplexity_model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                    ppl = math.exp(loss)
                    sentence_perplexities.append(ppl)

            if not sentence_perplexities:
                return 50.0, 0.0, 0.0

            avg_ppl = sum(sentence_perplexities) / len(sentence_perplexities)
            variance = np.var(sentence_perplexities) if len(sentence_perplexities) > 1 else 0.0

            # Burstiness: how much perplexity varies (normalized)
            if variance > 0:
                burstiness = np.std(sentence_perplexities) / avg_ppl if avg_ppl > 0 else 0.0
            else:
                burstiness = 0.0

            return avg_ppl, float(variance), float(burstiness)

        except Exception:
            return self._approximate_perplexity(doc)

    def _approximate_perplexity(self, doc: Doc) -> tuple[float, float, float]:
        """Statistical approximation of perplexity without GPT-2.

        Uses character-level n-gram entropy as proxy.
        """
        text = doc.text.lower()

        # Character trigram entropy
        trigrams = [text[i : i + 3] for i in range(len(text) - 2)]
        if not trigrams:
            return 50.0, 0.0, 0.0

        counter = Counter(trigrams)
        total = len(trigrams)

        entropy = 0.0
        for count in counter.values():
            prob = count / total
            entropy -= prob * math.log2(prob)

        # Convert to approximate perplexity
        ppl = 2**entropy

        # Sentence-level variance approximation
        sentences = list(doc.sents)
        sent_entropies = []
        for sent in sentences:
            sent_text = sent.text.lower()
            sent_trigrams = [sent_text[i : i + 3] for i in range(len(sent_text) - 2)]
            if sent_trigrams:
                sent_counter = Counter(sent_trigrams)
                sent_total = len(sent_trigrams)
                sent_entropy = 0.0
                for count in sent_counter.values():
                    prob = count / sent_total
                    sent_entropy -= prob * math.log2(prob)
                sent_entropies.append(sent_entropy)

        variance = np.var(sent_entropies) if len(sent_entropies) > 1 else 0.0
        burstiness = (
            np.std(sent_entropies) / np.mean(sent_entropies)
            if sent_entropies and np.mean(sent_entropies) > 0
            else 0.0
        )

        return ppl, float(variance), float(burstiness)

    def _calculate_word_entropy(self, tokens: list[str]) -> float:
        """Calculate word-level entropy."""
        if not tokens:
            return 0.0

        counter = Counter(tokens)
        total = len(tokens)

        entropy = 0.0
        for count in counter.values():
            prob = count / total
            entropy -= prob * math.log2(prob)

        return entropy

    def _calculate_ngram_entropy(self, tokens: list[str], n: int) -> float:
        """Calculate n-gram entropy."""
        if len(tokens) < n:
            return 0.0

        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        counter = Counter(ngrams)
        total = len(ngrams)

        entropy = 0.0
        for count in counter.values():
            prob = count / total
            entropy -= prob * math.log2(prob)

        return entropy

    def _calculate_zipf_coefficient(self, tokens: list[str]) -> float:
        """Calculate Zipf's law coefficient.

        Zipf's law: frequency ∝ 1/rank^α
        Returns α coefficient.
        """
        if len(tokens) < 10:
            return 1.0

        counter = Counter(tokens)
        frequencies = sorted(counter.values(), reverse=True)

        if len(frequencies) < 2:
            return 1.0

        # Linear regression in log-log space
        ranks = np.arange(1, len(frequencies) + 1)
        log_ranks = np.log(ranks)
        log_freqs = np.log(frequencies)

        # Simple least squares: α = -slope
        n = len(log_ranks)
        sum_x = np.sum(log_ranks)
        sum_y = np.sum(log_freqs)
        sum_xy = np.sum(log_ranks * log_freqs)
        sum_x2 = np.sum(log_ranks**2)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        alpha = -slope

        return float(alpha)

    def _calculate_heaps_coefficient(self, tokens: list[str]) -> float:
        """Calculate Heaps' law coefficient.

        Heaps' law: vocabulary size V ∝ N^β
        Returns β coefficient.
        """
        if len(tokens) < 20:
            return 0.5

        # Calculate vocabulary growth
        vocab_sizes = []
        seen = set()

        checkpoints = [
            int(len(tokens) * p) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ]
        checkpoints = [c for c in checkpoints if c > 0]

        for i, token in enumerate(tokens):
            seen.add(token)
            if i + 1 in checkpoints:
                vocab_sizes.append((i + 1, len(seen)))

        if len(vocab_sizes) < 2:
            return 0.5

        # Linear regression in log-log space
        ns = np.array([v[0] for v in vocab_sizes])
        vs = np.array([v[1] for v in vocab_sizes])

        log_ns = np.log(ns)
        log_vs = np.log(vs)

        n = len(log_ns)
        sum_x = np.sum(log_ns)
        sum_y = np.sum(log_vs)
        sum_xy = np.sum(log_ns * log_vs)
        sum_x2 = np.sum(log_ns**2)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

        return float(slope)

    def _calculate_exact_repetition(self, tokens: list[str]) -> float:
        """Calculate exact phrase repetition score.

        Measures how much of the text is repeated phrases.
        """
        if len(tokens) < 10:
            return 0.0

        # Check for repeated n-grams (3-5 words)
        repeated_tokens = 0

        for n in [3, 4, 5]:
            if len(tokens) < n * 2:
                continue

            ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            counter = Counter(ngrams)

            for _ngram, count in counter.items():
                if count > 1:
                    repeated_tokens += n * (count - 1)

        # Normalize by total tokens
        return min(1.0, repeated_tokens / len(tokens))

    def _calculate_semantic_repetition(self, doc: Doc) -> float:
        """Calculate semantic repetition using word vectors.

        Measures how semantically similar sentences are to each other.
        """
        sentences = list(doc.sents)

        if len(sentences) < 2:
            return 0.0

        # Get sentence vectors
        sent_vectors = []
        for sent in sentences:
            # Average of word vectors
            word_vectors = [
                token.vector for token in sent if token.has_vector and not token.is_stop
            ]
            if word_vectors:
                sent_vec = np.mean(word_vectors, axis=0)
                if not np.all(sent_vec == 0):
                    sent_vectors.append(sent_vec)

        if len(sent_vectors) < 2:
            return 0.0

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(sent_vectors)):
            for j in range(i + 1, len(sent_vectors)):
                sim = self._cosine_similarity(sent_vectors[i], sent_vectors[j])
                similarities.append(sim)

        # High similarity = more repetition
        avg_similarity = np.mean(similarities)

        # Threshold: similarities above 0.8 are "repetitive"
        return max(0.0, (avg_similarity - 0.5) * 2) if avg_similarity > 0.5 else 0.0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _calculate_predictability(self, tokens: list[str]) -> tuple[float, float]:
        """Calculate text predictability metrics.

        Uses n-gram statistics to estimate how predictable the text is.

        Returns:
            Tuple of (predictability_score, surprise_score)
        """
        if len(tokens) < 10:
            return 0.5, 0.5

        # Build n-gram model from the text itself
        # (In production, would use external corpus statistics)
        bigram_probs: dict[str, dict[str, float]] = {}

        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if w1 not in bigram_probs:
                bigram_probs[w1] = {}
            bigram_probs[w1][w2] = bigram_probs[w1].get(w2, 0) + 1

        # Normalize
        for w1 in bigram_probs:
            total = sum(bigram_probs[w1].values())
            for w2 in bigram_probs[w1]:
                bigram_probs[w1][w2] /= total

        # Calculate surprisal for each word
        surprisals = []
        for i in range(1, len(tokens)):
            w1, w2 = tokens[i - 1], tokens[i]
            if w1 in bigram_probs and w2 in bigram_probs[w1]:
                prob = bigram_probs[w1][w2]
                surprisal = -math.log2(prob)
            else:
                surprisal = 10.0  # High surprisal for unseen bigrams

            surprisals.append(surprisal)

        avg_surprisal = np.mean(surprisals)
        # Normalize surprisal to 0-1 range
        surprise_score = min(1.0, avg_surprisal / 10.0)

        # Predictability is inverse of surprise
        predictability = 1.0 - surprise_score

        return float(predictability), float(surprise_score)

    def _default_features(self) -> StatisticalFeatures:
        """Return default features for edge cases."""
        return StatisticalFeatures(
            perplexity=50.0,
            perplexity_variance=0.0,
            perplexity_burstiness=0.0,
            word_entropy=0.0,
            bigram_entropy=0.0,
            trigram_entropy=0.0,
            zipf_coefficient=1.0,
            heaps_coefficient=0.5,
            exact_repetition_score=0.0,
            semantic_repetition_score=0.0,
            predictability_score=0.5,
            surprise_score=0.5,
        )
