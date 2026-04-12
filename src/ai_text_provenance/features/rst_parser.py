"""RST (Rhetorical Structure Theory) Parser.

This module implements discourse parsing to extract RST features.
RST analysis is the key innovation — humanizer tools attack surface features
but can't easily modify discourse structure without rewriting the text.

Based on techniques from the RACE paper for AI text detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import spacy
from spacy.tokens import Doc, Span

from ai_text_provenance.models.schemas import RSTFeatures


@dataclass
class DiscourseUnit:
    """Elementary Discourse Unit (EDU).

    The basic building block of RST analysis. Typically a clause.
    """

    text: str
    tokens: list[str]
    start: int
    end: int
    sentence_idx: int


@dataclass
class RSTNode:
    """Node in an RST tree.

    Represents either a leaf (EDU) or an internal node (relation).
    """

    # Nuclearity: 'nucleus', 'satellite', or 'multinuclear'
    nuclearity: str
    # Relation type if internal node
    relation: str | None = None
    # EDU if leaf node
    edu: DiscourseUnit | None = None
    # Children if internal node
    children: list[RSTNode] = field(default_factory=list)
    # Depth in tree
    depth: int = 0


# RST relation types commonly used in discourse analysis
RST_RELATIONS = {
    # Presentational relations
    "attribution": ["said", "according", "stated", "claimed", "reported"],
    "antithesis": ["but", "however", "although", "despite", "nevertheless"],
    "background": ["because", "since", "as", "given"],
    "concession": ["although", "even though", "despite", "while"],
    "enablement": ["by", "through", "in order to"],
    "motivation": ["so that", "in order", "to"],
    "evidence": ["for example", "for instance", "such as"],
    "justify": ["because", "since", "therefore"],
    "restatement": ["in other words", "that is", "namely"],
    # Subject matter relations
    "elaboration": ["specifically", "in particular", "namely", "for example"],
    "circumstance": ["when", "while", "as", "during"],
    "cause": ["because", "since", "as", "therefore", "thus", "so"],
    "result": ["therefore", "thus", "consequently", "hence", "so"],
    "condition": ["if", "unless", "provided", "when"],
    "otherwise": ["otherwise", "else", "or else"],
    "interpretation": ["this means", "this suggests", "indicates"],
    "evaluation": ["importantly", "significantly", "notably"],
    "solutionhood": ["to solve", "to address", "the solution"],
    "purpose": ["to", "in order to", "so that", "for"],
    # Multinuclear relations
    "contrast": ["but", "however", "while", "whereas", "on the other hand"],
    "comparison": ["similarly", "likewise", "compared to", "in contrast"],
    "joint": ["and", "also", "moreover", "furthermore", "additionally"],
    "sequence": ["first", "then", "next", "finally", "subsequently"],
    "disjunction": ["or", "either", "alternatively"],
}


class RSTParser:
    """Rhetorical Structure Theory parser.

    Extracts discourse units and builds RST trees to analyze
    how ideas connect in text. This is the key differentiator
    for detecting humanized AI — structure is harder to fake.
    """

    def __init__(self, nlp: spacy.Language | None = None):
        """Initialize the RST parser.

        Args:
            nlp: SpaCy language model. If None, loads en_core_web_sm.
        """
        if nlp is None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Model not installed, download it
                import subprocess

                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp

    def segment_into_edus(self, doc: Doc) -> list[DiscourseUnit]:
        """Segment text into Elementary Discourse Units.

        EDUs are typically clauses. We use syntactic cues to identify
        clause boundaries.

        Args:
            doc: SpaCy processed document.

        Returns:
            List of discourse units.
        """
        edus: list[DiscourseUnit] = []

        for sent_idx, sent in enumerate(doc.sents):
            # Find clause boundaries within sentence
            clause_boundaries = self._find_clause_boundaries(sent)

            if not clause_boundaries:
                # Whole sentence is one EDU
                edus.append(
                    DiscourseUnit(
                        text=sent.text.strip(),
                        tokens=[t.text for t in sent],
                        start=sent.start,
                        end=sent.end,
                        sentence_idx=sent_idx,
                    )
                )
            else:
                # Split into multiple EDUs
                prev_boundary = sent.start
                for boundary in clause_boundaries:
                    if boundary > prev_boundary:
                        span = doc[prev_boundary:boundary]
                        if span.text.strip():
                            edus.append(
                                DiscourseUnit(
                                    text=span.text.strip(),
                                    tokens=[t.text for t in span],
                                    start=prev_boundary,
                                    end=boundary,
                                    sentence_idx=sent_idx,
                                )
                            )
                    prev_boundary = boundary

                # Don't forget the last segment
                if prev_boundary < sent.end:
                    span = doc[prev_boundary : sent.end]
                    if span.text.strip():
                        edus.append(
                            DiscourseUnit(
                                text=span.text.strip(),
                                tokens=[t.text for t in span],
                                start=prev_boundary,
                                end=sent.end,
                                sentence_idx=sent_idx,
                            )
                        )

        return edus

    def _find_clause_boundaries(self, sent: Span) -> list[int]:
        """Find clause boundaries within a sentence.

        Uses syntactic patterns to identify where clauses break:
        - Subordinating conjunctions (because, although, if, when, while)
        - Relative pronouns (who, which, that)
        - Coordinating conjunctions with clauses (and, but, or)
        - Punctuation-based splits (semicolons, colons)
        """
        boundaries = []

        for token in sent:
            # Subordinating conjunctions start new clauses
            if token.dep_ == "mark" and token.head.pos_ == "VERB":
                boundaries.append(token.i)

            # Relative clauses
            if token.dep_ == "relcl":
                # Find the relative pronoun
                for child in token.children:
                    if child.dep_ in ("nsubj", "dobj") and child.pos_ == "PRON":
                        boundaries.append(child.i)
                        break
                else:
                    boundaries.append(token.i)

            # Coordinating conjunctions between clauses
            if token.dep_ == "cc" and token.head.pos_ == "VERB":
                # Check if there's a verb after this conjunction
                for right in token.rights:
                    if right.dep_ == "conj" and right.pos_ == "VERB":
                        boundaries.append(token.i)
                        break

            # Punctuation boundaries
            if token.text in (";", ":") and token.i < sent.end - 1:
                boundaries.append(token.i + 1)

        return sorted(set(boundaries))

    def identify_relation(self, edu1: DiscourseUnit, edu2: DiscourseUnit) -> str:
        """Identify the RST relation between two adjacent EDUs.

        Uses lexical cues and syntactic patterns to determine
        the discourse relation.
        """
        text1_lower = edu1.text.lower()
        text2_lower = edu2.text.lower()
        combined = f"{text1_lower} {text2_lower}"

        # Check for explicit discourse markers
        for relation, markers in RST_RELATIONS.items():
            for marker in markers:
                # Check at start of second EDU (most common)
                if text2_lower.startswith(marker) or f" {marker} " in text2_lower[:50]:
                    return relation
                # Check for markers that span EDUs
                if marker in combined:
                    return relation

        # Default: assume elaboration (most common relation)
        return "elaboration"

    def build_rst_tree(self, edus: list[DiscourseUnit]) -> RSTNode:
        """Build an RST tree from discourse units.

        Uses a simplified bottom-up approach:
        1. Adjacent EDUs within sentences form local trees
        2. Cross-sentence relations form higher-level structure
        """
        if not edus:
            return RSTNode(nuclearity="none")

        if len(edus) == 1:
            return RSTNode(nuclearity="nucleus", edu=edus[0], depth=0)

        # Group EDUs by sentence
        sentences: dict[int, list[DiscourseUnit]] = {}
        for edu in edus:
            if edu.sentence_idx not in sentences:
                sentences[edu.sentence_idx] = []
            sentences[edu.sentence_idx].append(edu)

        # Build subtrees for each sentence
        sentence_trees: list[RSTNode] = []
        for sent_idx in sorted(sentences.keys()):
            sent_edus = sentences[sent_idx]
            if len(sent_edus) == 1:
                sentence_trees.append(RSTNode(nuclearity="nucleus", edu=sent_edus[0], depth=0))
            else:
                # Build intra-sentence tree
                tree = self._build_sentence_tree(sent_edus)
                sentence_trees.append(tree)

        # Combine sentence trees
        if len(sentence_trees) == 1:
            return sentence_trees[0]

        return self._combine_sentence_trees(sentence_trees)

    def _build_sentence_tree(self, edus: list[DiscourseUnit]) -> RSTNode:
        """Build RST subtree for EDUs within a single sentence."""
        if len(edus) == 1:
            return RSTNode(nuclearity="nucleus", edu=edus[0], depth=0)

        # Identify relations between adjacent EDUs
        children = []
        for i, edu in enumerate(edus):
            nuclearity = "nucleus" if i == 0 else "satellite"
            relation = None

            if i > 0:
                relation = self.identify_relation(edus[i - 1], edu)
                # First EDU with elaboration usually means it's the nucleus
                if relation in ("elaboration", "evidence", "background"):
                    nuclearity = "satellite"
                elif relation in ("contrast", "joint", "sequence"):
                    nuclearity = "multinuclear"

            children.append(
                RSTNode(
                    nuclearity=nuclearity,
                    relation=relation,
                    edu=edu,
                    depth=1,
                )
            )

        return RSTNode(
            nuclearity="span",
            relation="joint" if any(c.nuclearity == "multinuclear" for c in children) else None,
            children=children,
            depth=0,
        )

    def _combine_sentence_trees(self, trees: list[RSTNode]) -> RSTNode:
        """Combine sentence-level trees into document tree."""
        # Simple approach: adjacent sentences form elaboration/joint relations
        root_children = []

        for i, tree in enumerate(trees):
            tree.depth = 1
            if i == 0:
                tree.nuclearity = "nucleus"
            else:
                # Determine relation based on first EDU of each tree
                prev_edu = self._get_first_edu(trees[i - 1])
                curr_edu = self._get_first_edu(tree)

                if prev_edu and curr_edu:
                    relation = self.identify_relation(prev_edu, curr_edu)
                    tree.relation = relation
                    if relation in ("joint", "sequence", "contrast"):
                        tree.nuclearity = "multinuclear"
                    else:
                        tree.nuclearity = "satellite"
                else:
                    tree.nuclearity = "satellite"

            root_children.append(tree)

        return RSTNode(
            nuclearity="span",
            children=root_children,
            depth=0,
        )

    def _get_first_edu(self, node: RSTNode) -> DiscourseUnit | None:
        """Get the first EDU from a tree node."""
        if node.edu:
            return node.edu
        for child in node.children:
            edu = self._get_first_edu(child)
            if edu:
                return edu
        return None

    def _calculate_tree_depth(self, node: RSTNode, current_depth: int = 0) -> int:
        """Calculate maximum depth of RST tree."""
        if not node.children:
            return current_depth

        return max(self._calculate_tree_depth(child, current_depth + 1) for child in node.children)

    def _calculate_tree_balance(self, node: RSTNode) -> float:
        """Calculate how balanced the RST tree is.

        Returns value between 0 (completely unbalanced) and 1 (perfectly balanced).
        """
        if not node.children:
            return 1.0

        if len(node.children) < 2:
            return 1.0

        depths = [self._calculate_tree_depth(child) for child in node.children]
        max_depth = max(depths)
        min_depth = min(depths)

        if max_depth == 0:
            return 1.0

        return 1.0 - (max_depth - min_depth) / (max_depth + 1)

    def _count_relations(self, node: RSTNode, counts: dict[str, int]) -> None:
        """Count occurrences of each relation type in tree."""
        if node.relation:
            counts[node.relation] = counts.get(node.relation, 0) + 1
        for child in node.children:
            self._count_relations(child, counts)

    def _count_nuclearity(self, node: RSTNode, counts: dict[str, int]) -> None:
        """Count nuclearity types in tree."""
        counts[node.nuclearity] = counts.get(node.nuclearity, 0) + 1
        for child in node.children:
            self._count_nuclearity(child, counts)

    def extract_features(self, text: str) -> RSTFeatures:
        """Extract RST features from text.

        This is the main entry point for RST analysis.

        Args:
            text: Input text to analyze.

        Returns:
            RSTFeatures with all discourse metrics.
        """
        doc = self.nlp(text)

        # Segment into EDUs
        edus = self.segment_into_edus(doc)

        if not edus:
            # Return default features for very short text
            return self._default_features()

        # Build RST tree
        tree = self.build_rst_tree(edus)

        # Calculate EDU statistics
        edu_lengths = [len(edu.tokens) for edu in edus]
        avg_edu_length = sum(edu_lengths) / len(edu_lengths)
        edu_length_variance = (
            sum((el - avg_edu_length) ** 2 for el in edu_lengths) / len(edu_lengths)
            if len(edu_lengths) > 1
            else 0.0
        )

        # Tree metrics
        tree_depth = self._calculate_tree_depth(tree)
        tree_balance = self._calculate_tree_balance(tree)

        # Calculate average depth
        depths = []
        self._collect_leaf_depths(tree, 0, depths)
        tree_depth_avg = sum(depths) / len(depths) if depths else 0.0

        # Nuclearity counts
        nuclearity_counts: dict[str, int] = {}
        self._count_nuclearity(tree, nuclearity_counts)
        total_nodes = sum(nuclearity_counts.values())

        nucleus_count = nuclearity_counts.get("nucleus", 0)
        satellite_count = nuclearity_counts.get("satellite", 0)
        multinuclear_count = nuclearity_counts.get("multinuclear", 0)

        nucleus_ratio = (
            nucleus_count / (nucleus_count + satellite_count)
            if (nucleus_count + satellite_count) > 0
            else 0.5
        )
        multinuclear_ratio = multinuclear_count / total_nodes if total_nodes > 0 else 0.0

        # Relation counts
        relation_counts: dict[str, int] = {}
        self._count_relations(tree, relation_counts)
        total_relations = sum(relation_counts.values()) or 1

        # Calculate coherence scores
        local_coherence = self._calculate_local_coherence(doc)
        global_coherence = self._calculate_global_coherence(doc)
        coherence_breaks = self._count_coherence_breaks(edus)

        return RSTFeatures(
            num_edus=len(edus),
            avg_edu_length=avg_edu_length,
            edu_length_variance=edu_length_variance,
            tree_depth=tree_depth,
            tree_depth_avg=tree_depth_avg,
            tree_balance=tree_balance,
            nucleus_ratio=nucleus_ratio,
            multinuclear_ratio=multinuclear_ratio,
            elaboration_ratio=relation_counts.get("elaboration", 0) / total_relations,
            contrast_ratio=relation_counts.get("contrast", 0) / total_relations,
            cause_ratio=(relation_counts.get("cause", 0) + relation_counts.get("result", 0))
            / total_relations,
            temporal_ratio=(
                relation_counts.get("sequence", 0) + relation_counts.get("circumstance", 0)
            )
            / total_relations,
            attribution_ratio=relation_counts.get("attribution", 0) / total_relations,
            condition_ratio=relation_counts.get("condition", 0) / total_relations,
            local_coherence=local_coherence,
            global_coherence=global_coherence,
            coherence_breaks=coherence_breaks,
        )

    def _collect_leaf_depths(self, node: RSTNode, current_depth: int, depths: list[int]) -> None:
        """Collect depths of all leaf nodes."""
        if not node.children:
            depths.append(current_depth)
        else:
            for child in node.children:
                self._collect_leaf_depths(child, current_depth + 1, depths)

    def _calculate_local_coherence(self, doc: Doc) -> float:
        """Calculate coherence between adjacent sentences.

        Uses entity-based coherence (centering theory).
        """
        sentences = list(doc.sents)
        if len(sentences) < 2:
            return 1.0

        coherence_scores = []
        for i in range(len(sentences) - 1):
            sent1_entities = {ent.text.lower() for ent in sentences[i].ents}
            sent1_nouns = {t.lemma_.lower() for t in sentences[i] if t.pos_ in ("NOUN", "PROPN")}
            sent1_refs = sent1_entities | sent1_nouns

            sent2_entities = {ent.text.lower() for ent in sentences[i + 1].ents}
            sent2_nouns = {
                t.lemma_.lower() for t in sentences[i + 1] if t.pos_ in ("NOUN", "PROPN")
            }
            sent2_refs = sent2_entities | sent2_nouns

            # Coherence = overlap in referenced entities
            if sent1_refs and sent2_refs:
                overlap = len(sent1_refs & sent2_refs)
                max_possible = min(len(sent1_refs), len(sent2_refs))
                coherence_scores.append(overlap / max_possible if max_possible > 0 else 0.0)
            else:
                coherence_scores.append(0.0)

        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1.0

    def _calculate_global_coherence(self, doc: Doc) -> float:
        """Calculate document-level coherence.

        Measures how well topics are maintained throughout the document.
        """
        sentences = list(doc.sents)
        if len(sentences) < 3:
            return 1.0

        # Extract main topics from first paragraph
        first_third = sentences[: len(sentences) // 3]
        main_topics = set()
        for sent in first_third:
            for token in sent:
                if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                    main_topics.add(token.lemma_.lower())

        if not main_topics:
            return 0.5

        # Check topic presence throughout document
        topic_presence = []
        for sent in sentences:
            sent_topics = {t.lemma_.lower() for t in sent if t.pos_ in ("NOUN", "PROPN")}
            overlap = len(sent_topics & main_topics)
            topic_presence.append(1.0 if overlap > 0 else 0.0)

        return sum(topic_presence) / len(topic_presence)

    def _count_coherence_breaks(self, edus: list[DiscourseUnit]) -> int:
        """Count sudden topic shifts between EDUs.

        Humanized AI often has more breaks because paraphrasing
        disrupts natural topic flow.
        """
        if len(edus) < 2:
            return 0

        breaks = 0
        for i in range(len(edus) - 1):
            tokens1 = {t.lower() for t in edus[i].tokens if len(t) > 3}
            tokens2 = {t.lower() for t in edus[i + 1].tokens if len(t) > 3}

            # Check for content word overlap
            overlap = len(tokens1 & tokens2)
            if overlap == 0 and len(tokens1) > 2 and len(tokens2) > 2:
                breaks += 1

        return breaks

    def _default_features(self) -> RSTFeatures:
        """Return default features for edge cases."""
        return RSTFeatures(
            num_edus=0,
            avg_edu_length=0.0,
            edu_length_variance=0.0,
            tree_depth=0,
            tree_depth_avg=0.0,
            tree_balance=1.0,
            nucleus_ratio=0.5,
            multinuclear_ratio=0.0,
            elaboration_ratio=0.0,
            contrast_ratio=0.0,
            cause_ratio=0.0,
            temporal_ratio=0.0,
            attribution_ratio=0.0,
            condition_ratio=0.0,
            local_coherence=0.0,
            global_coherence=0.0,
            coherence_breaks=0,
        )
