"""AI Text Provenance Service — 4-way text classification."""

__version__ = "0.1.0"

from ai_text_provenance.models.classifier import ProvenanceClassifier
from ai_text_provenance.models.schemas import (
    ClassificationResult,
    ProvenanceClass,
)

__all__ = [
    "ProvenanceClassifier",
    "ClassificationResult",
    "ProvenanceClass",
]
