# AI Text Provenance Service

[![CI](https://github.com/phoenix-assistant/ai-text-provenance-service/actions/workflows/ci.yml/badge.svg)](https://github.com/phoenix-assistant/ai-text-provenance-service/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Know if text is human, AI, polished, or humanized—with receipts.**

4-way text classification API that goes beyond simple AI detection. Using Rhetorical Structure Theory (RST) analysis, we detect text provenance that humanizer tools can't easily defeat.

## Quick Start

```bash
pip install ai-text-provenance

# Classify text
provenance classify "Your text here..."

# Or start the API server
provenance server
```

```python
from ai_text_provenance import ProvenanceClassifier

classifier = ProvenanceClassifier()
result = classifier.classify("Your text here...")

print(result.prediction)     # "human", "ai", "polished_human", or "humanized_ai"
print(result.confidence)     # 0.87
print(result.probabilities)  # {"human": 0.05, "ai": 0.03, ...}
```

## Why This Matters

**Existing detectors fail** because they rely on surface features (perplexity, word frequency) that humanizer tools easily defeat.

**We analyze structure** using Rhetorical Structure Theory—how ideas connect, not how sentences look. This is fundamentally harder to fake.

| Category | Description | Example |
|----------|-------------|---------|
| **Human** | Original human writing | Blog post, email |
| **AI** | Direct LLM output | ChatGPT response |
| **Polished Human** | Human draft, AI-refined | Professional editing |
| **Humanized AI** | AI text, paraphrased to evade | Undetectable.ai output |

## Installation

```bash
# Basic installation
pip install ai-text-provenance

# With GPU support
pip install ai-text-provenance[gpu]

# For training your own models
pip install ai-text-provenance[training]
```

## API Usage

### REST API

```bash
# Start server
provenance server --port 8000

# Or with Docker
docker run -p 8000:8000 ghcr.io/phoenix-assistant/ai-text-provenance-service
```

```bash
# Classify text
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here...", "include_features": false}'
```

### Python SDK

```python
from ai_text_provenance import ProvenanceClassifier

# Initialize (downloads model on first run)
classifier = ProvenanceClassifier()

# Single text
result = classifier.classify(text, include_features=True)

# Batch processing
results = classifier.classify_batch(texts)
```

### CLI

```bash
# Classify text
provenance classify "Your text here..."
provenance classify -f document.txt --json

# Batch process
provenance batch input.jsonl output.jsonl

# Extract features only
provenance features "Your text here..."
```

## The RST Advantage

Rhetorical Structure Theory analyzes discourse—how ideas relate:

- **Nucleus/Satellite relations** — AI tends toward simpler structures
- **Discourse depth** — Humans have more varied nesting
- **Relation types** — AI overuses certain transitions
- **Coherence patterns** — Humanizers break these trying to vary text

These structural features survive paraphrasing because fundamentally changing them means rewriting the argument.

## Background

### The Problem

**Existing detectors don't work:**
- GPTZero, Turnitin AI: 15-30% false positive rates
- Humanizer tools ($500M+ market) defeat them easily
- No nuance: "AI or human" misses the reality of AI-assisted writing

**Real consequences:**
- False accusations destroy academic trust
- HR can't filter AI-spray job applicants
- Content authenticity is critical for trust

### Our Solution

Based on the RACE paper's techniques, we:
1. Parse text into Elementary Discourse Units (EDUs)
2. Build RST trees capturing argument structure
3. Extract features humanizers can't easily modify
4. Combine with transformer embeddings for robust classification

## Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   API Gateway                            │
│  POST /classify { text: string }                        │
│  Response: { class: "human"|"ai"|"polished"|"humanized",│
│              confidence: 0.87, explanation: {...} }      │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   Classification Pipeline                │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  1. RST Parser                                   │   │
│  │  • Segment text into Elementary Discourse Units  │   │
│  │  • Build RST tree (nucleus-satellite relations)  │   │
│  │  • Extract structural features                   │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────▼──────────────────────────┐   │
│  │  2. Feature Extraction                           │   │
│  │  • Discourse relation distribution               │   │
│  │  • Tree depth and balance metrics               │   │
│  │  • Nuclearity patterns                          │   │
│  │  • Cross-sentence coherence                     │   │
│  │  • Argumentation flow signatures                │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────▼──────────────────────────┐   │
│  │  3. Classification Model                         │   │
│  │  • Ensemble: RST features + fine-tuned BERT     │   │
│  │  • 4-way softmax output                         │   │
│  │  • Calibrated confidence scores                 │   │
│  │  • Explainability layer (which features fired)  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   Supporting Services                    │
│  • Model versioning (MLflow)                            │
│  • A/B testing infrastructure                           │
│  • Feedback collection (for model improvement)          │
│  • Rate limiting and authentication                     │
└─────────────────────────────────────────────────────────┘
```

**Stack:**
- RST Parser: SpaCy + custom discourse parser (or adapt existing: feng-hirst, DPLP)
- Classification: PyTorch, fine-tuned BERT + RST feature ensemble
- API: FastAPI (Python, same as ML stack)
- Serving: NVIDIA Triton or ONNX Runtime for inference
- Storage: PostgreSQL for logs, S3 for model artifacts
- MLOps: MLflow for versioning, Weights & Biases for experiments

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for:
- Local development setup
- Training custom models
- API reference
- Deployment guide

## Docker

```bash
# CPU version
docker-compose -f docker/docker-compose.yml up

# GPU version
docker-compose -f docker/docker-compose.yml --profile gpu up
```

## License

MIT License - see [LICENSE](LICENSE) for details.
