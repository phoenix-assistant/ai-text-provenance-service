# Development Guide

## Setup

### Prerequisites

- Python 3.10+
- pip or uv

### Installation

```bash
# Clone the repository
git clone https://github.com/phoenix-assistant/ai-text-provenance-service.git
cd ai-text-provenance-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai_text_provenance --cov-report=html

# Run specific test file
pytest tests/test_rst_parser.py -v
```

### Linting

```bash
# Check code style
ruff check src/

# Format code
ruff format src/

# Type checking
mypy src/
```

## Architecture

### Feature Extraction Pipeline

```
Text Input
    ↓
┌───────────────────────────────────────┐
│         Feature Extractor             │
├───────────────────────────────────────┤
│  RST Parser        → Discourse units  │
│                    → Tree structure   │
│                    → Relation types   │
│                    → Coherence        │
├───────────────────────────────────────┤
│  Linguistic Ext.   → Sentence stats   │
│                    → Vocabulary       │
│                    → Syntax           │
│                    → Transitions      │
├───────────────────────────────────────┤
│  Statistical Ext.  → Perplexity       │
│                    → Entropy          │
│                    → Repetition       │
│                    → Zipf/Heaps       │
└───────────────────────────────────────┘
    ↓
Feature Vector (47 dimensions)
```

### Model Architecture

```
┌─────────────────────────────────────────────────┐
│              Ensemble Classifier                │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────────┐    ┌─────────────────┐   │
│  │  Transformer    │    │   Feature MLP    │   │
│  │  Encoder        │    │                  │   │
│  │  (DistilBERT)   │    │  47 → 128 → 64  │   │
│  └────────┬────────┘    └────────┬────────┘   │
│           │                      │             │
│           └──────────┬───────────┘             │
│                      ↓                         │
│              ┌──────────────┐                  │
│              │   Fusion     │                  │
│              │   Layer      │                  │
│              └──────┬───────┘                  │
│                     ↓                          │
│              ┌──────────────┐                  │
│              │  Weighted    │                  │
│              │  Ensemble    │                  │
│              └──────┬───────┘                  │
│                     ↓                          │
│           4-class output logits               │
│                                                │
└─────────────────────────────────────────────────┘
```

## Training

### Create Sample Data

```bash
python scripts/train.py --create-sample-data --sample-size 500
```

This generates synthetic data for each class:
- `human`: Natural, varied writing with imperfections
- `ai`: Smooth, structured, comprehensive
- `polished_human`: Human draft refined with AI
- `humanized_ai`: AI text processed through humanizers

### Train Model

```bash
python scripts/train.py \
    --train data/train.jsonl \
    --val data/val.jsonl \
    --epochs 10 \
    --batch-size 16 \
    --lr 2e-5 \
    --output outputs/
```

With W&B logging:

```bash
python scripts/train.py \
    --train data/train.jsonl \
    --val data/val.jsonl \
    --wandb
```

### Export to ONNX

```bash
python scripts/export_onnx.py \
    --model outputs/best_model.pt \
    --output models/model.onnx
```

## Data Format

Training data should be JSONL format:

```json
{"text": "Sample text content...", "label": "human"}
{"text": "Another example...", "label": "ai"}
{"text": "Edited text...", "label": "polished_human"}
{"text": "Paraphrased AI...", "label": "humanized_ai"}
```

Labels:
- `human`: Original human-written text
- `ai`: Direct AI output (ChatGPT, Claude, etc.)
- `polished_human`: Human text edited/refined with AI assistance
- `humanized_ai`: AI text run through humanizer tools

## RST Features (Key Innovation)

Rhetorical Structure Theory (RST) analyzes how text segments relate:

### Why RST Works

1. **Discourse structure is hard to fake** - Humanizers change surface features but can't fundamentally restructure arguments

2. **AI has predictable patterns** - LLMs tend toward:
   - Simpler nucleus/satellite structures
   - Overuse of certain relations (elaboration, joint)
   - More uniform coherence patterns

3. **Human writing is messier** - Natural variation in:
   - Tree depth and balance
   - Relation type distribution
   - Local coherence fluctuations

### Key Features

| Feature | What It Measures | Why It Matters |
|---------|-----------------|----------------|
| `tree_depth` | Maximum RST tree depth | AI tends toward flatter structures |
| `tree_balance` | How balanced the tree is | Human writing is less balanced |
| `nucleus_ratio` | Nucleus vs satellite spans | AI overuses certain patterns |
| `coherence_breaks` | Sudden topic shifts | Humanizers create breaks |

## API Reference

### POST /classify

Classify a single text.

**Request:**
```json
{
  "text": "Your text here (50-100000 chars)...",
  "include_features": false
}
```

**Response:**
```json
{
  "prediction": "humanized_ai",
  "confidence": 0.87,
  "probabilities": {
    "human": 0.05,
    "ai": 0.03,
    "polished_human": 0.05,
    "humanized_ai": 0.87
  },
  "features": null
}
```

### POST /classify/batch

Classify multiple texts (max 100).

**Request:**
```json
{
  "texts": ["text1", "text2", "..."],
  "include_features": false
}
```

**Response:**
```json
{
  "results": [...],
  "total": 3,
  "processing_time_ms": 234.5
}
```

### GET /health

Health check endpoint.

## Deployment

### Docker

```bash
# Build
docker build -f docker/Dockerfile -t ai-text-provenance .

# Run
docker run -p 8000:8000 -v ./models:/models ai-text-provenance
```

### Docker Compose

```bash
# CPU version
docker-compose -f docker/docker-compose.yml up

# GPU version
docker-compose -f docker/docker-compose.yml --profile gpu up
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROVENANCE_MODEL_PATH` | None | Path to model weights |
| `PROVENANCE_USE_ONNX` | true | Use ONNX Runtime |
| `PROVENANCE_DEVICE` | auto | cpu, cuda, or mps |
| `PROVENANCE_MAX_BATCH_SIZE` | 32 | Maximum batch size |
| `PROVENANCE_CORS_ORIGINS` | ["*"] | CORS allowed origins |
| `PROVENANCE_LOG_LEVEL` | INFO | Logging level |

## CLI Usage

```bash
# Classify text
provenance classify "Your text here..."

# Classify from file
provenance classify -f document.txt

# With features
provenance classify -f document.txt --features --json

# Batch process
provenance batch input.jsonl output.jsonl

# Start server
provenance server --host 0.0.0.0 --port 8000

# Extract features only
provenance features "Your text here..."
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run linting: `ruff check src/ && ruff format --check src/`
5. Run tests: `pytest`
6. Commit: `git commit -m "Add my feature"`
7. Push and create a PR
