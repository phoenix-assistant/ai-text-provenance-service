# AI Text Provenance Service

> **One-liner:** Know if text is human, AI, polished, or humanized—with receipts.

## Problem

**Persona:** Professor Chen, department head at a research university

**Pain:**
- 40% of student submissions now show AI patterns, but she can't prove it
- Existing detectors (GPTZero, Turnitin AI) have 15-30% false positive rates
- "Humanizer" tools (Undetectable.ai, QuillBot) defeat current detectors
- False accusations destroy trust; missed AI submissions undermine integrity
- No reliable way to distinguish "AI-assisted editing" from "AI-generated"

**Quantified:**
- 65% of faculty report increased AI use suspicion (no way to verify)
- 23% false positive rate on GPTZero causes wrongful accusations
- Humanizer tools market is $500M+ (existence proves detector gap)
- Academic integrity cases up 300% since ChatGPT launch

**Secondary Persona:** HR Director at Fortune 500

- Receives 10,000 resumes/month, 60%+ now AI-generated
- Can't distinguish thoughtful candidates from AI-spray applicants
- AI cover letters all sound identical, impossible to filter

## Solution

**What:** 4-way classification API:
1. **Human** — Genuinely human-written
2. **AI-Generated** — Direct LLM output
3. **Polished** — Human draft, AI-edited/improved
4. **Humanized** — AI-generated, then processed to evade detection

**How:** 
- Based on RACE paper techniques (Rhetorical Structure Theory analysis)
- Analyzes discourse patterns, not surface features (word frequency, perplexity)
- RST measures how ideas connect—humans and AI structure arguments differently
- Robust to paraphrasing and humanizer tools (attacks surface, not structure)

**Why Us:**
- Read the research, understand the technique
- Can implement RST parser + classifier
- Not constrained by existing detector baggage

## Why Now

1. **Humanizers broke existing detectors** — Market needs next-gen approach
2. **RACE paper published recently** — Technique is new, not yet productized
3. **Regulatory pressure** — EU AI Act, academic integrity policies tightening
4. **Enterprise demand** — HR, legal, publishing all need provenance
5. **Trust crisis is acute** — Every week a new AI-detection scandal

## Market Landscape

**TAM:** $5B (content authenticity and verification)
**SAM:** $800M (AI text detection)
**SOM:** $80M (advanced AI provenance, Year 3)

### Competitors & Gaps

| Competitor | What They Do | Gap |
|------------|--------------|-----|
| **GPTZero** | Perplexity-based detection | High false positives, beaten by humanizers |
| **Turnitin AI** | Integrated with plagiarism | Same weaknesses, institutional lock-in |
| **Originality.ai** | AI detection + plagiarism | Surface-level features, humanizer-vulnerable |
| **Copyleaks** | Enterprise content verification | Detection accuracy issues, no 4-way classification |
| **Winston AI** | AI content detection | Small, same approach as others |
| **Undetectable.ai** | Humanizer (adversary) | Shows market gap—they win because detectors lose |

**White space:** RST-based detection that survives humanization, with nuanced 4-way classification.

## Competitive Advantages

1. **Novel technique** — RST analysis is fundamentally different from perplexity
2. **Humanizer-resistant** — Attacks surface features, not discourse structure
3. **4-way classification** — Nuance that competitors don't offer
4. **Research foundation** — Built on peer-reviewed techniques
5. **First-mover on RST** — No one's productized this approach yet

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

## Build Plan

| Week | Milestone |
|------|-----------|
| 1-2 | Literature review, RST parser evaluation (feng-hirst vs. others) |
| 3-4 | Training data collection: human, AI, polished, humanized samples |
| 5-6 | RST feature extraction pipeline |
| 7-8 | Baseline classifier training, benchmark vs. GPTZero |
| 9-10 | Ensemble model (RST + BERT), calibration |
| 11-12 | API development, rate limiting, documentation |
| 13-14 | Humanizer stress testing (Undetectable.ai, QuillBot) |
| 15-16 | Beta launch, feedback collection, iteration |
| 17-18 | Production hardening, scale testing, GA launch |

**Team:** 1 ML engineer (you) + potential NLP contractor for RST parser

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| RST parsing is inaccurate | Medium | High | Use multiple parsers, ensemble, human validation |
| Humanizers adapt to RST | Medium | High | Continuous training, adversarial testing |
| 4-way classification is too hard | Medium | Medium | Start with 2-way (human/AI), add nuance later |
| Training data is expensive | High | Medium | Synthetic generation, crowdsourcing, academic partnerships |
| False positives cause harm | High | High | Conservative thresholds, detailed explanations |

## Monetization

**Model:** API usage-based

| Tier | Price | Includes |
|------|-------|----------|
| Free | $0 | 100 classifications/mo, rate limited |
| Starter | $99/mo | 5,000 classifications/mo, API key |
| Pro | $299/mo | 25,000 classifications/mo, batch API, webhooks |
| Enterprise | Custom | Unlimited, SLA, on-prem, custom models |

**Path to $1M ARR:**

- **Target:**
  - 200 Starter @ $99/mo = $238k ARR
  - 150 Pro @ $299/mo = $538k ARR
  - 10 Enterprise @ $2k/mo = $240k ARR
  - Total: ~$1M ARR

- **Funnel:**
  - Academic partnerships (5 universities for credibility)
  - Free tier for researchers, convert institutions
  - Content: "How humanizers defeat detectors" (SEO magnet)
  - Integration with LMS (Canvas, Blackboard)

- **Timeline:** 24 months post-launch (longer due to enterprise sales cycle)

## Verdict

### 🟢 BUILD

**Reasoning:**
1. **Clear market failure** — Existing detectors don't work, humanizers prove it
2. **Novel technique** — RST-based approach is genuinely differentiated
3. **Research backing** — RACE paper provides scientific foundation
4. **Multiple markets** — Academia, HR, publishing, legal
5. **High switching cost** — Once integrated into workflow, sticky

**Caveats:**
- RST parsing is technically challenging; validate approach early
- False positives are reputation-destroying; be conservative
- Long sales cycle for enterprise/academic customers
- Continuous arms race with humanizer tools

**First step:** 
1. Implement RST parser on sample texts
2. Collect small labeled dataset (100 samples per class)
3. Train baseline classifier
4. Test against Undetectable.ai output
5. If accuracy > 80% on humanized text, proceed to full build
