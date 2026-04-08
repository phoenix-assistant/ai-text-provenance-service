"""Dataset utilities for training the provenance classifier.

Handles loading, preprocessing, and augmentation of training data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import random

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from ai_text_provenance.features.extractor import FeatureExtractor
from ai_text_provenance.models.schemas import ProvenanceClass


# Label mapping
LABEL_TO_IDX = {
    ProvenanceClass.HUMAN: 0,
    ProvenanceClass.AI: 1,
    ProvenanceClass.POLISHED_HUMAN: 2,
    ProvenanceClass.HUMANIZED_AI: 3,
    "human": 0,
    "ai": 1,
    "polished_human": 2,
    "humanized_ai": 3,
}


class ProvenanceDataset(Dataset):
    """Dataset for text provenance classification.

    Expects data in JSONL format with 'text' and 'label' fields.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        precompute_features: bool = True,
        cache_path: Optional[str] = None,
    ):
        """Initialize the dataset.

        Args:
            data_path: Path to JSONL file with training data.
            tokenizer_name: HuggingFace tokenizer to use.
            max_length: Maximum sequence length.
            precompute_features: Precompute features for all examples.
            cache_path: Path to cache precomputed features.
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.precompute_features = precompute_features

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load data
        self.examples = []
        with open(self.data_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                if "text" in example and "label" in example:
                    self.examples.append(example)

        if not self.examples:
            raise ValueError(f"No valid examples found in {data_path}")

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()

        # Precompute features if requested
        self.cached_features = {}
        if precompute_features:
            self._precompute_features(cache_path)

    def _precompute_features(self, cache_path: Optional[str] = None) -> None:
        """Precompute features for all examples."""
        cache_file = Path(cache_path) if cache_path else None

        # Try to load from cache
        if cache_file and cache_file.exists():
            with open(cache_file, "r") as f:
                self.cached_features = json.load(f)
            return

        # Compute features
        print(f"Precomputing features for {len(self.examples)} examples...")

        for i, example in enumerate(self.examples):
            features = self.feature_extractor.extract(example["text"])
            feature_vector = self.feature_extractor.to_vector(features)
            self.cached_features[str(i)] = feature_vector

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(self.examples)}")

        # Save to cache
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(self.cached_features, f)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        # Tokenize text
        encoding = self.tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Get features
        if str(idx) in self.cached_features:
            feature_vector = self.cached_features[str(idx)]
        else:
            features = self.feature_extractor.extract(example["text"])
            feature_vector = self.feature_extractor.to_vector(features)

        # Get label
        label = example["label"]
        if isinstance(label, str):
            label_idx = LABEL_TO_IDX.get(label.lower(), 0)
        else:
            label_idx = label

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "features": torch.tensor(feature_vector, dtype=torch.float32),
            "label": torch.tensor(label_idx, dtype=torch.long),
        }

    @staticmethod
    def create_sample_dataset(output_path: str, num_samples: int = 100) -> None:
        """Create a sample dataset for testing.

        Generates synthetic examples for each class.

        Args:
            output_path: Path to save the JSONL file.
            num_samples: Number of samples per class.
        """
        samples = []

        # Human-written samples (varied, personal, imperfect)
        human_templates = [
            "So I was thinking about {} the other day, and honestly? It's pretty wild how {} works. Like, {} is something I never really got until {}.",
            "OK so here's the thing about {} - most people don't realize that {} is actually connected to {}. I learned this the hard way when {}.",
            "You know what bugs me? When {} happens and nobody talks about {}. It's like {} doesn't even matter to {}.",
            "Been researching {} lately and stumbled on something interesting about {}. Turns out {} has a lot to do with {}.",
        ]

        # AI-generated samples (smooth, structured, comprehensive)
        ai_templates = [
            "{} is a fascinating topic that encompasses several key aspects. First, it's important to understand that {}. Additionally, {} plays a crucial role in {}. Furthermore, {} contributes significantly to {}.",
            "When examining {}, we must consider multiple factors. The primary consideration is {}. Equally important is {}. These elements combine to create {}.",
            "Understanding {} requires a comprehensive approach. {} forms the foundation, while {} builds upon it. Together, they demonstrate that {}.",
            "{} represents a complex phenomenon with various dimensions. The first dimension involves {}. The second concerns {}. Finally, {} ties everything together.",
        ]

        # Polished human (edited, refined, but still personal)
        polished_templates = [
            "After careful consideration of {}, I've come to appreciate the nuance of {}. While initially {} seemed straightforward, deeper analysis reveals {}.",
            "My experience with {} taught me valuable lessons about {}. The key insight was that {} directly influences {}.",
            "Reflecting on {}, several patterns emerge regarding {}. Most notably, {} correlates strongly with {}.",
            "Having worked extensively with {}, I can confirm that {} is essential for {}. This became clear when {}.",
        ]

        # Humanized AI (awkward variety, forced imperfections)
        humanized_templates = [
            "So basically {} is really something, you know? I mean {} has all these aspects like {}. Anyway the point is that {} matters a lot I think.",
            "Honestly speaking about {}, it's kind of complicated but {}. You might say {} is related to {} in some ways.",
            "From what I understand about {}, there's a lot to unpack with {}. Basically {} connects to {} somehow.",
            "Well {} is interesting actually. I've been thinking that {} and also {}. Makes sense when you consider {} right?",
        ]

        topics = [
            ("machine learning", "neural networks", "data patterns", "training models"),
            ("climate change", "carbon emissions", "renewable energy", "environmental policy"),
            ("remote work", "productivity", "work-life balance", "team collaboration"),
            ("social media", "mental health", "online communities", "digital wellness"),
            ("cryptocurrency", "blockchain technology", "decentralized finance", "market volatility"),
        ]

        for _ in range(num_samples):
            topic = random.choice(topics)

            # Human sample
            template = random.choice(human_templates)
            samples.append({
                "text": template.format(*random.sample(topic, 4)),
                "label": "human",
            })

            # AI sample
            template = random.choice(ai_templates)
            samples.append({
                "text": template.format(*random.sample(topic, 4)),
                "label": "ai",
            })

            # Polished sample
            template = random.choice(polished_templates)
            samples.append({
                "text": template.format(*random.sample(topic, 4)),
                "label": "polished_human",
            })

            # Humanized sample
            template = random.choice(humanized_templates)
            samples.append({
                "text": template.format(*random.sample(topic, 4)),
                "label": "humanized_ai",
            })

        # Shuffle and write
        random.shuffle(samples)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        print(f"Created sample dataset with {len(samples)} examples at {output_path}")
