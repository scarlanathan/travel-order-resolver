"""
spaCy NER Model for Origin/Destination Extraction

This module provides:
1. Dataset conversion to spaCy format
2. Model training with configurable parameters
3. Evaluation and comparison utilities
"""

import spacy
from spacy.tokens import DocBin, Doc
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
import time

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


@dataclass
class SpacyExtractionResult:
    """Result from spaCy NER extraction."""
    origin: Optional[str] = None
    destination: Optional[str] = None
    is_valid: bool = False
    confidence: float = 0.0
    method: str = "spacy_ner"


class SpacyNERTrainer:
    """
    Trainer for spaCy NER models.

    Supports training with:
    - Different base models (blank, sm, md, lg)
    - Different dataset sizes
    - Configurable training parameters
    """

    def __init__(
        self,
        base_model: str = "fr_core_news_sm",
        output_dir: str = "models/spacy_ner",
    ):
        """
        Initialize trainer.

        Args:
            base_model: Base spaCy model to use
                - "blank" for spacy.blank("fr")
                - "fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg"
            output_dir: Directory to save trained models
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # NER labels
        self.labels = ["ORIGIN", "DESTINATION"]

    def _load_base_model(self) -> spacy.Language:
        """Load the base spaCy model."""
        if self.base_model == "blank":
            nlp = spacy.blank("fr")
        else:
            try:
                nlp = spacy.load(self.base_model)
            except OSError:
                print(f"Model {self.base_model} not found. Downloading...")
                spacy.cli.download(self.base_model)
                nlp = spacy.load(self.base_model)
        return nlp

    def _find_entity_spans(
        self,
        text: str,
        origin: str,
        destination: str
    ) -> List[Tuple[int, int, str]]:
        """
        Find character spans for origin and destination in text.

        Returns list of (start, end, label) tuples.
        """
        entities = []
        text_lower = text.lower()

        # Find origin
        if origin and origin != "INVALID":
            origin_lower = origin.lower()
            start = text_lower.find(origin_lower)
            if start != -1:
                # Get the actual text (preserving original case)
                end = start + len(origin)
                entities.append((start, end, "ORIGIN"))

        # Find destination
        if destination and destination != "INVALID":
            dest_lower = destination.lower()
            # Search from after origin if possible to avoid overlap
            search_start = 0
            if entities:
                search_start = entities[0][1]  # After origin

            start = text_lower.find(dest_lower, search_start)
            if start == -1:
                # Try from beginning if not found after origin
                start = text_lower.find(dest_lower)

            if start != -1:
                end = start + len(destination)
                # Check for overlap with origin
                if not entities or not (entities[0][0] <= start < entities[0][1]):
                    entities.append((start, end, "DESTINATION"))

        return entities

    def convert_csv_to_spacy(
        self,
        csv_path: str,
        output_path: str,
        limit: Optional[int] = None
    ) -> int:
        """
        Convert CSV dataset to spaCy DocBin format.

        Args:
            csv_path: Path to CSV file with columns: sentence, origin, destination
            output_path: Path to save .spacy file
            limit: Maximum number of samples to convert

        Returns:
            Number of samples converted
        """
        nlp = spacy.blank("fr")
        doc_bin = DocBin()

        converted = 0
        skipped = 0

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break

                sentence = row['sentence']
                origin = row['origin']
                destination = row['destination']

                # Find entity spans
                entities = self._find_entity_spans(sentence, origin, destination)

                # Create Doc with entities
                doc = nlp.make_doc(sentence)
                ents = []

                for start, end, label in entities:
                    span = doc.char_span(start, end, label=label)
                    if span is not None:
                        ents.append(span)

                # Only add if we found at least one entity or it's invalid
                if ents or (origin == "INVALID" and destination == "INVALID"):
                    doc.ents = ents
                    doc_bin.add(doc)
                    converted += 1
                else:
                    skipped += 1

        # Save DocBin
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        doc_bin.to_disk(output_path)

        print(f"Converted {converted} samples to {output_path}")
        print(f"Skipped {skipped} samples (entities not found in text)")

        return converted

    def prepare_training_data(
        self,
        train_csv: str,
        val_csv: str,
        output_dir: str = "data/spacy",
        train_limit: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Prepare training and validation data in spaCy format.

        Args:
            train_csv: Path to training CSV
            val_csv: Path to validation CSV
            output_dir: Directory to save .spacy files
            train_limit: Limit training samples (for experiments)

        Returns:
            Tuple of (train_path, val_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"_{train_limit}" if train_limit else ""
        train_path = str(output_dir / f"train{suffix}.spacy")
        val_path = str(output_dir / "val.spacy")

        self.convert_csv_to_spacy(train_csv, train_path, limit=train_limit)
        self.convert_csv_to_spacy(val_csv, val_path)

        return train_path, val_path

    def create_config(
        self,
        train_path: str,
        val_path: str,
        output_path: str,
        n_iter: int = 30,
        batch_size: int = 128,
        learn_rate: float = 0.001,
    ) -> str:
        """
        Create spaCy training config file.

        Args:
            train_path: Path to training .spacy file
            val_path: Path to validation .spacy file
            output_path: Path to save config
            n_iter: Number of training iterations
            batch_size: Batch size
            learn_rate: Learning rate

        Returns:
            Path to config file
        """
        # Determine base model config
        if self.base_model == "blank":
            base_config = """
[paths]
train = "{train_path}"
dev = "{val_path}"

[system]
gpu_allocator = null

[nlp]
lang = "fr"
pipeline = ["tok2vec", "ner"]

[components]

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v2"

[components.tok2vec.model.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 96
attrs = ["ORTH", "SHAPE", "PREFIX", "SUFFIX"]
rows = [5000, 2500, 1000, 1000]
include_static_vectors = false

[components.tok2vec.model.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 96
depth = 4
window_size = 1
maxout_pieces = 3

[components.ner]
factory = "ner"

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = true
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = ${{components.tok2vec.model.encode.width}}

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
max_epochs = {n_iter}
seed = 42

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
size = {batch_size}

[training.optimizer]
@optimizers = "Adam.v1"
learn_rate = {learn_rate}

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${{paths.train}}

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${{paths.dev}}
"""
        else:
            base_config = """
[paths]
train = "{train_path}"
dev = "{val_path}"

[system]
gpu_allocator = null

[nlp]
lang = "fr"
pipeline = ["tok2vec", "ner"]
batch_size = {batch_size}

[nlp.tokenizer]
@tokenizers = "spacy.Tokenizer.v1"

[components]

[components.tok2vec]
source = "{base_model}"

[components.ner]
factory = "ner"

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = true
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = 96

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
max_epochs = {n_iter}
seed = 42

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
size = {batch_size}

[training.optimizer]
@optimizers = "Adam.v1"
learn_rate = {learn_rate}

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${{paths.train}}

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${{paths.dev}}
"""

        config_content = base_config.format(
            train_path=train_path.replace("\\", "/"),
            val_path=val_path.replace("\\", "/"),
            base_model=self.base_model,
            n_iter=n_iter,
            batch_size=batch_size,
            learn_rate=learn_rate,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(config_content)

        return output_path

    def train_simple(
        self,
        train_data: List[Tuple[str, Dict]],
        val_data: List[Tuple[str, Dict]],
        n_iter: int = 30,
        dropout: float = 0.35,
    ) -> Tuple[spacy.Language, List[float]]:
        """
        Train NER model using simple API (no config file).

        Args:
            train_data: List of (text, {"entities": [(start, end, label), ...]})
            val_data: Validation data in same format
            n_iter: Number of iterations
            dropout: Dropout rate

        Returns:
            Tuple of (trained spaCy model, loss history per iteration)
        """
        nlp = self._load_base_model()

        # Add NER component if not present
        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")

        # Add labels
        for label in self.labels:
            ner.add_label(label)

        # Get pipes to disable during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

        loss_history = []

        # Training
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.begin_training()

            for iteration in tqdm(range(n_iter), desc="Training spaCy NER"):
                random.shuffle(train_data)
                losses = {}

                # Create batches
                batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))

                for batch in batches:
                    examples = []
                    for text, annotations in batch:
                        doc = nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)

                    nlp.update(examples, drop=dropout, losses=losses)

                iter_loss = losses.get('ner', 0)
                loss_history.append(iter_loss)

                if (iteration + 1) % 5 == 0:
                    print(f"Iteration {iteration + 1}/{n_iter} - Loss: {iter_loss:.4f}")

        return nlp, loss_history

    def load_training_data_from_csv(
        self,
        csv_path: str,
        limit: Optional[int] = None
    ) -> List[Tuple[str, Dict]]:
        """
        Load training data from CSV for simple training API.

        Args:
            csv_path: Path to CSV file
            limit: Maximum samples to load

        Returns:
            List of (text, {"entities": [...]}) tuples
        """
        data = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break

                sentence = row['sentence']
                origin = row['origin']
                destination = row['destination']

                entities = self._find_entity_spans(sentence, origin, destination)

                if entities or (origin == "INVALID" and destination == "INVALID"):
                    data.append((sentence, {"entities": entities}))

        return data

    def save_model(self, nlp: spacy.Language, model_name: str) -> str:
        """Save trained model to disk."""
        model_path = self.output_dir / model_name
        nlp.to_disk(model_path)
        print(f"Model saved to {model_path}")
        return str(model_path)


class SpacyNERModel:
    """
    spaCy NER model for inference.

    Compatible with the evaluator interface.
    """

    def __init__(self, model_path: str):
        """
        Load trained spaCy model.

        Args:
            model_path: Path to saved spaCy model
        """
        self.nlp = spacy.load(model_path)
        self.model_path = model_path

    def predict(self, sentence: str) -> SpacyExtractionResult:
        """
        Extract origin and destination from sentence.

        Args:
            sentence: Input sentence

        Returns:
            SpacyExtractionResult with extracted entities
        """
        doc = self.nlp(sentence)

        origin = None
        destination = None

        for ent in doc.ents:
            if ent.label_ == "ORIGIN" and origin is None:
                origin = ent.text
            elif ent.label_ == "DESTINATION" and destination is None:
                destination = ent.text

        is_valid = origin is not None and destination is not None

        # Confidence based on number of entities found
        confidence = 0.0
        if origin:
            confidence += 0.5
        if destination:
            confidence += 0.5

        return SpacyExtractionResult(
            origin=origin,
            destination=destination,
            is_valid=is_valid,
            confidence=confidence,
            method="spacy_ner"
        )


def train_model_experiment(
    base_model: str,
    train_limit: Optional[int],
    n_iter: int = 30,
    output_name: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Run a single training experiment.

    Args:
        base_model: Base model to use
        train_limit: Training data limit
        n_iter: Number of iterations
        output_name: Custom output name

    Returns:
        Tuple of (model_path, training_info)
    """
    # Generate model name
    if output_name is None:
        size_str = str(train_limit) if train_limit else "full"
        base_str = base_model.replace("fr_core_news_", "").replace("spacy.", "")
        output_name = f"ner_{base_str}_{size_str}"

    print(f"\n{'='*60}")
    print(f"Training: {output_name}")
    print(f"  Base model: {base_model}")
    print(f"  Training samples: {train_limit if train_limit else 'all'}")
    print(f"  Iterations: {n_iter}")
    print(f"{'='*60}\n")

    # Initialize trainer
    trainer = SpacyNERTrainer(base_model=base_model)

    # Load data
    train_data = trainer.load_training_data_from_csv(
        "data/training/train.csv",
        limit=train_limit
    )
    val_data = trainer.load_training_data_from_csv(
        "data/training/val.csv"
    )

    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(val_data)} validation samples")

    # Train
    start_time = time.time()
    nlp, loss_history = trainer.train_simple(train_data, val_data, n_iter=n_iter)
    training_time = time.time() - start_time

    # Save model
    model_path = trainer.save_model(nlp, output_name)

    # Training info
    info = {
        "model_name": output_name,
        "base_model": base_model,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "n_iter": n_iter,
        "training_time_s": training_time,
        "model_path": model_path,
        "loss_history": [float(l) for l in loss_history],
    }

    return model_path, info


if __name__ == "__main__":
    # Example: Train a simple model
    print("SpaCy NER Training Module")
    print("Use train_model_experiment() to train models")
    print("Use SpacyNERModel(path) to load and use trained models")
