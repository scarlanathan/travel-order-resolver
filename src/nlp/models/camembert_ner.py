"""
CamemBERT NER Model for Origin/Destination Extraction

Fine-tuned CamemBERT (French BERT) for token classification.
"""

import os
import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    CamembertTokenizerFast,
    CamembertForTokenClassification,
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForTokenClassification,
)
import numpy as np


@dataclass
class CamembertExtractionResult:
    """Result from CamemBERT NER extraction."""
    origin: Optional[str] = None
    destination: Optional[str] = None
    is_valid: bool = False
    confidence: float = 0.0
    method: str = "camembert_ner"


# Label mapping
LABEL2ID = {
    "O": 0,
    "B-ORIGIN": 1,
    "I-ORIGIN": 2,
    "B-DESTINATION": 3,
    "I-DESTINATION": 4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class LossLoggerCallback(TrainerCallback):
    """Callback to log train and eval losses per epoch."""

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])


class NERDataset(Dataset):
    """Dataset for NER token classification."""

    def __init__(
        self,
        csv_path: str,
        tokenizer: CamembertTokenizerFast,
        max_length: int = 128,
        limit: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                self.samples.append({
                    'sentence': row['sentence'],
                    'origin': row['origin'],
                    'destination': row['destination'],
                })

    def __len__(self):
        return len(self.samples)

    def _find_entity_char_spans(
        self,
        text: str,
        origin: str,
        destination: str
    ) -> List[Tuple[int, int, str]]:
        """Find character spans for entities."""
        entities = []
        text_lower = text.lower()

        origin_span = None
        dest_span = None

        # Find origin
        if origin and origin != "INVALID":
            origin_lower = origin.lower()
            start = text_lower.find(origin_lower)
            if start != -1:
                origin_span = (start, start + len(origin_lower), "ORIGIN")

        # Find destination (search entire text, not just after origin)
        if destination and destination != "INVALID":
            dest_lower = destination.lower()
            start = text_lower.find(dest_lower)
            if start != -1:
                # Check for overlap with origin
                if origin_span:
                    o_start, o_end, _ = origin_span
                    d_end = start + len(dest_lower)
                    # If overlapping, try to find another occurrence
                    if not (d_end <= o_start or start >= o_end):
                        # Overlapping - search after origin
                        start = text_lower.find(dest_lower, o_end)
                if start != -1:
                    dest_span = (start, start + len(dest_lower), "DESTINATION")

        # Add non-None spans
        if origin_span:
            entities.append(origin_span)
        if dest_span:
            entities.append(dest_span)

        return entities

    def _align_labels_with_tokens(
        self,
        tokenized,
        char_spans: List[Tuple[int, int, str]]
    ) -> List[int]:
        """Align character-level entity spans with tokenized output."""
        labels = []
        previous_word_idx = None
        entity_started = {}  # Track which entities have started

        for word_idx in tokenized.word_ids():
            if word_idx is None:
                labels.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                # First token of a word
                word_span = tokenized.word_to_chars(word_idx)
                token_start = word_span.start
                token_end = word_span.end

                label = "O"
                for char_start, char_end, entity_type in char_spans:
                    # Check for overlap (not strict containment)
                    # Token overlaps with entity if: token_start < char_end AND token_end > char_start
                    if token_start < char_end and token_end > char_start:
                        if entity_type not in entity_started:
                            label = f"B-{entity_type}"
                            entity_started[entity_type] = True
                        else:
                            label = f"I-{entity_type}"
                        break

                labels.append(LABEL2ID[label])
            else:
                # Continuation of a word (subword token)
                prev_label_id = labels[-1]
                if prev_label_id == -100 or prev_label_id == 0:
                    labels.append(LABEL2ID["O"])
                else:
                    prev_label = ID2LABEL[prev_label_id]
                    if prev_label.startswith("B-"):
                        entity_type = prev_label[2:]
                        labels.append(LABEL2ID[f"I-{entity_type}"])
                    else:
                        labels.append(prev_label_id)

            previous_word_idx = word_idx

        return labels

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Tokenize
        tokenized = self.tokenizer(
            sample['sentence'],
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        # Find entity spans
        char_spans = self._find_entity_char_spans(
            sample['sentence'],
            sample['origin'],
            sample['destination']
        )

        # Align labels
        labels = self._align_labels_with_tokens(tokenized, char_spans)

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels,
        }


class CamembertNERTrainer:
    """Trainer for CamemBERT NER model."""

    # Known CamemBERT-compatible models
    CAMEMBERT_MODELS = {"camembert-base", "camembert/camembert-base-ccnet"}

    def __init__(
        self,
        model_name: str = "camembert-base",
        output_dir: str = "models/camembert_ner",
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use AutoTokenizer for non-standard model names (e.g. distilcamembert)
        if model_name in self.CAMEMBERT_MODELS:
            self.tokenizer = CamembertTokenizerFast.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None

    def _init_model(self):
        """Initialize model for training."""
        if self.model_name in self.CAMEMBERT_MODELS:
            self.model = CamembertForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(LABEL2ID),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(LABEL2ID),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )

    def train(
        self,
        train_csv: str,
        val_csv: str,
        train_limit: Optional[int] = None,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        lr_scheduler_type: str = "linear",
        output_name: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Train CamemBERT NER model.

        Args:
            train_csv: Path to training CSV
            val_csv: Path to validation CSV
            train_limit: Limit training samples
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            lr_scheduler_type: LR scheduler type ('linear', 'cosine', etc.)
            output_name: Custom output name

        Returns:
            Tuple of (model_path, training_info)
        """
        # Generate output name
        if output_name is None:
            size_str = str(train_limit) if train_limit else "full"
            output_name = f"camembert_ner_{size_str}"

        model_path = self.output_dir / output_name

        print(f"\n{'='*60}")
        print(f"Training: {output_name}")
        print(f"  Base model: {self.model_name}")
        print(f"  Training samples: {train_limit if train_limit else 'all'}")
        print(f"  Epochs: {num_epochs}")
        print(f"  LR: {learning_rate}, Scheduler: {lr_scheduler_type}, Batch: {batch_size}")
        print(f"{'='*60}\n")

        # Initialize model
        self._init_model()

        # Load datasets
        train_dataset = NERDataset(train_csv, self.tokenizer, limit=train_limit)
        val_dataset = NERDataset(val_csv, self.tokenizer)

        print(f"Loaded {len(train_dataset)} training samples")
        print(f"Loaded {len(val_dataset)} validation samples")

        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
        )

        # Loss logger callback
        loss_logger = LossLoggerCallback()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(model_path),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            push_to_hub=False,
            logging_steps=50,
            report_to="none",
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            callbacks=[loss_logger],
        )

        # Train
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        # Save model
        trainer.save_model(str(model_path))
        self.tokenizer.save_pretrained(str(model_path))

        print(f"\nModel saved to {model_path}")

        # Training info
        info = {
            "model_name": output_name,
            "base_model": self.model_name,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lr_scheduler_type": lr_scheduler_type,
            "training_time_s": training_time,
            "model_path": str(model_path),
            "train_losses": loss_logger.train_losses,
            "eval_losses": loss_logger.eval_losses,
        }

        return str(model_path), info


class CamembertNERModel:
    """CamemBERT NER model for inference."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        try:
            self.tokenizer = CamembertTokenizerFast.from_pretrained(model_path)
            self.model = CamembertForTokenClassification.from_pretrained(model_path)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def _clean_entity(self, text: str) -> str:
        """Clean extracted entity text by removing unwanted characters."""
        if not text:
            return text
        # Strip leading/trailing whitespace and punctuation
        text = text.strip()
        # Remove trailing punctuation (but keep hyphens within the text)
        while text and text[-1] in '.,;:!?':
            text = text[:-1]
        return text.strip()
    def predict(self, sentence: str) -> CamembertExtractionResult:
        """Extract origin and destination from sentence."""
        # Tokenize
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=128,
        )

        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
            probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()

        # Extract entities from predictions
        origin = None
        destination = None
        origin_conf = 0.0
        dest_conf = 0.0

        current_entity = None
        current_start = None
        current_end = None
        current_probs = []

        for idx, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == end:  # Special token
                continue

            label = ID2LABEL.get(pred, "O")

            if label.startswith("B-"):
                # Save previous entity
                if current_entity and current_start is not None:
                    entity_text = self._clean_entity(sentence[current_start:current_end])

                    avg_conf = sum(current_probs) / len(current_probs) if current_probs else 0
                    if current_entity == "ORIGIN" and origin is None:
                        origin = entity_text
                        origin_conf = avg_conf
                    elif current_entity == "DESTINATION" and destination is None:
                        destination = entity_text
                        dest_conf = avg_conf

                # Start new entity
                current_entity = label[2:]
                current_start = start
                current_end = end
                current_probs = [probabilities[idx][pred]]

            elif label.startswith("I-") and current_entity == label[2:]:
                # Continue entity
                current_end = end
                current_probs.append(probabilities[idx][pred])

            else:
                # End entity
                if current_entity and current_start is not None:
                    entity_text = self._clean_entity(sentence[current_start:current_end])
                    avg_conf = sum(current_probs) / len(current_probs) if current_probs else 0
                    if current_entity == "ORIGIN" and origin is None:
                        origin = entity_text
                        origin_conf = avg_conf
                    elif current_entity == "DESTINATION" and destination is None:
                        destination = entity_text
                        dest_conf = avg_conf

                current_entity = None
                current_start = None
                current_end = None
                current_probs = []

        # Handle last entity
        if current_entity and current_start is not None:
            entity_text = self._clean_entity(sentence[current_start:current_end])
            avg_conf = sum(current_probs) / len(current_probs) if current_probs else 0
            if current_entity == "ORIGIN" and origin is None:
                origin = entity_text
                origin_conf = avg_conf
            elif current_entity == "DESTINATION" and destination is None:
                destination = entity_text
                dest_conf = avg_conf

        is_valid = origin is not None and destination is not None
        confidence = (origin_conf + dest_conf) / 2 if is_valid else 0.0

        return CamembertExtractionResult(
            origin=origin,
            destination=destination,
            is_valid=is_valid,
            confidence=confidence,
            method="camembert_ner"
        )


if __name__ == "__main__":
    print("CamemBERT NER Module")
    print("Use CamembertNERTrainer to train models")
    print("Use CamembertNERModel to load and use trained models")
