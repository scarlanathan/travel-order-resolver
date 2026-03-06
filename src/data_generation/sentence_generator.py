"""
Sentence Generator for Training Data

Generates ~10,000 diverse French sentences for training the NLP model.
Includes variations: no capitals, misspellings, no accents, etc.
"""

import random
import re
from typing import List, Tuple, Dict
from pathlib import Path
import pandas as pd
from loguru import logger

from src.data_generation.templates import (
    get_all_templates,
    get_invalid_templates,
    get_cities,
    get_names,
    INTERMEDIATE_TEMPLATES,
    SMS_TEMPLATES,
)


class SentenceGenerator:
    """Generate training sentences with variations."""

    def __init__(self, num_sentences: int = 10000, seed: int = 42):
        """
        Initialize the sentence generator.

        Args:
            num_sentences: Number of sentences to generate
            seed: Random seed for reproducibility
        """
        self.num_sentences = num_sentences
        self.seed = seed
        random.seed(seed)

        self.templates = get_all_templates()
        self.invalid_templates = get_invalid_templates()
        self.cities = get_cities()
        self.names = get_names()

        logger.info(f"Initialized generator with {len(self.templates)} templates")
        logger.info(f"Using {len(self.cities)} cities")

    def generate_valid_sentence(self) -> Tuple[str, str, str]:
        """
        Generate a valid travel order sentence.

        Returns:
            Tuple of (sentence, origin, destination)
        """
        template = random.choice(self.templates)

        # Select origin and destination (must be different)
        origin = random.choice(self.cities)
        destination = random.choice([c for c in self.cities if c != origin])

        # Handle templates with intermediate stops
        if "{intermediate}" in template:
            intermediate = random.choice(
                [c for c in self.cities if c not in [origin, destination]]
            )
            sentence = template.format(
                origin=origin,
                destination=destination,
                intermediate=intermediate
            )
        # Handle templates with names
        elif "{name}" in template or "{name1}" in template:
            name = random.choice(self.names)
            name1 = random.choice(self.names)
            name2 = random.choice([n for n in self.names if n != name1])

            sentence = template.format(
                origin=origin,
                destination=destination,
                name=name,
                name1=name1,
                name2=name2,
            )
        # Regular templates
        else:
            sentence = template.format(origin=origin, destination=destination)

        return sentence, origin, destination

    def generate_invalid_sentence(self) -> Tuple[str, str, str]:
        """
        Generate an invalid sentence (not a travel order).

        Returns:
            Tuple of (sentence, "INVALID", "INVALID")
        """
        template = random.choice(self.invalid_templates)

        # Some templates have placeholders
        if "{city}" in template:
            city = random.choice(self.cities)
            sentence = template.format(city=city)
        else:
            sentence = template

        return sentence, "INVALID", "INVALID"

    def remove_capitals(self, text: str) -> str:
        """Remove all capital letters."""
        return text.lower()

    def remove_accents(self, text: str) -> str:
        """Remove French accents."""
        accents = {
            'à': 'a', 'â': 'a', 'ä': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'î': 'i', 'ï': 'i',
            'ô': 'o', 'ö': 'o',
            'ù': 'u', 'û': 'u', 'ü': 'u',
            'ÿ': 'y',
            'ç': 'c',
            'À': 'A', 'Â': 'A', 'Ä': 'A',
            'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
            'Î': 'I', 'Ï': 'I',
            'Ô': 'O', 'Ö': 'O',
            'Ù': 'U', 'Û': 'U', 'Ü': 'U',
            'Ÿ': 'Y',
            'Ç': 'C',
        }
        for accent, replacement in accents.items():
            text = text.replace(accent, replacement)
        return text

    def add_misspellings(self, text: str) -> str:
        """Add random misspellings to the text."""
        words = text.split()
        num_errors = max(1, len(words) // 10)  # ~10% of words

        for _ in range(num_errors):
            if len(words) < 2:
                break

            idx = random.randint(0, len(words) - 1)
            word = words[idx]

            if len(word) < 3:
                continue

            # Random misspelling types
            error_type = random.choice(['swap', 'duplicate', 'delete', 'replace'])

            if error_type == 'swap' and len(word) > 2:
                # Swap two adjacent letters
                pos = random.randint(0, len(word) - 2)
                word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]

            elif error_type == 'duplicate':
                # Duplicate a letter
                pos = random.randint(0, len(word) - 1)
                word = word[:pos] + word[pos] + word[pos:]

            elif error_type == 'delete' and len(word) > 3:
                # Delete a letter
                pos = random.randint(1, len(word) - 2)
                word = word[:pos] + word[pos + 1:]

            elif error_type == 'replace':
                # Replace with nearby keyboard letter
                pos = random.randint(0, len(word) - 1)
                nearby = {'a': 'z', 'e': 'r', 'i': 'o', 'o': 'p', 'u': 'y'}
                if word[pos].lower() in nearby:
                    word = word[:pos] + nearby[word[pos].lower()] + word[pos + 1:]

            words[idx] = word

        return ' '.join(words)

    def remove_hyphens(self, text: str) -> str:
        """Remove hyphens from city names."""
        return text.replace('-', ' ')

    def apply_variations(
        self,
        sentence: str,
        origin: str,
        destination: str,
    ) -> List[Tuple[str, str, str]]:
        """
        Apply various transformations to create dataset diversity.

        Args:
            sentence: Original sentence
            origin: Origin city
            destination: Destination city

        Returns:
            List of (sentence, origin, destination) tuples
        """
        variations = [(sentence, origin, destination)]

        # No capitals
        if random.random() < 0.3:
            variations.append((
                self.remove_capitals(sentence),
                origin,
                destination
            ))

        # No accents
        if random.random() < 0.2:
            variations.append((
                self.remove_accents(sentence),
                origin,
                destination
            ))

        # No capitals + no accents
        if random.random() < 0.15:
            variations.append((
                self.remove_accents(self.remove_capitals(sentence)),
                origin,
                destination
            ))

        # Misspellings
        if random.random() < 0.1:
            variations.append((
                self.add_misspellings(sentence),
                origin,
                destination
            ))

        # No hyphens
        if random.random() < 0.1:
            variations.append((
                self.remove_hyphens(sentence),
                origin,
                destination
            ))

        return variations

    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate complete dataset with variations.

        Returns:
            DataFrame with columns: sentence_id, sentence, origin, destination
        """
        logger.info(f"Generating {self.num_sentences} sentences...")

        data = []
        sentence_id = 1

        # Generate 85% valid, 15% invalid
        num_valid = int(self.num_sentences * 0.85)
        num_invalid = self.num_sentences - num_valid

        # Generate valid sentences
        valid_count = 0
        while valid_count < num_valid:
            sentence, origin, destination = self.generate_valid_sentence()

            # Apply variations
            variations = self.apply_variations(sentence, origin, destination)

            for var_sentence, var_origin, var_dest in variations:
                if valid_count >= num_valid:
                    break

                data.append({
                    'sentence_id': sentence_id,
                    'sentence': var_sentence,
                    'origin': var_origin,
                    'destination': var_dest,
                })
                sentence_id += 1
                valid_count += 1

        # Generate invalid sentences
        for _ in range(num_invalid):
            sentence, origin, destination = self.generate_invalid_sentence()
            data.append({
                'sentence_id': sentence_id,
                'sentence': sentence,
                'origin': origin,
                'destination': destination,
            })
            sentence_id += 1

        df = pd.DataFrame(data)

        # Shuffle
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        # Update sentence_id to be sequential
        df['sentence_id'] = range(1, len(df) + 1)

        logger.info(f"Generated {len(df)} sentences")
        logger.info(f"Valid: {len(df[df['origin'] != 'INVALID'])}")
        logger.info(f"Invalid: {len(df[df['origin'] == 'INVALID'])}")

        return df

    def split_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/val/test sets with disjoint city partitions.

        Cities are split into 3 disjoint sets (70/15/15) so that train, val,
        and test sets contain sentences with different cities. Some SMS templates
        are reserved for the test set only.

        Args:
            df: Full dataset
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5

        # Get all unique cities from valid sentences
        valid_df = df[df['origin'] != 'INVALID']
        invalid_df = df[df['origin'] == 'INVALID']

        all_cities = list(set(valid_df['origin'].tolist() + valid_df['destination'].tolist()))
        random.shuffle(all_cities)

        # Partition cities into 3 disjoint sets
        n_cities = len(all_cities)
        n_train = max(1, int(n_cities * train_ratio))
        n_val = max(1, int(n_cities * val_ratio))

        train_cities = set(all_cities[:n_train])
        val_cities = set(all_cities[n_train:n_train + n_val])
        test_cities = set(all_cities[n_train + n_val:])

        # Ensure test has at least some cities
        if not test_cities and val_cities:
            moved = val_cities.pop()
            test_cities.add(moved)

        logger.info(f"City split: train={len(train_cities)}, val={len(val_cities)}, test={len(test_cities)}")

        # Identify SMS template sentences for test reservation
        sms_patterns = set()
        for tmpl in SMS_TEMPLATES:
            # Extract the static part of the template (first 10 chars before placeholder)
            prefix = tmpl.split("{")[0].strip().lower()
            if prefix:
                sms_patterns.add(prefix)

        def is_sms_sentence(sentence: str) -> bool:
            s_lower = sentence.lower().strip()
            return any(s_lower.startswith(p) for p in sms_patterns)

        # Assign valid sentences based on city membership
        train_mask = valid_df.apply(
            lambda r: r['origin'] in train_cities and r['destination'] in train_cities,
            axis=1,
        )
        val_mask = valid_df.apply(
            lambda r: r['origin'] in val_cities and r['destination'] in val_cities,
            axis=1,
        )
        test_mask = valid_df.apply(
            lambda r: r['origin'] in test_cities and r['destination'] in test_cities,
            axis=1,
        )

        # Sentences with cross-partition cities go to train (largest set)
        unassigned_mask = ~(train_mask | val_mask | test_mask)

        # SMS sentences go to test when possible
        sms_mask = valid_df['sentence'].apply(is_sms_sentence)
        sms_to_test = valid_df[sms_mask & unassigned_mask]
        remaining_unassigned = valid_df[unassigned_mask & ~sms_mask]

        train_valid = pd.concat([valid_df[train_mask], remaining_unassigned])
        val_valid = valid_df[val_mask]
        test_valid = pd.concat([valid_df[test_mask], sms_to_test])

        # Split invalid sentences proportionally
        invalid_shuffled = invalid_df.sample(frac=1, random_state=self.seed)
        n_inv = len(invalid_shuffled)
        n_inv_train = int(n_inv * train_ratio)
        n_inv_val = int(n_inv * val_ratio)

        train_invalid = invalid_shuffled.iloc[:n_inv_train]
        val_invalid = invalid_shuffled.iloc[n_inv_train:n_inv_train + n_inv_val]
        test_invalid = invalid_shuffled.iloc[n_inv_train + n_inv_val:]

        # Combine and shuffle
        train_df = pd.concat([train_valid, train_invalid]).sample(frac=1, random_state=self.seed).reset_index(drop=True)
        val_df = pd.concat([val_valid, val_invalid]).sample(frac=1, random_state=self.seed + 1).reset_index(drop=True)
        test_df = pd.concat([test_valid, test_invalid]).sample(frac=1, random_state=self.seed + 2).reset_index(drop=True)

        # Update sentence_ids
        train_df['sentence_id'] = range(1, len(train_df) + 1)
        val_df['sentence_id'] = range(1, len(val_df) + 1)
        test_df['sentence_id'] = range(1, len(test_df) + 1)

        logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

        return train_df, val_df, test_df

    def save_dataset(
        self,
        df: pd.DataFrame,
        output_path: Path,
        format: str = 'csv',
    ):
        """
        Save dataset to file.

        Args:
            df: Dataset DataFrame
            output_path: Output file path
            format: File format ('csv' or 'jsonl')
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'csv':
            df.to_csv(output_path, index=False, encoding='utf-8')
        elif format == 'jsonl':
            df.to_json(output_path, orient='records', lines=True, force_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved {len(df)} sentences to {output_path}")


# MAIN FUNCTION

def main():
    """Generate and save dataset."""
    from pathlib import Path

    # Configuration
    NUM_SENTENCES = 10000
    OUTPUT_DIR = Path("data/datasets")
    TRAINING_DIR = Path("data/training")

    # Initialize generator
    generator = SentenceGenerator(num_sentences=NUM_SENTENCES)

    # Generate dataset
    df = generator.generate_dataset()

    # Save full dataset
    generator.save_dataset(df, OUTPUT_DIR / "full_dataset.csv")

    # Split and save
    train_df, val_df, test_df = generator.split_dataset(df)

    generator.save_dataset(train_df, TRAINING_DIR / "train.csv")
    generator.save_dataset(val_df, TRAINING_DIR / "val.csv")
    generator.save_dataset(test_df, TRAINING_DIR / "test.csv")

    # Generate statistics
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    print(f"Total sentences: {len(df)}")
    print(f"Train: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"Val: {len(val_df)} ({len(val_df) / len(df) * 100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")
    print(f"\nValid orders: {len(df[df['origin'] != 'INVALID'])}")
    print(f"Invalid orders: {len(df[df['origin'] == 'INVALID'])}")
    print("\nSample sentences:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
