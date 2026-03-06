#!/usr/bin/env python
"""
Script to generate training dataset for NER model.

Usage:
    python scripts/generate_dataset.py --num-sentences 10000
    python scripts/generate_dataset.py --num-sentences 1000 --output data/datasets/small_dataset.csv
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.sentence_generator import SentenceGenerator
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Generate training dataset")
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=10000,
        help="Number of sentences to generate (default: 10000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/datasets/full_dataset.csv",
        help="Output path for full dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )

    args = parser.parse_args()

    # Setup logging
    logger.info(f"Generating {args.num_sentences} sentences...")

    # Create output directories
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    training_dir = Path("data/training")
    training_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = SentenceGenerator(
        num_sentences=args.num_sentences,
        seed=args.seed,
    )

    # Generate dataset
    df = generator.generate_dataset()

    # Save full dataset
    generator.save_dataset(df, output_path)

    # Split and save
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    train_df, val_df, test_df = generator.split_dataset(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
    )

    generator.save_dataset(train_df, training_dir / "train.csv")
    generator.save_dataset(val_df, training_dir / "val.csv")
    generator.save_dataset(test_df, training_dir / "test.csv")

    # Print statistics
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal sentences: {len(df)}")
    print(f"  - Valid orders:   {len(df[df['origin'] != 'INVALID']):>6}")
    print(f"  - Invalid orders: {len(df[df['origin'] == 'INVALID']):>6}")
    print(f"\nSplit:")
    print(f"  - Train: {len(train_df):>6} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  - Val:   {len(val_df):>6} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  - Test:  {len(test_df):>6} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"\nFiles saved:")
    print(f"  - {output_path}")
    print(f"  - {training_dir / 'train.csv'}")
    print(f"  - {training_dir / 'val.csv'}")
    print(f"  - {training_dir / 'test.csv'}")
    print("\nSample sentences:")
    print("-" * 60)
    for _, row in df.head(5).iterrows():
        print(f"  [{row['sentence_id']}] {row['sentence']}")
        print(f"       -> Origin: {row['origin']}, Dest: {row['destination']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
