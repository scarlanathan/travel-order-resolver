"""Data generation module for training dataset."""

from src.data_generation.templates import (
    get_all_templates,
    get_invalid_templates,
    get_cities,
    get_names,
    ALL_CITIES,
    MAJOR_CITIES,
    TRICKY_CITIES,
)
from src.data_generation.sentence_generator import SentenceGenerator

__all__ = [
    "SentenceGenerator",
    "get_all_templates",
    "get_invalid_templates",
    "get_cities",
    "get_names",
    "ALL_CITIES",
    "MAJOR_CITIES",
    "TRICKY_CITIES",
]
