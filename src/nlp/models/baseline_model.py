"""
Baseline Model for Travel Order NLP

A simple rule-based approach using:
1. City gazetteer matching (training data cities + SNCF stations),
   tolerant to case, accents, hyphens and small misspellings
2. Cue words before each city (de, depuis, à, vers, pour, à destination de...)
3. Word order as last resort (origin before destination)

This serves as a baseline to compare against more advanced models.
"""

from typing import Optional, List
from dataclasses import dataclass

from src.nlp.gazetteer import CityGazetteer


@dataclass
class ExtractionResult:
    """Result of entity extraction."""
    origin: Optional[str]
    destination: Optional[str]
    is_valid: bool
    confidence: float
    method: str  # Which extraction method was used


class BaselineModel:
    """
    Baseline NLP model using a city gazetteer and cue words.

    Extraction Strategy:
    1. Find every known city in the sentence
    2. Fewer than two distinct cities -> invalid request
    3. Assign origin/destination with cue words, then word order
    """

    def __init__(
        self,
        cities: Optional[List[str]] = None,
        use_sncf: bool = True,
        gazetteer: Optional[CityGazetteer] = None,
    ):
        """
        Initialize the baseline model.

        Args:
            cities: List of valid city names for gazetteer matching
                    (default: cities of the training set + SNCF stations)
            use_sncf: If True, add SNCF stations to the gazetteer
            gazetteer: Prebuilt gazetteer (overrides cities/use_sncf)
        """
        self.gazetteer = gazetteer or CityGazetteer(cities=cities, use_sncf=use_sncf)
        self.cities = self.gazetteer.cities

    def predict(self, sentence: str) -> ExtractionResult:
        """
        Extract origin and destination from a sentence.

        Args:
            sentence: Input sentence in French

        Returns:
            ExtractionResult with origin, destination, and metadata
        """
        mentions = self.gazetteer.find_mentions(sentence)

        if len({m.city for m in mentions}) < 2:
            return ExtractionResult(
                origin=None,
                destination=None,
                is_valid=False,
                confidence=0.8 if not mentions else 0.6,
                method="not_travel_request",
            )

        origin, destination = self.gazetteer.assign_roles(sentence, mentions)
        if origin is None or destination is None or origin.city == destination.city:
            return ExtractionResult(
                origin=None,
                destination=None,
                is_valid=False,
                confidence=0.3,
                method="extraction_failed",
            )

        cued = sum(
            self.gazetteer.cue_role(sentence, m.start) is not None
            for m in (origin, destination)
        )
        return ExtractionResult(
            origin=origin.city,
            destination=destination.city,
            is_valid=True,
            confidence=0.5 + 0.2 * cued,
            method="gazetteer_cues" if cued else "gazetteer_order",
        )

    def predict_batch(self, sentences: List[str]) -> List[ExtractionResult]:
        """
        Process multiple sentences.

        Args:
            sentences: List of input sentences

        Returns:
            List of ExtractionResults
        """
        return [self.predict(s) for s in sentences]


# For command-line testing
if __name__ == "__main__":
    model = BaselineModel()

    test_sentences = [
        "Je voudrais un billet Paris Lyon.",
        "Comment me rendre à Marseille depuis Toulouse ?",
        "Je veux aller à Tours voir mon ami Albert en partant de Bordeaux.",
        "Avec mes amis florence et paris, je voudrais aller de paris a florence.",
        "je souhaite me rendre à Paris depuis Toulouse",
        "Bonjour, comment allez-vous ?",
        "Quel temps fait-il à Paris ?",
        "À quelle heure y a-t-il des trains vers Paris en partance de Toulouse ?",
        "faut que j'aille de nantes à cognac",
    ]

    print("=" * 70)
    print("BASELINE MODEL TEST")
    print("=" * 70)

    for sentence in test_sentences:
        result = model.predict(sentence)
        print(f"\nInput: {sentence}")
        if result.is_valid:
            print(f"  -> Origin: {result.origin}")
            print(f"  -> Destination: {result.destination}")
            print(f"  -> Confidence: {result.confidence:.2f}")
            print(f"  -> Method: {result.method}")
        else:
            print(f"  -> INVALID (method: {result.method})")
