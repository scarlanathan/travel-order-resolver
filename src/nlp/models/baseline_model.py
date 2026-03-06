"""
Baseline Model for Travel Order NLP

A simple rule-based approach using:
1. Preposition patterns (de, depuis, à, vers, pour)
2. City name gazetteer matching
3. Basic heuristics for origin/destination detection

This serves as a baseline to compare against more advanced models.
"""

import re
from typing import Tuple, Optional, List, Set
from dataclasses import dataclass


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
    Baseline NLP model using preposition patterns and gazetteer.

    Extraction Strategy:
    1. Try pattern matching with prepositions
    2. Fall back to gazetteer-based extraction
    3. Use word order heuristics as last resort
    """

    def __init__(self, cities: Optional[List[str]] = None, use_sncf: bool = True):
        """
        Initialize the baseline model.

        Args:
            cities: List of valid city names for gazetteer matching
            use_sncf: If True, use SNCF API data for cities
        """
        # Default French cities (SNCF stations)
        self.cities = cities or self._get_default_cities(use_sncf)

        # Create normalized lookup for fuzzy matching
        self.city_lookup = self._build_city_lookup()

        # Preposition patterns for extraction
        self.origin_patterns = self._build_origin_patterns()
        self.destination_patterns = self._build_destination_patterns()

        # Combined patterns
        self.combined_patterns = self._build_combined_patterns()

    def _get_default_cities(self, use_sncf: bool = True) -> List[str]:
        """Return default list of French cities."""
        # Try to get cities from SNCF API data
        if use_sncf:
            try:
                from src.api.sncf_client import SNCFClient
                client = SNCFClient()
                client.load_data(use_api=True)
                stations = client.get_all_stations()
                sncf_cities = list(set(s.city for s in stations))
                if sncf_cities:
                    return sncf_cities
            except ImportError:
                pass

        # Fallback to static list
        return [
            # Major French cities
            "Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux",
            "Lille", "Nice", "Nantes", "Strasbourg", "Montpellier",
            "Rennes", "Reims", "Tours", "Saint-Étienne", "Le Havre",
            "Grenoble", "Dijon", "Angers", "Nîmes", "Clermont-Ferrand",
            "Aix-en-Provence", "Brest", "Le Mans", "Amiens", "Limoges",
            "Perpignan", "Metz", "Besançon", "Orléans", "Rouen",
            "Mulhouse", "Caen", "Nancy", "Avignon", "Poitiers",
            "Toulon", "Cannes", "Antibes", "Saint-Nazaire", "Valence",
            "Chambéry", "Annecy", "Bayonne", "Pau", "Tarbes",
            "La Baule", "Vannes", "Quimper", "Lorient", "Saint-Brieuc",
            "Chartres", "Blois", "Bourges", "Vichy", "Aurillac",
            "Rodez", "Albi", "Carcassonne", "Béziers", "Sète",
            "Arles", "Hyères", "Fréjus", "Menton", "Monaco",
            "Colmar", "Épinal", "Troyes", "Châlons-en-Champagne", "Charleville-Mézières",
            "Dunkerque", "Calais", "Boulogne-sur-Mer", "Valenciennes", "Douai",
            # Tricky cities (homonyms, common words)
            "Albert", "Orange", "Cognac", "Port-Boulet", "Bourg-en-Bresse",
            "Château-Thierry", "La Rochelle", "Saint-Malo", "Mont-de-Marsan",
            "Lourdes", "Vic-sur-Cère", "Dax", "Agen", "Auch",
            "Gap", "Briançon", "Digne-les-Bains", "Sisteron", "Manosque",
            "Salon-de-Provence", "Martigues", "Istres", "La Ciotat", "Cassis",
            # International cities reachable by train
            "Florence", "Milan", "Genève", "Bruxelles", "Luxembourg",
        ]

    def _normalize(self, text: str) -> str:
        """Normalize text for matching."""
        # Lowercase
        text = text.lower()
        # Remove accents
        accents = {
            'à': 'a', 'â': 'a', 'ä': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'î': 'i', 'ï': 'i',
            'ô': 'o', 'ö': 'o',
            'ù': 'u', 'û': 'u', 'ü': 'u',
            'ÿ': 'y', 'ç': 'c',
        }
        for accent, replacement in accents.items():
            text = text.replace(accent, replacement)
        # Normalize hyphens and spaces
        text = re.sub(r'[-\s]+', ' ', text)
        return text.strip()

    def _build_city_lookup(self) -> dict:
        """Build normalized city name lookup table."""
        lookup = {}
        for city in self.cities:
            normalized = self._normalize(city)
            lookup[normalized] = city
            # Also add without hyphens
            no_hyphen = normalized.replace('-', ' ')
            if no_hyphen != normalized:
                lookup[no_hyphen] = city
        return lookup

    def _build_origin_patterns(self) -> List[re.Pattern]:
        """Build regex patterns for origin extraction."""
        # Patterns that typically precede origin
        patterns = [
            r'(?:partir|partant|pars|part)\s+(?:de|depuis)\s+([A-Za-zÀ-ÿ\-\s]+?)(?:\s+(?:pour|vers|à|a)|[,\.\?!]|$)',
            r'(?:de|depuis)\s+([A-Za-zÀ-ÿ\-\s]+?)\s+(?:pour|vers|à|a|jusqu)',
            r'(?:en\s+provenance\s+de|au\s+départ\s+de|en\s+partance\s+de)\s+([A-Za-zÀ-ÿ\-\s]+?)(?:\s|[,\.\?!]|$)',
            r'(?:départ|depart)\s+(?:de\s+)?([A-Za-zÀ-ÿ\-\s]+?)(?:\s|[,\.\?!]|$)',
            r'(?:à\s+partir\s+de)\s+([A-Za-zÀ-ÿ\-\s]+?)(?:\s|[,\.\?!]|$)',
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def _build_destination_patterns(self) -> List[re.Pattern]:
        """Build regex patterns for destination extraction."""
        patterns = [
            r'(?:pour|vers|à|a)\s+([A-Za-zÀ-ÿ\-\s]+?)(?:\s+(?:depuis|de|en\s+partant)|[,\.\?!]|$)',
            r'(?:aller\s+à|rendre\s+à|direction)\s+([A-Za-zÀ-ÿ\-\s]+?)(?:\s|[,\.\?!]|$)',
            r'(?:à\s+destination\s+de|arrivée\s+(?:à|vers))\s+([A-Za-zÀ-ÿ\-\s]+?)(?:\s|[,\.\?!]|$)',
            r'(?:destination)\s+([A-Za-zÀ-ÿ\-\s]+?)(?:\s|[,\.\?!]|$)',
            r'(?:jusqu\'?à|jusque)\s+([A-Za-zÀ-ÿ\-\s]+?)(?:\s|[,\.\?!]|$)',
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def _build_combined_patterns(self) -> List[Tuple[re.Pattern, str, str]]:
        """Build patterns that extract both origin and destination at once."""
        # (pattern, origin_group, destination_group)
        patterns = [
            # "de X à Y", "de X vers Y"
            (r'(?:de|depuis)\s+([A-Za-zÀ-ÿ\-\s]+?)\s+(?:à|a|vers|pour)\s+([A-Za-zÀ-ÿ\-\s]+?)(?:[,\.\?!]|$)', 1, 2),
            # "billet X Y" (direct format)
            (r'(?:billet|trajet)\s+([A-Za-zÀ-ÿ\-]+)\s+([A-Za-zÀ-ÿ\-]+)(?:[,\.\?!]|$)', 1, 2),
            # "entre X et Y"
            (r'entre\s+([A-Za-zÀ-ÿ\-\s]+?)\s+et\s+([A-Za-zÀ-ÿ\-\s]+?)(?:[,\.\?!]|$)', 1, 2),
        ]
        return [(re.compile(p, re.IGNORECASE), og, dg) for p, og, dg in patterns]

    def _find_city_in_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Find all city names in text.

        Returns:
            List of (city_name, start_pos, end_pos)
        """
        found = []
        normalized_text = self._normalize(text)

        for norm_city, original_city in self.city_lookup.items():
            # Use word boundaries for matching
            pattern = r'\b' + re.escape(norm_city) + r'\b'
            for match in re.finditer(pattern, normalized_text):
                found.append((original_city, match.start(), match.end()))

        # Sort by position
        found.sort(key=lambda x: x[1])
        return found

    def _match_city(self, candidate: str) -> Optional[str]:
        """Try to match a candidate string to a known city."""
        candidate = candidate.strip()
        normalized = self._normalize(candidate)

        # Direct lookup
        if normalized in self.city_lookup:
            return self.city_lookup[normalized]

        # Try each word separately (for cases like "mon ami Paris")
        words = normalized.split()
        for word in words:
            if word in self.city_lookup:
                return self.city_lookup[word]

        return None

    def _extract_with_combined_patterns(self, text: str) -> Optional[ExtractionResult]:
        """Try to extract origin and destination using combined patterns."""
        for pattern, origin_group, dest_group in self.combined_patterns:
            match = pattern.search(text)
            if match:
                origin_candidate = match.group(origin_group)
                dest_candidate = match.group(dest_group)

                origin = self._match_city(origin_candidate)
                destination = self._match_city(dest_candidate)

                if origin and destination and origin != destination:
                    return ExtractionResult(
                        origin=origin,
                        destination=destination,
                        is_valid=True,
                        confidence=0.9,
                        method="combined_pattern"
                    )
        return None

    def _extract_with_separate_patterns(self, text: str) -> Optional[ExtractionResult]:
        """Try to extract using separate origin and destination patterns."""
        origin = None
        destination = None

        # Find origin
        for pattern in self.origin_patterns:
            match = pattern.search(text)
            if match:
                candidate = match.group(1)
                matched_city = self._match_city(candidate)
                if matched_city:
                    origin = matched_city
                    break

        # Find destination
        for pattern in self.destination_patterns:
            match = pattern.search(text)
            if match:
                candidate = match.group(1)
                matched_city = self._match_city(candidate)
                if matched_city and matched_city != origin:
                    destination = matched_city
                    break

        if origin and destination:
            return ExtractionResult(
                origin=origin,
                destination=destination,
                is_valid=True,
                confidence=0.7,
                method="separate_patterns"
            )
        return None

    def _extract_with_gazetteer(self, text: str) -> Optional[ExtractionResult]:
        """Fall back to simple gazetteer matching based on word order."""
        found_cities = self._find_city_in_text(text)

        if len(found_cities) >= 2:
            # Use position heuristics: first city often origin, second often destination
            origin = found_cities[0][0]
            destination = found_cities[1][0]

            if origin != destination:
                return ExtractionResult(
                    origin=origin,
                    destination=destination,
                    is_valid=True,
                    confidence=0.5,
                    method="gazetteer_order"
                )
        return None

    def _is_travel_request(self, text: str) -> bool:
        """Check if text appears to be a travel request."""
        travel_indicators = [
            r'\b(train|billet|voyage|trajet|aller|partir|rendre|horaire)\b',
            r'\b(réserv|reserv)\b',
            r'\b(de|depuis|vers|pour|à)\s+[A-Z]',
        ]

        text_lower = text.lower()
        for pattern in travel_indicators:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        # Also check if we have at least one city
        found_cities = self._find_city_in_text(text)
        return len(found_cities) >= 1

    def predict(self, sentence: str) -> ExtractionResult:
        """
        Extract origin and destination from a sentence.

        Args:
            sentence: Input sentence in French

        Returns:
            ExtractionResult with origin, destination, and metadata
        """
        # Check if it looks like a travel request
        if not self._is_travel_request(sentence):
            return ExtractionResult(
                origin=None,
                destination=None,
                is_valid=False,
                confidence=0.8,
                method="not_travel_request"
            )

        # Try combined patterns first (most reliable)
        result = self._extract_with_combined_patterns(sentence)
        if result:
            return result

        # Try separate patterns
        result = self._extract_with_separate_patterns(sentence)
        if result:
            return result

        # Fall back to gazetteer
        result = self._extract_with_gazetteer(sentence)
        if result:
            return result

        # Failed to extract
        return ExtractionResult(
            origin=None,
            destination=None,
            is_valid=False,
            confidence=0.3,
            method="extraction_failed"
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
