#!/usr/bin/env python
"""
Travel Order Resolver - Complete Pipeline

Combines NLP extraction with pathfinding to resolve travel requests.
Input: Natural language sentence (e.g., "Je veux aller de Paris à Lyon")
Output: Optimal train route with duration and changes

Usage:
    python scripts/travel_resolver.py "Je voudrais un billet de Marseille à Bordeaux"
    python scripts/travel_resolver.py --interactive
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.models.spacy_ner import SpacyNERModel
from src.nlp.models.baseline_model import BaselineModel
from src.nlp.models.camembert_ner import CamembertNERModel
from src.pathfinder.dijkstra import TrainGraph, Route


class TravelResolver:
    """Complete pipeline: NLP extraction + Pathfinding."""

    def __init__(self, nlp_model: str = "spacy"):
        """
        Initialize the travel resolver.

        Args:
            nlp_model: "spacy", "baseline", "camembert", or path to a model
        """
        self.graph = TrainGraph()

        # Load NLP model
        if nlp_model == "baseline":
            self.nlp = BaselineModel()
            self.model_name = "Baseline"
        elif nlp_model == "camembert":
            model_path = "models/camembert_ner/camembert_ner_1000"
            if Path(model_path).exists():
                self.nlp = CamembertNERModel(model_path)
                self.model_name = "CamemBERT NER (1k)"
            else:
                # Try full model
                model_path = "models/camembert_ner/camembert_ner_full"
                if Path(model_path).exists():
                    self.nlp = CamembertNERModel(model_path)
                    self.model_name = "CamemBERT NER (full)"
                else:
                    print(f"CamemBERT model not found, using baseline")
                    self.nlp = BaselineModel()
                    self.model_name = "Baseline"
        elif nlp_model == "spacy":
            model_path = "models/spacy_ner/ner_blank_1000"
            if Path(model_path).exists():
                self.nlp = SpacyNERModel(model_path)
                self.model_name = "spaCy NER (1k)"
            else:
                print(f"spaCy model not found at {model_path}, using baseline")
                self.nlp = BaselineModel()
                self.model_name = "Baseline"
        else:
            # Assume it's a path to a model
            if Path(nlp_model).exists():
                # Try CamemBERT first (check for config.json with model_type)
                config_path = Path(nlp_model) / "config.json"
                if config_path.exists():
                    import json
                    with open(config_path) as f:
                        config = json.load(f)
                    if "camembert" in config.get("model_type", "").lower():
                        self.nlp = CamembertNERModel(nlp_model)
                        self.model_name = f"CamemBERT ({nlp_model})"
                    else:
                        self.nlp = CamembertNERModel(nlp_model)
                        self.model_name = f"Transformer ({nlp_model})"
                else:
                    self.nlp = SpacyNERModel(nlp_model)
                    self.model_name = f"spaCy ({nlp_model})"
            else:
                self.nlp = SpacyNERModel(nlp_model)
                self.model_name = f"spaCy ({nlp_model})"

        print(f"NLP Model: {self.model_name}")
        stats = self.graph.get_network_stats()
        print(f"Network: {stats['num_stations']} stations, {stats['num_connections']} connections\n")

    def _resolve_city(self, city_name: str) -> Optional[str]:
        """
        Resolve a city name to a known station city, using fuzzy matching
        and nearest station fallback.

        Returns the resolved city name or None.
        """
        # Try exact/fuzzy match first
        station = self.graph.client.find_station_by_city(city_name)
        if station:
            return station.city

        # Try nearest station for cities without a station
        result = self.graph.client.find_nearest_station(city_name)
        if result:
            nearest_station, distance = result
            print(f"Info: '{city_name}' has no station. "
                  f"Using nearest: {nearest_station.city} ({distance} km away)")
            return nearest_station.city

        return None

    def resolve(self, sentence: str) -> Optional[Route]:
        """
        Resolve a travel request sentence (optimized by duration).

        Args:
            sentence: Natural language travel request

        Returns:
            Route object or None if resolution failed
        """
        # Step 1: NLP extraction
        result = self.nlp.predict(sentence)

        print(f"Input: \"{sentence}\"")
        print(f"Extracted: origin={result.origin}, destination={result.destination}")

        if not result.origin or not result.destination:
            print("ERROR: Could not extract origin and/or destination")
            return None

        if result.origin.upper() == "INVALID" or result.destination.upper() == "INVALID":
            print("ERROR: Invalid travel request (missing origin or destination)")
            return None

        # Step 2: Resolve cities (fuzzy + nearest station fallback)
        resolved_origin = self._resolve_city(result.origin)
        resolved_destination = self._resolve_city(result.destination)

        if not resolved_origin:
            print(f"ERROR: City '{result.origin}' not found in station database")
            return None
        if not resolved_destination:
            print(f"ERROR: City '{result.destination}' not found in station database")
            return None

        # Step 3: Find route (always optimized by duration)
        route = self.graph.find_shortest_path(
            resolved_origin,
            resolved_destination,
        )

        if not route:
            print(f"ERROR: No route found between {resolved_origin} and {resolved_destination}")
            return None

        return route

    def resolve_with_alternatives(
        self,
        sentence: str,
        max_routes: int = 3,
    ):
        """
        Resolve a travel request and show alternative routes (optimized by duration).

        Args:
            sentence: Natural language travel request
            max_routes: Maximum number of alternative routes
        """
        result = self.nlp.predict(sentence)

        print(f"Input: \"{sentence}\"")
        print(f"Extracted: origin={result.origin}, destination={result.destination}")

        if not result.origin or not result.destination:
            print("ERROR: Could not extract origin and/or destination")
            return

        routes = self.graph.find_all_paths(
            result.origin,
            result.destination,
            max_paths=max_routes,
        )

        if not routes:
            print(f"No routes found between {result.origin} and {result.destination}")
            return

        print(f"\nFound {len(routes)} route(s):\n")
        for i, route in enumerate(routes, 1):
            print(f"--- Option {i} ---")
            print(route)
            print()


def interactive_mode(resolver: TravelResolver):
    """Run interactive mode."""
    print("=" * 60)
    print("TRAVEL ORDER RESOLVER - Interactive Mode")
    print("=" * 60)
    print("Enter your travel request in French.")
    print("Examples:")
    print("  - Je voudrais aller de Paris à Lyon")
    print("  - Un billet Marseille Bordeaux s'il vous plaît")
    print("  - Direction Nice au départ de Toulouse")
    print("\nCommands:")
    print("  'quit' or 'exit' to quit")
    print("  'alt <sentence>' for alternative routes")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower().startswith("alt "):
            sentence = user_input[4:].strip()
            resolver.resolve_with_alternatives(sentence)
        else:
            route = resolver.resolve(user_input)
            if route:
                print(f"\n{route}")


def main():
    parser = argparse.ArgumentParser(
        description="Travel Order Resolver - NLP + Pathfinding"
    )
    parser.add_argument(
        "sentence",
        nargs="?",
        help="Travel request sentence to resolve"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--model", "-m",
        default="spacy",
        choices=["spacy", "baseline", "camembert"],
        help="NLP model to use (default: spacy)"
    )
    parser.add_argument(
        "--alternatives", "-a",
        action="store_true",
        help="Show alternative routes"
    )

    args = parser.parse_args()

    resolver = TravelResolver(nlp_model=args.model)

    if args.interactive:
        interactive_mode(resolver)
    elif args.sentence:
        if args.alternatives:
            resolver.resolve_with_alternatives(args.sentence)
        else:
            route = resolver.resolve(args.sentence)
            if route:
                print(f"\n{route}")
    else:
        # Demo mode
        print("=" * 60)
        print("TRAVEL ORDER RESOLVER - Demo")
        print("=" * 60)

        test_sentences = [
            "Je voudrais aller de Paris à Marseille",
            "Un billet Lyon Bordeaux s'il vous plaît",
            "Direction Nice au départ de Lille",
            "Toulouse Strasbourg pour demain matin",
            "Je pars de Nantes pour aller à Rennes",
        ]

        for sentence in test_sentences:
            print(f"\n{'='*60}")
            route = resolver.resolve(sentence)
            if route:
                print(f"\n{route}")


if __name__ == "__main__":
    main()
