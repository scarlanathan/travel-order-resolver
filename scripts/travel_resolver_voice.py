#!/usr/bin/env python
"""
Travel Order Resolver - Voice Pipeline

Extends the TravelResolver with speech input (Whisper STT) and output (pyttsx3 TTS).

Usage:
    python scripts/travel_resolver_voice.py                     # Voice interactive mode
    python scripts/travel_resolver_voice.py --text              # Text fallback mode
    python scripts/travel_resolver_voice.py --model-size small  # Use larger Whisper model
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.speech.transcriber import SpeechTranscriber
from src.speech.synthesizer import SpeechSynthesizer

try:
    from scripts.travel_resolver import TravelResolver
except ImportError as e:
    print(f"WARN: Could not import TravelResolver: {e}")
    print("NLP modules (src.nlp.models) not yet implemented.")
    print("Voice module (STT/TTS) can still be tested standalone.\n")
    TravelResolver = None


def voice_interactive_mode(resolver, transcriber, synthesizer, optimize="duration"):
    """Run interactive voice mode: listen -> transcribe -> resolve -> speak."""
    print("=" * 60)
    print("TRAVEL ORDER RESOLVER - Mode Vocal")
    print("=" * 60)
    print("Dites votre demande de voyage en francais.")
    print("Exemples :")
    print("  - Je voudrais aller de Paris a Lyon")
    print("  - Un billet Marseille Bordeaux s'il vous plait")
    print("\nCommandes clavier :")
    print("  'quit' ou 'exit' pour quitter")
    print("  'text' pour basculer en mode texte")
    print("=" * 60)

    synthesizer.speak("Bonjour, je suis pret a recevoir votre demande de voyage.")

    while True:
        print("\nParlez maintenant...")
        text = transcriber.listen()

        if text is None:
            print("Aucune parole detectee. Reessayez.")
            continue

        print(f"Transcription : \"{text}\"")

        lower = text.lower().strip()
        if lower in ("quit", "exit", "quitter", "sortir", "au revoir"):
            synthesizer.speak("Au revoir !")
            print("Au revoir !")
            break

        if lower in ("text", "texte", "mode texte"):
            print("Basculement en mode texte.")
            text_fallback_mode(resolver, synthesizer, optimize)
            break

        route = resolver.resolve(text, optimize=optimize)
        if route:
            print(f"\n{route}")
            synthesizer.speak_route(route)
        else:
            msg = "Desole, je n'ai pas pu trouver d'itineraire pour cette demande."
            print(msg)
            synthesizer.speak(msg)


def text_fallback_mode(resolver, synthesizer, optimize="duration"):
    """Text input mode with TTS output."""
    print("=" * 60)
    print("TRAVEL ORDER RESOLVER - Mode Texte (avec synthese vocale)")
    print("=" * 60)
    print("Entrez votre demande de voyage.")
    print("'quit' ou 'exit' pour quitter")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break

        route = resolver.resolve(user_input, optimize=optimize)
        if route:
            print(f"\n{route}")
            synthesizer.speak_route(route)
        else:
            msg = "Desole, je n'ai pas pu trouver d'itineraire pour cette demande."
            print(msg)
            synthesizer.speak(msg)


def main():
    parser = argparse.ArgumentParser(
        description="Travel Order Resolver - Voice Pipeline (Whisper + pyttsx3)"
    )
    parser.add_argument(
        "--text", "-t",
        action="store_true",
        help="Use text input mode instead of voice"
    )
    parser.add_argument(
        "--nlp-model", "-m",
        default="spacy",
        choices=["spacy", "baseline"],
        help="NLP model to use (default: spacy)"
    )
    parser.add_argument(
        "--optimize", "-o",
        default="duration",
        choices=["duration", "distance"],
        help="Optimize for duration or distance (default: duration)"
    )
    parser.add_argument(
        "--model-size",
        default="base",
        choices=["tiny", "base", "small", "medium"],
        help="Whisper model size (default: base)"
    )

    args = parser.parse_args()

    # Initialize TTS
    synthesizer = SpeechSynthesizer()

    # Initialize resolver (may not be available if NLP modules missing)
    resolver = None
    if TravelResolver is not None:
        try:
            resolver = TravelResolver(nlp_model=args.nlp_model)
        except Exception as e:
            print(f"WARN: Could not init TravelResolver: {e}")

    if resolver is None:
        print("Mode standalone : test STT/TTS uniquement (pas de NLP).")
        standalone_voice_test(synthesizer, args)
        return

    if args.text:
        text_fallback_mode(resolver, synthesizer, optimize=args.optimize)
    else:
        # Initialize STT
        try:
            transcriber = SpeechTranscriber(
                model_size=args.model_size, language="fr"
            )
            transcriber.calibrate()
        except Exception as e:
            print(f"Erreur initialisation micro : {e}")
            print("Basculement en mode texte.")
            text_fallback_mode(resolver, synthesizer, optimize=args.optimize)
            return

        voice_interactive_mode(
            resolver, transcriber, synthesizer, optimize=args.optimize
        )


def standalone_voice_test(synthesizer, args):
    """Test STT + TTS without NLP pipeline."""
    print("=" * 60)
    print("VOICE MODULE - Test Standalone")
    print("=" * 60)

    synthesizer.speak("Bonjour, le module vocal fonctionne.")

    try:
        transcriber = SpeechTranscriber(
            model_size=args.model_size, language="fr"
        )
        transcriber.calibrate()
    except Exception as e:
        print(f"Erreur micro : {e}")
        print("Test TTS seul : OK")
        return

    print("\nParlez pour tester la transcription ('quit' pour quitter) :")
    while True:
        print("\nParlez maintenant...")
        text = transcriber.listen()
        if text is None:
            print("Aucune parole detectee.")
            continue
        print(f"Transcription : \"{text}\"")
        synthesizer.speak(f"Vous avez dit : {text}")
        if text.lower().strip() in ("quit", "exit", "quitter", "au revoir"):
            break


if __name__ == "__main__":
    main()
