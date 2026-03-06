"""Text-to-Speech module using pyttsx3 for French speech synthesis."""

import pyttsx3


class SpeechSynthesizer:
    """Synthesizes French speech from text using pyttsx3."""

    def __init__(self, rate=160, volume=1.0):
        """
        Initialize the synthesizer.

        Args:
            rate: Speech rate in words per minute
            volume: Volume level (0.0 to 1.0)
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        self._set_french_voice()

    def _set_french_voice(self):
        """Try to set a French voice if available."""
        for voice in self.engine.getProperty('voices'):
            if 'french' in voice.name.lower() or 'fr' in str(voice.languages).lower():
                self.engine.setProperty('voice', voice.id)
                return

    def speak(self, text):
        """
        Speak the given text aloud.

        Args:
            text: Text to synthesize
        """
        self.engine.say(text)
        self.engine.runAndWait()

    def speak_route(self, route):
        """
        Format and speak a Route object.

        Args:
            route: Route dataclass from pathfinder
        """
        hours = route.total_duration_minutes // 60
        minutes = route.total_duration_minutes % 60
        text = (
            f"Itineraire de {route.origin.city} vers {route.destination.city}. "
            f"Duree {hours} heures {minutes} minutes. "
            f"Distance {route.total_distance_km:.0f} kilometres. "
            f"{route.num_changes} correspondance{'s' if route.num_changes > 1 else ''}."
        )
        self.speak(text)
