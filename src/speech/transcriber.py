"""Speech-to-Text module using OpenAI Whisper for French transcription."""

import os
import tempfile

import speech_recognition as sr
import whisper


class SpeechTranscriber:
    """Transcribes French speech to text using Whisper."""

    def __init__(self, model_size="base", language="fr"):
        """
        Initialize the transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            language: Language code for transcription
        """
        self.language = language
        self.recognizer = sr.Recognizer()
        print(f"Loading Whisper model '{model_size}'...")
        self.model = whisper.load_model(model_size)
        print("Whisper model loaded.")

    def calibrate(self, duration=2):
        """
        Calibrate microphone for ambient noise.

        Args:
            duration: Seconds to listen for ambient noise calibration
        """
        with sr.Microphone() as source:
            print("Calibration du micro...")
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)
            self.recognizer.dynamic_energy_threshold = True
            print("Calibration terminee.")

    def listen(self, timeout=10, phrase_time_limit=15):
        """
        Listen to microphone and transcribe speech.

        Args:
            timeout: Max seconds to wait for speech to start
            phrase_time_limit: Max seconds for a single phrase

        Returns:
            Transcribed text or None if timeout/no speech detected
        """
        with sr.Microphone() as source:
            try:
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )
            except sr.WaitTimeoutError:
                return None

        return self._transcribe_audio(audio)

    def transcribe_file(self, audio_path):
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file (WAV, MP3, etc.)

        Returns:
            Transcribed text
        """
        result = self.model.transcribe(str(audio_path), language=self.language)
        return result["text"].strip()

    def _transcribe_audio(self, audio):
        """
        Transcribe a SpeechRecognition AudioData object via Whisper.

        Args:
            audio: sr.AudioData captured from microphone

        Returns:
            Transcribed text
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio.get_wav_data())
            tmp_path = tmp.name
        try:
            result = self.model.transcribe(tmp_path, language=self.language)
            return result["text"].strip()
        finally:
            os.unlink(tmp_path)
