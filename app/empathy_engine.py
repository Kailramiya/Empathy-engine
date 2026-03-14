"""
Core Empathy Engine — combines emotion detection with voice synthesis.

This module provides the central orchestration class that:
  1. Accepts text input (English, Hindi, or Hinglish)
  2. Translates non-English text for emotion analysis
  3. Detects emotion (with confidence/intensity)
  4. Maps emotion → voice parameters (with intensity scaling)
  5. Generates emotionally modulated speech audio in the chosen language
  6. Returns a structured result with full analysis metadata
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .emotion_detector import EmotionDetector, EmotionResult
from .voice_synthesizer import VoiceSynthesizer, VoiceParameters


@dataclass
class EngineResult:
    """Complete result from the Empathy Engine pipeline."""
    text: str
    language: str
    voice_used: str
    emotion: EmotionResult
    voice_params: VoiceParameters
    audio_filename: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-friendly dictionary for the API."""
        result = {
            "text": self.text,
            "language": self.language,
            "voice_used": self.voice_used,
            "emotion": {
                "detected": self.emotion.emotion,
                "confidence": self.emotion.confidence,
                "all_scores": self.emotion.all_scores,
                "method": self.emotion.method,
            },
            "voice_parameters": {
                "rate": self.voice_params.rate,
                "pitch": self.voice_params.pitch,
                "volume": self.voice_params.volume,
                "description": self.voice_params.description,
                "ssml": self.voice_params.ssml.replace("{text}", self.text),
            },
            "audio_url": f"/audio/{self.audio_filename}",
        }

        # Include translated text if the input was non-English
        if self.emotion.translated_text:
            result["emotion"]["translated_text"] = self.emotion.translated_text

        return result


class EmpathyEngine:
    """
    High-level orchestrator for the Empathy Engine.

    Usage:
        engine = EmpathyEngine()

        # English
        result = engine.process("I'm thrilled about this!")

        # Hindi
        result = engine.process("मुझे बहुत खुशी हो रही है!", language="hi")

        # Hinglish
        result = engine.process("Yaar mujhe bahut gussa aa raha hai!", language="hinglish")
    """

    def __init__(self, use_transformer: bool = True, output_dir: str = "output"):
        self.detector = EmotionDetector(use_transformer=use_transformer)
        self.synthesizer = VoiceSynthesizer(output_dir=output_dir)

    def get_available_voices(self, language: str = "en") -> List[dict]:
        """Return available voices for a given language."""
        return self.synthesizer.get_available_voices(language)

    def process(
        self, text: str, language: str = "auto", voice: str = "",
    ) -> EngineResult:
        """
        Synchronous pipeline: detect emotion → synthesize speech.
        Uses the emotion confidence as the intensity scaling factor.
        """
        # Detect language if auto
        effective_lang = self.detector._detect_language(text, language)

        emotion = self.detector.detect(text, language=effective_lang)

        resolved_voice = voice or self.synthesizer.get_default_voice(effective_lang)

        filename, params = self.synthesizer.synthesize(
            text, emotion.emotion, emotion.confidence,
            language=effective_lang, voice=voice,
        )
        return EngineResult(
            text=text,
            language=effective_lang,
            voice_used=resolved_voice,
            emotion=emotion,
            voice_params=params,
            audio_filename=filename,
        )

    async def process_async(
        self, text: str, language: str = "auto", voice: str = "",
    ) -> EngineResult:
        """Async pipeline for use inside FastAPI / async contexts."""
        effective_lang = self.detector._detect_language(text, language)

        emotion = self.detector.detect(text, language=effective_lang)

        resolved_voice = voice or self.synthesizer.get_default_voice(effective_lang)

        filename, params = await self.synthesizer.synthesize_async(
            text, emotion.emotion, emotion.confidence,
            language=effective_lang, voice=voice,
        )
        return EngineResult(
            text=text,
            language=effective_lang,
            voice_used=resolved_voice,
            emotion=emotion,
            voice_params=params,
            audio_filename=filename,
        )
