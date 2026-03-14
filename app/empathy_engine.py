"""
Core Empathy Engine — combines emotion detection with voice synthesis.

This module provides the central orchestration class that:
  1. Accepts text input
  2. Detects emotion (with confidence/intensity)
  3. Maps emotion → voice parameters (with intensity scaling)
  4. Generates emotionally modulated speech audio
  5. Returns a structured result with full analysis metadata
"""

from dataclasses import dataclass
from typing import Any, Dict

from .emotion_detector import EmotionDetector, EmotionResult
from .voice_synthesizer import VoiceSynthesizer, VoiceParameters


@dataclass
class EngineResult:
    """Complete result from the Empathy Engine pipeline."""
    text: str
    emotion: EmotionResult
    voice_params: VoiceParameters
    audio_filename: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-friendly dictionary for the API."""
        return {
            "text": self.text,
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


class EmpathyEngine:
    """
    High-level orchestrator for the Empathy Engine.

    Usage:
        engine = EmpathyEngine()
        result = engine.process("I'm thrilled about this opportunity!")
        # result.emotion.emotion  → "joy"
        # result.audio_filename   → "speech_abc123.mp3"
    """

    def __init__(self, use_transformer: bool = True, output_dir: str = "output"):
        self.detector = EmotionDetector(use_transformer=use_transformer)
        self.synthesizer = VoiceSynthesizer(output_dir=output_dir)

    def process(self, text: str) -> EngineResult:
        """
        Synchronous pipeline: detect emotion → synthesize speech.
        Uses the emotion confidence as the intensity scaling factor.
        """
        emotion = self.detector.detect(text)
        filename, params = self.synthesizer.synthesize(
            text, emotion.emotion, emotion.confidence
        )
        return EngineResult(
            text=text,
            emotion=emotion,
            voice_params=params,
            audio_filename=filename,
        )

    async def process_async(self, text: str) -> EngineResult:
        """
        Async pipeline for use inside FastAPI / async contexts.
        """
        emotion = self.detector.detect(text)
        filename, params = await self.synthesizer.synthesize_async(
            text, emotion.emotion, emotion.confidence
        )
        return EngineResult(
            text=text,
            emotion=emotion,
            voice_params=params,
            audio_filename=filename,
        )
