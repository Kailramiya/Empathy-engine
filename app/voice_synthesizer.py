"""
Voice Synthesizer Module for The Empathy Engine.

Maps detected emotions to vocal parameters (rate, pitch, volume) and
generates expressive speech audio with intensity scaling.

Supports multiple languages and neural voices:
  English:  en-US-AriaNeural, en-US-JennyNeural, en-US-GuyNeural, en-GB-SoniaNeural
  Hindi:    hi-IN-SwaraNeural, hi-IN-MadhurNeural
  Hinglish: Uses Hindi neural voices (handles mixed script naturally)

TTS Engines (auto-detected, in priority order):
  1. edge-tts  — Microsoft Neural voices, SSML prosody control, free (needs internet)
  2. espeak-ng  — Offline, full pitch/rate/amplitude control via subprocess
  3. pyttsx3    — Offline fallback, rate + volume control

Emotion-to-Voice Mapping Design Rationale:
  The parameter choices are grounded in prosody research from affective computing.
  See: Scherer, K.R. (2003) "Vocal communication of emotion"
"""

import asyncio
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class VoiceParameters:
    """Vocal parameters applied to the TTS output."""
    rate: str           # e.g. "+15%"
    pitch: str          # e.g. "+8Hz"
    volume: str         # e.g. "+10%"
    description: str    # Human-readable explanation
    ssml: str           # Generated SSML markup


@dataclass
class VoiceOption:
    """Represents a selectable TTS voice."""
    id: str             # e.g. "en-US-AriaNeural"
    name: str           # e.g. "Aria (US Female)"
    language: str       # e.g. "en", "hi", "hinglish"
    gender: str         # "female" or "male"


# ── Available Voices ─────────────────────────────────────────────────────

VOICE_OPTIONS: Dict[str, List[VoiceOption]] = {
    "en": [
        VoiceOption("en-US-AriaNeural",   "Aria (US Female, Expressive)",   "en", "female"),
        VoiceOption("en-US-JennyNeural",  "Jenny (US Female, Warm)",        "en", "female"),
        VoiceOption("en-US-GuyNeural",    "Guy (US Male, Casual)",          "en", "male"),
        VoiceOption("en-US-DavisNeural",  "Davis (US Male, Professional)",  "en", "male"),
        VoiceOption("en-GB-SoniaNeural",  "Sonia (UK Female, British)",     "en", "female"),
        VoiceOption("en-IN-NeerjaNeural", "Neerja (Indian English Female)", "en", "female"),
    ],
    "hi": [
        VoiceOption("hi-IN-SwaraNeural",  "Swara (Hindi Female)",           "hi", "female"),
        VoiceOption("hi-IN-MadhurNeural", "Madhur (Hindi Male)",            "hi", "male"),
    ],
    "hinglish": [
        VoiceOption("hi-IN-SwaraNeural",  "Swara (Hinglish Female)",        "hinglish", "female"),
        VoiceOption("hi-IN-MadhurNeural", "Madhur (Hinglish Male)",         "hinglish", "male"),
        VoiceOption("en-IN-NeerjaNeural", "Neerja (Indian English Female)", "hinglish", "female"),
    ],
}

# Default voice per language
DEFAULT_VOICES = {
    "en": "en-US-AriaNeural",
    "hi": "hi-IN-SwaraNeural",
    "hinglish": "hi-IN-SwaraNeural",
}


# ── Emotion → Voice Profile Mapping ──────────────────────────────────────

EMOTION_PROFILES = {
    "joy": {
        "rate": 30,
        "pitch": 25,
        "volume": 20,
        "desc": "Energetic, upbeat delivery with raised pitch and faster pace",
    },
    "anger": {
        "rate": 15,
        "pitch": -15,
        "volume": 40,
        "desc": "Forceful, intense delivery with lower pitch and amplified volume",
    },
    "sadness": {
        "rate": -40,
        "pitch": -30,
        "volume": -25,
        "desc": "Slow, somber delivery with lowered pitch and subdued volume",
    },
    "fear": {
        "rate": 35,
        "pitch": 28,
        "volume": -15,
        "desc": "Rapid, tense delivery with raised pitch — conveying anxiety",
    },
    "surprise": {
        "rate": 20,
        "pitch": 35,
        "volume": 25,
        "desc": "Exclamatory delivery with notably raised pitch and volume",
    },
    "disgust": {
        "rate": -20,
        "pitch": -18,
        "volume": -10,
        "desc": "Deliberate, disdainful delivery with lowered pitch",
    },
    "neutral": {
        "rate": 0,
        "pitch": 0,
        "volume": 0,
        "desc": "Natural, balanced delivery — no emotional modulation",
    },
}


class VoiceSynthesizer:
    """
    Generates emotionally modulated speech audio.

    Automatically selects the best available TTS engine and applies
    emotion-driven prosody adjustments with intensity scaling.
    Supports English, Hindi, and Hinglish with multiple neural voices.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self._engine = self._detect_engine()
        os.makedirs(output_dir, exist_ok=True)
        print(f"[Empathy Engine] ✓ TTS engine: {self._engine}")

    @property
    def engine_name(self) -> str:
        return self._engine

    @staticmethod
    def get_available_voices(language: str = "en") -> List[dict]:
        """Return available voices for a given language."""
        voices = VOICE_OPTIONS.get(language, VOICE_OPTIONS["en"])
        return [{"id": v.id, "name": v.name, "gender": v.gender} for v in voices]

    @staticmethod
    def get_default_voice(language: str = "en") -> str:
        """Return the default voice ID for a language."""
        return DEFAULT_VOICES.get(language, DEFAULT_VOICES["en"])

    # ── Engine detection ─────────────────────────────────────────────────

    def _detect_engine(self) -> str:
        """Auto-detect the best available TTS engine."""
        try:
            import edge_tts  # noqa: F401
            return "edge-tts"
        except ImportError:
            pass

        if shutil.which("espeak-ng"):
            return "espeak-ng"

        try:
            import pyttsx3  # noqa: F401
            return "pyttsx3"
        except ImportError:
            pass

        raise RuntimeError(
            "No TTS engine found. Install one of: edge-tts, espeak-ng, pyttsx3"
        )

    # ── Parameter computation ────────────────────────────────────────────

    def get_voice_parameters(self, emotion: str, intensity: float = 1.0, language: str = "en") -> VoiceParameters:
        """
        Compute voice parameters for a given emotion and intensity.

        Intensity scaling: the raw profile values are multiplied by the
        confidence score so that subtle text gets subtle modulation while
        highly emotional text gets dramatic shifts.
        """
        profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])

        scaled_rate = int(profile["rate"] * intensity)
        scaled_pitch = int(profile["pitch"] * intensity)
        scaled_volume = int(profile["volume"] * intensity)

        rate_str = f"{'+' if scaled_rate >= 0 else ''}{scaled_rate}%"
        pitch_str = f"{'+' if scaled_pitch >= 0 else ''}{scaled_pitch}Hz"
        volume_str = f"{'+' if scaled_volume >= 0 else ''}{scaled_volume}%"

        # Determine xml:lang for SSML
        lang_tag = {"en": "en-US", "hi": "hi-IN", "hinglish": "hi-IN"}.get(language, "en-US")

        ssml = (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{lang_tag}">\n'
            f'  <prosody rate="{rate_str}" pitch="{pitch_str}" volume="{volume_str}">\n'
            "    {text}\n"
            "  </prosody>\n"
            "</speak>"
        )

        return VoiceParameters(
            rate=rate_str,
            pitch=pitch_str,
            volume=volume_str,
            description=profile["desc"],
            ssml=ssml,
        )

    # ── Resolve voice ────────────────────────────────────────────────────

    def _resolve_voice(self, voice: str = "", language: str = "en") -> str:
        """Pick the right voice: user choice > language default > fallback."""
        if voice:
            return voice
        return self.get_default_voice(language)

    # ── Synthesis (sync wrapper) ─────────────────────────────────────────

    def synthesize(
        self, text: str, emotion: str, intensity: float = 1.0,
        language: str = "en", voice: str = "",
    ) -> Tuple[str, VoiceParameters]:
        """Generate speech audio file. Returns (filename, parameters)."""
        params = self.get_voice_parameters(emotion, intensity, language)
        resolved_voice = self._resolve_voice(voice, language)
        filename = f"speech_{uuid.uuid4().hex[:10]}.mp3"
        output_path = os.path.join(self.output_dir, filename)

        if self._engine == "edge-tts":
            asyncio.run(self._synthesize_edge_tts(text, params, output_path, resolved_voice))
        elif self._engine == "espeak-ng":
            filename = filename.replace(".mp3", ".wav")
            output_path = output_path.replace(".mp3", ".wav")
            self._synthesize_espeak(text, params, output_path, language)
        elif self._engine == "pyttsx3":
            filename = filename.replace(".mp3", ".wav")
            output_path = output_path.replace(".mp3", ".wav")
            self._synthesize_pyttsx3(text, params, output_path)

        return filename, params

    # ── Synthesis (async for FastAPI) ────────────────────────────────────

    async def synthesize_async(
        self, text: str, emotion: str, intensity: float = 1.0,
        language: str = "en", voice: str = "",
    ) -> Tuple[str, VoiceParameters]:
        """Async version for use inside FastAPI event loop."""
        params = self.get_voice_parameters(emotion, intensity, language)
        resolved_voice = self._resolve_voice(voice, language)
        filename = f"speech_{uuid.uuid4().hex[:10]}.mp3"
        output_path = os.path.join(self.output_dir, filename)

        if self._engine == "edge-tts":
            await self._synthesize_edge_tts(text, params, output_path, resolved_voice)
        elif self._engine == "espeak-ng":
            filename = filename.replace(".mp3", ".wav")
            output_path = output_path.replace(".mp3", ".wav")
            self._synthesize_espeak(text, params, output_path, language)
        elif self._engine == "pyttsx3":
            filename = filename.replace(".mp3", ".wav")
            output_path = output_path.replace(".mp3", ".wav")
            self._synthesize_pyttsx3(text, params, output_path)

        return filename, params

    # ── Engine-specific implementations ──────────────────────────────────

    async def _synthesize_edge_tts(self, text: str, params: VoiceParameters, path: str, voice: str):
        """Microsoft Edge Neural TTS — best quality, SSML prosody control."""
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=params.rate,
            pitch=params.pitch,
            volume=params.volume,
        )
        await communicate.save(path)

    def _synthesize_espeak(self, text: str, params: VoiceParameters, path: str, language: str = "en"):
        """espeak-ng — offline, full pitch/rate/amplitude control."""
        base_rate = 175
        rate_pct = int(params.rate.replace("%", "").replace("+", ""))
        rate = int(base_rate * (1 + rate_pct / 100))

        base_pitch = 50
        pitch_hz = int(params.pitch.replace("Hz", "").replace("+", ""))
        pitch = max(0, min(99, base_pitch + pitch_hz * 2))

        base_amp = 100
        vol_pct = int(params.volume.replace("%", "").replace("+", ""))
        amplitude = max(0, min(200, int(base_amp * (1 + vol_pct / 100))))

        # Map language to espeak voice
        espeak_voice = {"en": "en", "hi": "hi", "hinglish": "hi"}.get(language, "en")

        cmd = [
            "espeak-ng",
            "-v", espeak_voice,
            "-s", str(rate),
            "-p", str(pitch),
            "-a", str(amplitude),
            "-w", path,
            text,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def _synthesize_pyttsx3(self, text: str, params: VoiceParameters, path: str):
        """pyttsx3 — offline fallback with rate + volume control."""
        import pyttsx3

        engine = pyttsx3.init()

        base_rate = engine.getProperty("rate")
        rate_pct = int(params.rate.replace("%", "").replace("+", ""))
        engine.setProperty("rate", int(base_rate * (1 + rate_pct / 100)))

        base_vol = engine.getProperty("volume")
        vol_pct = int(params.volume.replace("%", "").replace("+", ""))
        engine.setProperty("volume", max(0.0, min(1.0, base_vol * (1 + vol_pct / 100))))

        engine.save_to_file(text, path)
        engine.runAndWait()
