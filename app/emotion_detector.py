"""
Emotion Detection Module for The Empathy Engine.

Analyzes input text and classifies it into 7 distinct emotional categories:
joy, anger, sadness, fear, surprise, disgust, neutral.

Supports multilingual input (English, Hindi, Hinglish):
  - English text is analyzed directly by the transformer model.
  - Hindi/Hinglish text is translated to English first, then analyzed.

Primary:  HuggingFace transformer (j-hartmann/emotion-english-distilroberta-base)
Fallback: VADER sentiment analysis with keyword-based emotion refinement
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class EmotionResult:
    """Result of emotion analysis on input text."""
    emotion: str
    confidence: float
    all_scores: Dict[str, float]
    method: str               # "transformer" or "vader"
    translated_text: Optional[str] = None  # set when non-English input was translated


class EmotionDetector:
    """
    Detects emotion from text using a pre-trained transformer model.

    Supports 7 granular emotions: joy, anger, sadness, fear, surprise, disgust, neutral.
    Falls back to VADER + keyword heuristics when the transformer model is unavailable.
    Handles Hindi/Hinglish via automatic translation before analysis.
    """

    EMOTIONS = ["joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"]

    # Keyword lexicon for VADER fallback — maps words to specific emotions
    EMOTION_KEYWORDS = {
        "anger": [
            "angry", "furious", "outraged", "annoyed", "irritated", "mad",
            "hate", "terrible", "awful", "worst", "rage", "livid", "infuriated",
        ],
        "fear": [
            "afraid", "scared", "terrified", "worried", "anxious", "nervous",
            "panic", "dread", "frightened", "uneasy", "alarmed", "horror",
        ],
        "surprise": [
            "surprised", "shocked", "amazed", "astonished", "wow",
            "unexpected", "incredible", "unbelievable", "stunned", "whoa",
        ],
        "disgust": [
            "disgusting", "gross", "repulsive", "revolting", "nasty",
            "vile", "sick", "appalling", "repugnant", "loathsome",
        ],
        "sadness": [
            "sad", "depressed", "heartbroken", "miserable", "devastated",
            "grief", "sorrow", "crying", "lonely", "hopeless", "gloomy",
            "unhappy", "disappointed", "regret",
        ],
        "joy": [
            "happy", "excited", "thrilled", "delighted", "wonderful",
            "fantastic", "amazing", "love", "great", "awesome", "excellent",
            "ecstatic", "overjoyed", "grateful", "blessed", "cheerful",
        ],
    }

    # Hindi/Hinglish keywords for quick emotion hints (supplements translation)
    HINDI_EMOTION_KEYWORDS = {
        "joy": ["khushi", "khush", "maza", "badhai", "pyaar", "pyar", "accha", "badhiya", "zabardast", "shandar"],
        "anger": ["gussa", "naraz", "krodh", "chid", "bura", "ghatiya", "bakwas"],
        "sadness": ["dukh", "udaas", "rona", "akela", "tanha", "dard", "takleef"],
        "fear": ["darr", "dar", "khauf", "chinta", "pareshan", "ghabra"],
        "surprise": ["hairaan", "achanak", "sach", "yakeen", "chamak"],
    }

    def __init__(self, use_transformer: bool = True):
        self._pipeline = None
        self._vader = None
        self._translator = None
        self._method = "none"
        self._initialize(use_transformer)

    def _initialize(self, use_transformer: bool):
        """Try to load transformer model, fall back to VADER."""
        if use_transformer:
            try:
                from transformers import pipeline

                print("[Empathy Engine] Loading emotion classification model...")
                self._pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None,
                    device=-1,  # Force CPU
                )
                self._method = "transformer"
                print("[Empathy Engine] ✓ Transformer model loaded (7-class emotion)")
            except Exception as e:
                print(f"[Empathy Engine] Transformer unavailable: {e}")
                print("[Empathy Engine] Falling back to VADER sentiment analysis...")

        if not self._pipeline:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
                self._method = "vader"
                print("[Empathy Engine] ✓ VADER sentiment analyzer loaded")
            except ImportError:
                raise RuntimeError(
                    "No analysis backend available. "
                    "Install 'transformers' + 'torch' or 'vaderSentiment'."
                )

        # Load translator for Hindi/Hinglish support
        try:
            from deep_translator import GoogleTranslator
            self._translator = GoogleTranslator
            print("[Empathy Engine] ✓ Translation support loaded (Hindi/Hinglish)")
        except ImportError:
            print("[Empathy Engine] ⚠ deep-translator not installed; Hindi/Hinglish will use basic fallback")

    # ── Language utilities ───────────────────────────────────────────────

    @staticmethod
    def _has_devanagari(text: str) -> bool:
        """Check if text contains Devanagari (Hindi) script characters."""
        return bool(re.search(r'[\u0900-\u097F]', text))

    @staticmethod
    def _detect_language(text: str, language: str = "auto") -> str:
        """
        Determine the effective language of the text.
        If user specified a language, trust it. Otherwise auto-detect.
        """
        if language in ("hi", "hindi"):
            return "hi"
        if language in ("hinglish", "hi-en"):
            return "hinglish"
        if language in ("en", "english"):
            return "en"

        # Auto-detect: check for Devanagari characters
        if EmotionDetector._has_devanagari(text):
            return "hi"

        return "en"

    def _translate_to_english(self, text: str, source_lang: str) -> Optional[str]:
        """Translate Hindi/Hinglish text to English for emotion analysis."""
        if not self._translator:
            return None
        try:
            src = "hi" if source_lang == "hi" else "auto"
            translated = self._translator(source=src, target="en").translate(text)
            return translated
        except Exception as e:
            print(f"[Empathy Engine] Translation failed: {e}")
            return None

    # ── Main detect method ───────────────────────────────────────────────

    def detect(self, text: str, language: str = "auto") -> EmotionResult:
        """
        Analyze text and return detected emotion with confidence scores.

        For Hindi/Hinglish: translates to English first, then analyzes.
        The original text is still used for TTS output.
        """
        lang = self._detect_language(text, language)
        translated_text = None
        analysis_text = text

        # Translate if non-English
        if lang in ("hi", "hinglish"):
            translated = self._translate_to_english(text, lang)
            if translated:
                translated_text = translated
                analysis_text = translated

        # Run emotion detection on (possibly translated) text
        if self._pipeline:
            result = self._detect_transformer(analysis_text)
        else:
            result = self._detect_vader(analysis_text)

        result.translated_text = translated_text
        return result

    # ── Transformer-based detection ──────────────────────────────────────

    def _detect_transformer(self, text: str) -> EmotionResult:
        """Use the distilroberta emotion model for 7-class classification."""
        results = self._pipeline(text)[0]
        scores = {r["label"]: round(r["score"], 4) for r in results}
        top = max(results, key=lambda x: x["score"])
        return EmotionResult(
            emotion=top["label"],
            confidence=round(top["score"], 4),
            all_scores=scores,
            method="transformer",
        )

    # ── VADER fallback with keyword refinement ───────────────────────────

    def _detect_vader(self, text: str) -> EmotionResult:
        """
        Use VADER compound score + keyword matching to approximate
        granular emotions from basic positive/negative/neutral sentiment.
        """
        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]
        text_lower = text.lower()

        # Count keyword matches for each emotion
        keyword_hits = {}
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                keyword_hits[emotion] = count

        # Also check Hindi/Hinglish keywords
        for emotion, keywords in self.HINDI_EMOTION_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                keyword_hits[emotion] = keyword_hits.get(emotion, 0) + count

        # Determine primary emotion
        if keyword_hits:
            primary = max(keyword_hits, key=keyword_hits.get)
            confidence = min(0.45 + abs(compound) * 0.5 + keyword_hits[primary] * 0.05, 0.95)
        elif compound >= 0.3:
            primary, confidence = "joy", min(0.4 + compound * 0.55, 0.95)
        elif compound <= -0.3:
            has_emphasis = "!" in text or sum(1 for c in text if c.isupper()) > len(text) * 0.3
            primary = "anger" if has_emphasis else "sadness"
            confidence = min(0.4 + abs(compound) * 0.55, 0.95)
        elif compound > 0.05:
            primary, confidence = "joy", 0.35 + compound
        elif compound < -0.05:
            primary, confidence = "sadness", 0.35 + abs(compound)
        else:
            primary, confidence = "neutral", 0.70

        all_scores = self._build_vader_scores(scores, keyword_hits, primary, confidence)

        return EmotionResult(
            emotion=primary,
            confidence=round(confidence, 4),
            all_scores=all_scores,
            method="vader",
        )

    def _build_vader_scores(
        self,
        vader_scores: dict,
        keyword_hits: dict,
        primary: str,
        primary_confidence: float,
    ) -> Dict[str, float]:
        """Build a plausible score distribution from VADER + keyword data."""
        raw = {
            "joy": max(vader_scores["pos"] * 0.8, 0.01),
            "anger": max(vader_scores["neg"] * 0.30, 0.01),
            "sadness": max(vader_scores["neg"] * 0.30, 0.01),
            "fear": max(vader_scores["neg"] * 0.15, 0.01),
            "disgust": max(vader_scores["neg"] * 0.10, 0.01),
            "surprise": 0.02,
            "neutral": max(vader_scores["neu"] * 0.5, 0.05),
        }

        for emotion, count in keyword_hits.items():
            raw[emotion] = max(raw[emotion], 0.25 + count * 0.12)

        raw[primary] = primary_confidence

        total = sum(raw.values())
        return {k: round(v / total, 4) for k, v in raw.items()}
