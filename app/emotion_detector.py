"""
Emotion Detection Module for The Empathy Engine.

Analyzes input text and classifies it into 7 distinct emotional categories:
joy, anger, sadness, fear, surprise, disgust, neutral.

Primary:  HuggingFace transformer (j-hartmann/emotion-english-distilroberta-base)
Fallback: VADER sentiment analysis with keyword-based emotion refinement
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class EmotionResult:
    """Result of emotion analysis on input text."""
    emotion: str
    confidence: float
    all_scores: Dict[str, float]
    method: str  # "transformer" or "vader"


class EmotionDetector:
    """
    Detects emotion from text using a pre-trained transformer model.

    Supports 7 granular emotions: joy, anger, sadness, fear, surprise, disgust, neutral.
    Falls back to VADER + keyword heuristics when the transformer model is unavailable.
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

    def __init__(self, use_transformer: bool = True):
        self._pipeline = None
        self._vader = None
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
                return
            except Exception as e:
                print(f"[Empathy Engine] Transformer unavailable: {e}")
                print("[Empathy Engine] Falling back to VADER sentiment analysis...")

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

    def detect(self, text: str) -> EmotionResult:
        """Analyze text and return detected emotion with confidence scores."""
        if self._pipeline:
            return self._detect_transformer(text)
        return self._detect_vader(text)

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

        # Determine primary emotion
        if keyword_hits:
            primary = max(keyword_hits, key=keyword_hits.get)
            confidence = min(0.45 + abs(compound) * 0.5 + keyword_hits[primary] * 0.05, 0.95)
        elif compound >= 0.3:
            primary, confidence = "joy", min(0.4 + compound * 0.55, 0.95)
        elif compound <= -0.3:
            # Distinguish anger vs sadness by exclamation / caps
            has_emphasis = "!" in text or sum(1 for c in text if c.isupper()) > len(text) * 0.3
            primary = "anger" if has_emphasis else "sadness"
            confidence = min(0.4 + abs(compound) * 0.55, 0.95)
        elif compound > 0.05:
            primary, confidence = "joy", 0.35 + compound
        elif compound < -0.05:
            primary, confidence = "sadness", 0.35 + abs(compound)
        else:
            primary, confidence = "neutral", 0.70

        # Build score distribution
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

        # Boost keyword-matched emotions
        for emotion, count in keyword_hits.items():
            raw[emotion] = max(raw[emotion], 0.25 + count * 0.12)

        raw[primary] = primary_confidence

        # Normalize to sum ≈ 1.0
        total = sum(raw.values())
        return {k: round(v / total, 4) for k, v in raw.items()}
