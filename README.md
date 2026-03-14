# The Empathy Engine: Giving AI a Human Voice

> An emotionally intelligent Text-to-Speech service that detects emotion from text and dynamically modulates vocal parameters (rate, pitch, volume) to produce expressive, human-like speech.

---

## Overview

Standard TTS systems produce flat, robotic speech. **The Empathy Engine** bridges the gap between text-based sentiment and expressive audio output by:

1. **Analyzing** input text for emotional content (7 granular emotions)
2. **Mapping** detected emotions to scientifically-grounded vocal parameters
3. **Scaling** modulation intensity based on the emotion's confidence score
4. **Generating** emotionally expressive speech audio

The result: *"I got promoted!"* sounds genuinely excited, while *"I feel so lonely"* sounds subdued and somber.

---

## Features

| Feature | Description |
|---------|-------------|
| **7 Granular Emotions** | Joy, Anger, Sadness, Fear, Surprise, Disgust, Neutral |
| **Intensity Scaling** | Modulation strength scales with confidence — subtle text gets subtle changes |
| **3 Vocal Parameters** | Rate (speed), Pitch (tone), Volume (amplitude) — all dynamically adjusted |
| **Web Interface** | Beautiful, interactive UI with real-time emotion visualization |
| **CLI Interface** | Full command-line interface for scripting and quick use |
| **SSML Generation** | Generates W3C-compliant SSML markup for each synthesis |
| **Multi-Engine TTS** | Auto-detects best engine: Edge-TTS (neural) → espeak-ng → pyttsx3 |
| **Graceful Fallback** | Transformer model → VADER sentiment (works even without GPU) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Text Input                                │
│              "I'm so excited about this!"                        │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Emotion Detector                                │
│  ┌─────────────────────┐    ┌──────────────────────────────┐    │
│  │ HuggingFace          │    │ VADER + Keyword Heuristics   │    │
│  │ distilroberta-base   │ OR │ (lightweight fallback)       │    │
│  │ 7-class emotion      │    │ compound score + keyword     │    │
│  └──────────┬──────────┘    └──────────────┬───────────────┘    │
│             └──────────────┬───────────────┘                     │
│                            ▼                                     │
│              Emotion: JOY  (confidence: 0.94)                    │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              Voice Parameter Modulator                           │
│                                                                  │
│  Emotion Profile (joy):                                          │
│    Rate: +18%  ──┐                                               │
│    Pitch: +10Hz  ├── × intensity (0.94) ── Scaled Parameters     │
│    Volume: +12%  ┘                                               │
│                                                                  │
│  Result: Rate: +16%,  Pitch: +9Hz,  Volume: +11%                │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                   TTS Engine                                     │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────┐            │
│  │  Edge-TTS    │  │ espeak-ng  │  │   pyttsx3    │            │
│  │  (neural)    │  │ (offline)  │  │  (fallback)  │            │
│  └──────┬───────┘  └─────┬──────┘  └──────┬───────┘            │
│         └────────────────┼─────────────────┘                     │
│                          ▼                                       │
│               Emotionally Modulated Audio (.mp3/.wav)            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Emotion-to-Voice Mapping

The vocal parameter profiles are grounded in prosody research from affective computing (Scherer, 2003 — *"Vocal Communication of Emotion"*):

| Emotion | Rate | Pitch | Volume | Rationale |
|---------|------|-------|--------|-----------|
| **Joy** | +18% | +10Hz | +12% | Energetic, upbeat — faster pace and raised pitch convey excitement |
| **Anger** | +8% | -6Hz | +25% | Forceful, intense — lower pitch with amplified volume signals authority |
| **Sadness** | -25% | -10Hz | -18% | Slow, somber — reduced rate and pitch reflect low energy and withdrawal |
| **Fear** | +22% | +12Hz | -12% | Rapid, tense — faster speech with raised pitch signals anxiety |
| **Surprise** | +12% | +15Hz | +18% | Exclamatory — notably raised pitch and volume for shock/amazement |
| **Disgust** | -12% | -7Hz | -6% | Deliberate, disdainful — slower delivery with lowered pitch |
| **Neutral** | 0% | 0Hz | 0% | No modulation — natural baseline delivery |

### Intensity Scaling

The raw profile values are **multiplied by the emotion's confidence score**:

- *"This is good"* → Joy (confidence: 0.62) → Rate: +11%, Pitch: +6Hz, Volume: +7%
- *"This is the BEST NEWS EVER!"* → Joy (confidence: 0.95) → Rate: +17%, Pitch: +9Hz, Volume: +11%

This ensures subtle text gets subtle modulation, while highly emotional text gets dramatic vocal shifts.

---

## Setup

### Prerequisites

- Python 3.9+
- Internet connection (for Edge-TTS and model download on first run)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/empathy-engine.git
cd empathy-engine

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install espeak-ng for offline TTS fallback
sudo apt-get install espeak-ng    # Ubuntu/Debian
brew install espeak-ng             # macOS
```

> **Note:** On first run, the Hugging Face transformer model (~330MB) will be downloaded automatically. Subsequent runs use the cached model.

### Quick Setup (one command)

```bash
chmod +x setup.sh && ./setup.sh
```

---

## Usage

### Web Interface (recommended)

```bash
python run_web.py
```

Open **http://localhost:8000** in your browser. The interactive UI lets you:
- Enter any text
- Click pre-made emotional examples
- See real-time emotion analysis with color-coded score bars
- View the voice parameter adjustments
- Listen to and download the generated audio
- Inspect the generated SSML markup

### Command Line Interface

```bash
# Single text
python cli.py "I'm absolutely thrilled about this opportunity!"

# Interactive mode
python cli.py
```

CLI output includes:
- Detected emotion with confidence score
- Full emotion score distribution (visual bars)
- Applied voice parameters with intensity scaling
- Generated SSML markup
- Path to saved audio file

---

## Project Structure

```
empathy-engine/
├── app/
│   ├── __init__.py
│   ├── emotion_detector.py    # 7-class emotion detection (transformer + VADER)
│   ├── voice_synthesizer.py   # TTS with emotion-driven prosody control
│   ├── empathy_engine.py      # Core orchestrator
│   ├── api.py                 # FastAPI web server
│   └── templates/
│       └── index.html         # Interactive web UI
├── output/                    # Generated audio files
├── cli.py                     # CLI entry point
├── run_web.py                 # Web server launcher
├── requirements.txt           # Python dependencies
├── setup.sh                   # One-command setup script
└── README.md                  # This file
```

---

## Design Decisions

### Why HuggingFace Transformers over TextBlob/VADER?

VADER provides basic positive/negative/neutral sentiment but cannot distinguish between anger and sadness (both negative) or fear and surprise. The `j-hartmann/emotion-english-distilroberta-base` model was specifically trained on 7 emotion classes, giving much richer emotional granularity. VADER is retained as a fallback with keyword-based refinement to approximate 7-class detection.

### Why Edge-TTS as primary engine?

Edge-TTS uses Microsoft's neural voice synthesis (same as Edge browser's Read Aloud), producing natural-sounding speech. It supports direct prosody control via rate, pitch, and volume parameters — perfectly matching our requirements. It's free, requires no API key, and generates high-quality MP3 output.

### Why intensity scaling?

A flat mapping (e.g., joy always gets +18% rate) ignores the nuance of language. The sentence *"That's nice"* and *"THIS IS INCREDIBLE!!!"* might both be classified as joy, but at very different intensities. By scaling parameters with the confidence score, we capture this gradation — delivering proportional expressiveness.

### Why three TTS fallback engines?

Different environments have different constraints. Edge-TTS needs internet; espeak-ng needs a system package; pyttsx3 is pure Python. The auto-detection cascade ensures the engine works out-of-the-box in virtually any environment.

---

## Bonus Features Implemented

- [x] **Granular Emotions** — 7 distinct categories (not just positive/negative/neutral)
- [x] **Intensity Scaling** — Modulation strength proportional to confidence score
- [x] **Web Interface** — Beautiful, responsive UI with FastAPI backend
- [x] **SSML Integration** — W3C-compliant SSML markup generated and displayed

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Emotion Detection | HuggingFace Transformers (distilroberta) + VADER |
| TTS (primary) | Edge-TTS (Microsoft Neural Voices) |
| TTS (offline) | espeak-ng / pyttsx3 |
| Web Framework | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (no build step) |

---

## License

MIT
