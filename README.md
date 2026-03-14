# The Empathy Engine: Giving AI a Human Voice

> A multilingual, emotionally intelligent Text-to-Speech service that detects emotion from text and dynamically modulates vocal parameters (rate, pitch, volume) to produce expressive, human-like speech in **English, Hindi, and Hinglish**.

---

## Overview

Standard TTS systems produce flat, robotic speech. **The Empathy Engine** bridges the gap between text-based sentiment and expressive audio output by:

1. **Analyzing** input text for emotional content (7 granular emotions)
2. **Translating** Hindi/Hinglish text to English for accurate emotion analysis
3. **Mapping** detected emotions to scientifically-grounded vocal parameters
4. **Scaling** modulation intensity based on the emotion's confidence score
5. **Generating** emotionally expressive speech using **neural voices** (human-like, not robotic)

The result: *"I got promoted!"* sounds genuinely excited, while *"मुझे बहुत दुख हो रहा है"* sounds slow and somber in a natural Hindi voice.

---

## Features

| Feature | Description |
|---------|-------------|
| **7 Granular Emotions** | Joy, Anger, Sadness, Fear, Surprise, Disgust, Neutral |
| **3 Languages** | English, Hindi, and Hinglish with auto-detection |
| **Neural Voices** | 8+ human-like Microsoft Neural voices (not robotic!) |
| **Intensity Scaling** | Modulation strength scales with confidence — subtle text gets subtle changes |
| **3 Vocal Parameters** | Rate (speed), Pitch (tone), Volume (amplitude) — all dynamically adjusted |
| **Web Interface** | Beautiful, interactive UI with real-time emotion visualization |
| **CLI Interface** | Full command-line interface with language switching |
| **SSML Generation** | Generates W3C-compliant SSML markup for each synthesis |
| **Translation Pipeline** | Hindi/Hinglish → English translation for accurate emotion analysis |
| **Graceful Fallback** | Transformer model → VADER sentiment; Edge-TTS → espeak-ng → pyttsx3 |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Text Input                                │
│  English: "I'm so excited!"                                      │
│  Hindi:   "मुझे बहुत खुशी हो रही है!"                                │
│  Hinglish: "Yaar bahut maza aa raha hai!"                        │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│               Language Detection + Translation                   │
│  Auto-detect: Devanagari script → Hindi                          │
│  Hindi/Hinglish → Google Translate → English (for analysis)      │
│  Original text preserved for TTS output                          │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Emotion Detector                                │
│  ┌─────────────────────┐    ┌──────────────────────────────┐    │
│  │ HuggingFace          │    │ VADER + Keyword Heuristics   │    │
│  │ distilroberta-base   │ OR │ (lightweight fallback)       │    │
│  │ 7-class emotion      │    │ + Hindi keyword support      │    │
│  └──────────┬──────────┘    └──────────────┬───────────────┘    │
│             └──────────────┬───────────────┘                     │
│              Emotion: JOY  (confidence: 0.94)                    │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              Voice Parameter Modulator                           │
│                                                                  │
│  Rate: +18%  ──┐                                                 │
│  Pitch: +10Hz  ├── × intensity (0.94) ── Scaled Parameters       │
│  Volume: +12%  ┘                                                 │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              Neural Voice Selection                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ English: Aria, Jenny, Guy, Davis, Sonia, Neerja         │    │
│  │ Hindi:   Swara (Female), Madhur (Male)                  │    │
│  │ Hinglish: Swara / Madhur / Neerja                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ▼                                       │
│           Emotionally Modulated Audio (.mp3)                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Available Voices

### English Voices
| Voice ID | Name | Description |
|----------|------|-------------|
| `en-US-AriaNeural` | Aria | US Female, Expressive (default) |
| `en-US-JennyNeural` | Jenny | US Female, Warm |
| `en-US-GuyNeural` | Guy | US Male, Casual |
| `en-US-DavisNeural` | Davis | US Male, Professional |
| `en-GB-SoniaNeural` | Sonia | UK Female, British |
| `en-IN-NeerjaNeural` | Neerja | Indian English Female |

### Hindi / Hinglish Voices
| Voice ID | Name | Description |
|----------|------|-------------|
| `hi-IN-SwaraNeural` | Swara | Hindi Female (default for Hindi/Hinglish) |
| `hi-IN-MadhurNeural` | Madhur | Hindi Male |

All voices are **Microsoft Neural voices** — they sound natural and human-like, not robotic.

---

## Emotion-to-Voice Mapping

The vocal parameter profiles are grounded in prosody research from affective computing (Scherer, 2003 — *"Vocal Communication of Emotion"*):

| Emotion | Rate | Pitch | Volume | Rationale |
|---------|------|-------|--------|-----------|
| **Joy** | +30% | +25Hz | +20% | Energetic, upbeat — faster pace and raised pitch convey excitement |
| **Anger** | +15% | -15Hz | +40% | Forceful, intense — lower pitch with amplified volume signals authority |
| **Sadness** | -40% | -30Hz | -25% | Slow, somber — reduced rate and pitch reflect low energy |
| **Fear** | +35% | +28Hz | -15% | Rapid, tense — faster speech with raised pitch signals anxiety |
| **Surprise** | +20% | +35Hz | +25% | Exclamatory — notably raised pitch and volume |
| **Disgust** | -20% | -18Hz | -10% | Deliberate, disdainful — slower delivery with lowered pitch |
| **Neutral** | 0% | 0Hz | 0% | No modulation — natural baseline delivery |

### Intensity Scaling

Raw profile values are **multiplied by the emotion's confidence score**:

- *"This is good"* → Joy (confidence: 0.62) → Rate: +11%, Pitch: +6Hz, Volume: +7%
- *"This is the BEST NEWS EVER!"* → Joy (confidence: 0.95) → Rate: +17%, Pitch: +9Hz, Volume: +11%

---

## Setup

### Prerequisites

- Python 3.9+
- Internet connection (for Edge-TTS neural voices and model download)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Kailramiya/Empathy-engine.git
cd Empathy-engine

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Quick Setup (one command)

```bash
chmod +x setup.sh && ./setup.sh
```

> **Note:** On first run, the HuggingFace transformer model (~330MB) will be downloaded automatically.

---

## Usage

### Web Interface (recommended)

```bash
python run_web.py
```

Open **http://localhost:8000** in your browser. Features:
- Enter text in **English, Hindi, or Hinglish**
- Select language and voice from dropdowns
- Click pre-made examples in all 3 languages
- See real-time emotion analysis with color-coded score bars
- See the translated text (for Hindi/Hinglish inputs)
- Listen to and download the generated audio
- Inspect the generated SSML markup

### Command Line Interface

```bash
# English (default)
python cli.py "I'm absolutely thrilled about this!"

# Hindi
python cli.py --lang hi "मुझे बहुत खुशी हो रही है!"

# Hinglish
python cli.py --lang hinglish "Yaar bahut maza aa raha hai!"

# Specific voice
python cli.py --voice hi-IN-MadhurNeural --lang hi "यह बहुत अच्छा है!"

# Interactive mode
python cli.py
# Then type 'lang hi' to switch to Hindi mid-session
```

---

## Project Structure

```
empathy-engine/
├── app/
│   ├── __init__.py
│   ├── emotion_detector.py    # 7-class emotion + Hindi/Hinglish translation
│   ├── voice_synthesizer.py   # Multi-voice TTS with prosody control
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

VADER provides basic positive/negative/neutral sentiment but cannot distinguish between anger and sadness (both negative) or fear and surprise. The `j-hartmann/emotion-english-distilroberta-base` model was specifically trained on 7 emotion classes. VADER is retained as a fallback with keyword-based refinement.

### Why translate Hindi/Hinglish before analysis?

The emotion classification model is trained on English text. Rather than using a weaker multilingual sentiment model, we translate to English using Google Translate (via `deep-translator`), then analyze with the full 7-class model. The original text is preserved for TTS — so the speech output is in the user's chosen language.

### Why Edge-TTS neural voices?

Edge-TTS uses Microsoft's Neural voice synthesis (same technology as Edge browser's Read Aloud). These voices sound **natural and human-like**, not robotic. They support direct prosody control via rate, pitch, and volume parameters, and offer voices in multiple languages including Hindi.

### Why intensity scaling?

A flat mapping ignores language nuance. *"That's nice"* and *"THIS IS INCREDIBLE!!!"* might both classify as joy, but at different intensities. Scaling parameters with confidence delivers proportional expressiveness.

---

## Bonus Features Implemented

- [x] **Granular Emotions** — 7 distinct categories (not just positive/negative/neutral)
- [x] **Intensity Scaling** — Modulation strength proportional to confidence score
- [x] **Web Interface** — Beautiful, responsive UI with FastAPI backend
- [x] **SSML Integration** — W3C-compliant SSML markup generated and displayed
- [x] **Multilingual** — English, Hindi, and Hinglish support with neural voices
- [x] **Multiple Voice Options** — 8+ selectable human-like neural voices

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Emotion Detection | HuggingFace Transformers (distilroberta) + VADER |
| Translation | deep-translator (Google Translate) |
| TTS (primary) | Edge-TTS (Microsoft Neural Voices) |
| TTS (offline) | espeak-ng / pyttsx3 |
| Web Framework | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (no build step) |

---

## License

MIT
