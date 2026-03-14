"""
FastAPI Web Application for The Empathy Engine.

Provides:
  GET  /                    — Web UI (interactive demo page)
  POST /api/synthesize       — Emotion detection + speech synthesis
  GET  /api/voices/{lang}    — List available voices for a language
  GET  /audio/{file}         — Serve generated audio files
"""

import os
from typing import Optional

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .empathy_engine import EmpathyEngine

# ── App setup ────────────────────────────────────────────────────────────

app = FastAPI(
    title="The Empathy Engine",
    description="Giving AI a Human Voice — emotionally modulated multilingual TTS",
    version="2.0.0",
)

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Initialize engine (loads model on startup)
engine = EmpathyEngine(output_dir=OUTPUT_DIR)


# ── Request / Response models ────────────────────────────────────────────

class SynthesizeRequest(BaseModel):
    text: str
    language: Optional[str] = "auto"   # "en", "hi", "hinglish", or "auto"
    voice: Optional[str] = ""          # voice ID or "" for default


# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the interactive web UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/voices/{language}")
async def list_voices(language: str):
    """Return available voices for a given language."""
    voices = engine.get_available_voices(language)
    return {"language": language, "voices": voices}


@app.post("/api/synthesize")
async def synthesize(req: SynthesizeRequest):
    """
    Main endpoint: analyze emotion and generate modulated speech.

    Request body:
      {
        "text": "Your text here",
        "language": "en" | "hi" | "hinglish" | "auto",
        "voice": "en-US-AriaNeural" | "" (for default)
      }
    """
    text = req.text.strip()

    if not text:
        return JSONResponse({"error": "Text cannot be empty."}, status_code=400)
    if len(text) > 2000:
        return JSONResponse({"error": "Text too long (max 2000 characters)."}, status_code=400)

    result = await engine.process_async(
        text,
        language=req.language or "auto",
        voice=req.voice or "",
    )
    return result.to_dict()


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve a generated audio file."""
    safe_name = os.path.basename(filename)
    path = os.path.join(OUTPUT_DIR, safe_name)

    if not os.path.exists(path):
        return JSONResponse({"error": "Audio file not found."}, status_code=404)

    media_type = "audio/mpeg" if safe_name.endswith(".mp3") else "audio/wav"
    return FileResponse(path, media_type=media_type, filename=safe_name)
