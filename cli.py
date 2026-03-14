#!/usr/bin/env python3
"""
CLI interface for The Empathy Engine.

Usage:
    python cli.py "I'm so excited about this!"                     # English (default)
    python cli.py --lang hi "मुझे बहुत खुशी हो रही है!"              # Hindi
    python cli.py --lang hinglish "Yaar bahut maza aa raha hai!"   # Hinglish
    python cli.py --voice en-US-GuyNeural "Hello world!"           # Specific voice
    python cli.py                                                   # Interactive mode
"""

import argparse
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


LANG_NAMES = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish", "auto": "Auto"}


def print_banner():
    print()
    print("=" * 62)
    print("   The Empathy Engine — Giving AI a Human Voice")
    print("   Supports: English | Hindi | Hinglish")
    print("=" * 62)


def process_text(engine, text: str, language: str = "auto", voice: str = ""):
    """Analyze and synthesize a single text input."""
    print(f'\n  Input: "{text}"')
    print(f"  Language: {LANG_NAMES.get(language, language)}")
    print("-" * 62)

    result = engine.process(text, language=language, voice=voice)

    # ── Translation info ──
    if result.emotion.translated_text:
        print(f"\n  Translated: \"{result.emotion.translated_text}\"")

    # ── Emotion analysis ──
    print(f"\n  Detected Emotion:  {result.emotion.emotion.upper()}")
    print(f"  Confidence:        {result.emotion.confidence:.1%}")
    print(f"  Analysis Method:   {result.emotion.method}")

    print(f"\n  Emotion Scores:")
    sorted_scores = sorted(
        result.emotion.all_scores.items(), key=lambda x: x[1], reverse=True
    )
    for emotion, score in sorted_scores:
        bar_len = int(score * 30)
        bar = "\u2588" * bar_len + "\u2591" * (30 - bar_len)
        marker = " <--" if emotion == result.emotion.emotion else ""
        print(f"    {emotion:>10}  {bar}  {score:.1%}{marker}")

    # ── Voice parameters ──
    print(f"\n  Voice Parameters (intensity-scaled):")
    print(f"    Rate:    {result.voice_params.rate:>7}")
    print(f"    Pitch:   {result.voice_params.pitch:>7}")
    print(f"    Volume:  {result.voice_params.volume:>7}")
    print(f"    Voice:   {result.voice_used}")
    print(f"    Effect:  {result.voice_params.description}")

    # ── SSML ──
    ssml = result.voice_params.ssml.replace("{text}", text)
    print(f"\n  Generated SSML:")
    for line in ssml.split("\n"):
        print(f"    {line}")

    # ── Audio ──
    print(f"\n  Audio saved: output/{result.audio_filename}")
    print("-" * 62)


def interactive_mode(engine, language: str = "auto", voice: str = ""):
    """Run interactive prompt loop."""
    lang_label = LANG_NAMES.get(language, language)
    print(f"\n  Interactive Mode ({lang_label}) — type text and hear it spoken with emotion!")
    print("  Commands: 'quit' to exit, 'lang <en|hi|hinglish>' to switch language\n")

    current_lang = language

    while True:
        try:
            text = input("  > ").strip()
            if text.lower() in ("quit", "exit", "q"):
                print("\n  Goodbye!\n")
                break
            if text.lower().startswith("lang "):
                new_lang = text.split(maxsplit=1)[1].strip().lower()
                if new_lang in ("en", "hi", "hinglish", "auto"):
                    current_lang = new_lang
                    print(f"  Language set to: {LANG_NAMES.get(current_lang, current_lang)}\n")
                else:
                    print(f"  Unknown language. Use: en, hi, hinglish, auto\n")
                continue
            if not text:
                continue
            process_text(engine, text, language=current_lang, voice=voice)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!\n")
            break


def main():
    parser = argparse.ArgumentParser(description="The Empathy Engine — CLI")
    parser.add_argument("text", nargs="*", help="Text to synthesize (omit for interactive mode)")
    parser.add_argument("--lang", default="auto", choices=["auto", "en", "hi", "hinglish"],
                        help="Language: en, hi, hinglish, or auto (default: auto)")
    parser.add_argument("--voice", default="", help="Voice ID (e.g. hi-IN-SwaraNeural)")
    args = parser.parse_args()

    print_banner()

    from app.empathy_engine import EmpathyEngine

    engine = EmpathyEngine(output_dir="output")
    print()

    if args.text:
        text = " ".join(args.text)
        process_text(engine, text, language=args.lang, voice=args.voice)
    else:
        interactive_mode(engine, language=args.lang, voice=args.voice)


if __name__ == "__main__":
    main()
