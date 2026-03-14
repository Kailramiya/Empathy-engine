#!/usr/bin/env python3
"""
CLI interface for The Empathy Engine.

Usage:
    python cli.py "I'm so excited about this!"     # Single text
    python cli.py                                   # Interactive mode
"""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    print()
    print("=" * 62)
    print("   The Empathy Engine — Giving AI a Human Voice")
    print("=" * 62)


def process_text(engine, text: str):
    """Analyze and synthesize a single text input."""
    print(f'\n  Input: "{text}"')
    print("-" * 62)

    result = engine.process(text)

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
    print(f"    Effect:  {result.voice_params.description}")

    # ── SSML ──
    ssml = result.voice_params.ssml.replace("{text}", text)
    print(f"\n  Generated SSML:")
    for line in ssml.split("\n"):
        print(f"    {line}")

    # ── Audio ──
    print(f"\n  Audio saved: output/{result.audio_filename}")
    print("-" * 62)


def interactive_mode(engine):
    """Run interactive prompt loop."""
    print("\n  Interactive Mode — type text and hear it spoken with emotion!")
    print("  Type 'quit' or 'q' to exit.\n")

    while True:
        try:
            text = input("  > ").strip()
            if text.lower() in ("quit", "exit", "q"):
                print("\n  Goodbye!\n")
                break
            if not text:
                continue
            process_text(engine, text)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!\n")
            break


def main():
    print_banner()

    from app.empathy_engine import EmpathyEngine

    engine = EmpathyEngine(output_dir="output")
    print()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        process_text(engine, text)
    else:
        interactive_mode(engine)


if __name__ == "__main__":
    main()
