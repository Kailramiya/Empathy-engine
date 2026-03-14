#!/usr/bin/env python3
"""
Launch the Empathy Engine web interface.

Usage:
    python run_web.py              # Start on http://localhost:8000
    python run_web.py --port 3000  # Custom port
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="The Empathy Engine — Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print()
    print("=" * 62)
    print("   The Empathy Engine — Web Interface")
    print(f"   Starting at http://{args.host}:{args.port}")
    print("=" * 62)
    print()

    uvicorn.run(
        "app.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
