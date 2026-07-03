"""Root launcher for the local web UI — mirrors run.py for the CLI.

Usage::

    python run_ui.py [--port 8765] [--no-browser]
"""

from zzz_dmg_calc.ui.server import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ZZZ DMG Optimizer local web UI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true",
                        help="don't open the browser automatically")
    args = parser.parse_args()
    main(port=args.port, open_browser=not args.no_browser)
