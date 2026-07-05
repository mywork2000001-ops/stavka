"""Convenience wrapper — `python demo.py` runs the same thing as `bukmeker demo`."""

from bukmeker.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo"]))
