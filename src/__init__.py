"""Explainable Fake News Headline Analyzer package.

AL-2002 Artificial Intelligence Laboratory (Fall 2026)
FAST-NUCES Karachi Campus — Section BCY-5A (Idea 19, Part B)

Team:
- Muhammad Hasnain Memon (24K-2001) — Group Leader
- Akbar Hussain (24K-2026) — Member 2
- Ateeb Ahmed Qureshi (24K-2036) — Member 3
"""

from __future__ import annotations

SEED: int = 42
LABEL_NAMES: dict[int, str] = {0: "Real", 1: "Fake"}

__all__ = ["SEED", "LABEL_NAMES"]
