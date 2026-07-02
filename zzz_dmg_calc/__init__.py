"""ZZZ DMG Optimizer — direct-hit damage calculator for Zenless Zone Zero.

Package layout (see DOCS/zzz_dmg_calc_plan.md §3):

- ``formulas``  — pure damage math, plain numbers in / plain numbers out.
- ``constants`` — loads and validates ``data/constants.json``.
- ``data/``     — all game data as JSON; no game values are hardcoded in code.

Later phases add ``discs``, ``enemies``, ``agent``, ``api`` and ``main``.
"""
