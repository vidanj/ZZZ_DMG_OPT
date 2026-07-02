"""Loader and validator for the global damage-formula constants.

All values live in ``data/constants.json`` (never hardcoded here, per
.CLAUDE/RULES.md). This module is the only place that reads that file; the
rest of the code asks for validated constants through :class:`Constants`.

Usage::

    from zzz_dmg_calc.constants import load_constants

    consts = load_constants()
    coef = consts.level_coefficient(60)   # -> 794.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: Default location of the constants data file, relative to this package.
DATA_FILE = Path(__file__).parent / "data" / "constants.json"


class ConstantsError(ValueError):
    """Raised when the constants JSON is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class Constants:
    """Validated, immutable view of ``data/constants.json``.

    Attributes:
        level_coefficients: Attacker level -> level factor used in the DEF
            multiplier (e.g. ``{60: 794.0}``).
        base_crit_rate: CRIT Rate every agent starts with (fraction, 0.05).
        base_crit_dmg: CRIT DMG every agent starts with (fraction, 0.50).
        crit_rate_cap: Maximum effective CRIT Rate (fraction, 1.0).
    """

    level_coefficients: dict[int, float]
    base_crit_rate: float
    base_crit_dmg: float
    crit_rate_cap: float

    def level_coefficient(self, level: int) -> float:
        """Return the attacker level factor for ``level``.

        Raises:
            ConstantsError: if the level is not in the data table. Levels are
                never guessed or interpolated — add them to the JSON instead.
        """
        try:
            return self.level_coefficients[level]
        except KeyError:
            known = sorted(self.level_coefficients)
            raise ConstantsError(
                f"No level coefficient for attacker level {level}; "
                f"levels in data file: {known}"
            ) from None


def _require_fraction(data: dict, key: str) -> float:
    """Extract ``data[key]`` as a non-negative number, or raise."""
    if key not in data:
        raise ConstantsError(f"constants.json is missing required key '{key}'")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConstantsError(f"'{key}' must be a number, got {value!r}")
    if value < 0:
        raise ConstantsError(f"'{key}' must be >= 0, got {value}")
    return float(value)


def load_constants(path: Path = DATA_FILE) -> Constants:
    """Load and validate the constants file.

    Args:
        path: Alternate JSON file, mainly for tests.

    Raises:
        ConstantsError: if the file is missing, is not valid JSON, or fails
            validation (missing keys, wrong types, negative values).
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConstantsError(f"Constants file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ConstantsError(f"Constants file is not valid JSON: {exc}") from None

    if not isinstance(raw, dict):
        raise ConstantsError("constants.json must contain a JSON object")

    coeffs_raw = raw.get("level_coefficients")
    if not isinstance(coeffs_raw, dict) or not coeffs_raw:
        raise ConstantsError(
            "'level_coefficients' must be a non-empty object of level -> factor"
        )
    coefficients: dict[int, float] = {}
    for level_str, factor in coeffs_raw.items():
        try:
            level = int(level_str)
        except ValueError:
            raise ConstantsError(
                f"Level key {level_str!r} in 'level_coefficients' is not an integer"
            ) from None
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor <= 0:
            raise ConstantsError(
                f"Level coefficient for level {level} must be a positive number, "
                f"got {factor!r}"
            )
        coefficients[level] = float(factor)

    return Constants(
        level_coefficients=coefficients,
        base_crit_rate=_require_fraction(raw, "base_crit_rate"),
        base_crit_dmg=_require_fraction(raw, "base_crit_dmg"),
        crit_rate_cap=_require_fraction(raw, "crit_rate_cap"),
    )
