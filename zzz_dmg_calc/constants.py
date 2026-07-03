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
        anomaly_level_multipliers: Attacker level -> anomaly level multiplier
            (e.g. ``{60: 2.0}`` — provisional, see data file note).
        base_crit_rate: CRIT Rate every agent starts with (fraction, 0.05).
        base_crit_dmg: CRIT DMG every agent starts with (fraction, 0.50).
        crit_rate_cap: Maximum effective CRIT Rate (fraction, 1.0).
        skill_tags: Skill-type tag key -> display name (e.g. ``"ultimate"``
            -> ``"Ultimate"``). Gates skill-type-conditional DMG% bonuses:
            a hit's tag is chosen by the user before calculating.
    """

    level_coefficients: dict[int, float]
    anomaly_level_multipliers: dict[int, float]
    base_crit_rate: float
    base_crit_dmg: float
    crit_rate_cap: float
    skill_tags: dict[str, str]

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

    def anomaly_level_multiplier(self, level: int) -> float:
        """Return the anomaly level multiplier for ``level``.

        Raises:
            ConstantsError: if the level is not in the data table.
        """
        try:
            return self.anomaly_level_multipliers[level]
        except KeyError:
            known = sorted(self.anomaly_level_multipliers)
            raise ConstantsError(
                f"No anomaly level multiplier for attacker level {level}; "
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

    def level_table(key: str) -> dict[int, float]:
        """Validate a level -> positive-number table under ``key``."""
        table_raw = raw.get(key)
        if not isinstance(table_raw, dict) or not table_raw:
            raise ConstantsError(
                f"'{key}' must be a non-empty object of level -> value"
            )
        table: dict[int, float] = {}
        for level_str, factor in table_raw.items():
            try:
                level = int(level_str)
            except ValueError:
                raise ConstantsError(
                    f"Level key {level_str!r} in '{key}' is not an integer"
                ) from None
            if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor <= 0:
                raise ConstantsError(
                    f"'{key}' value for level {level} must be a positive "
                    f"number, got {factor!r}"
                )
            table[level] = float(factor)
        return table

    tags_raw = raw.get("skill_tags")
    if not isinstance(tags_raw, dict) or not tags_raw:
        raise ConstantsError(
            "'skill_tags' must be a non-empty object of tag key -> display name"
        )
    skill_tags: dict[str, str] = {}
    for tag, label in tags_raw.items():
        if not isinstance(tag, str) or not tag.strip():
            raise ConstantsError(f"skill_tags key {tag!r} must be a non-empty string")
        if not isinstance(label, str) or not label.strip():
            raise ConstantsError(
                f"skill_tags['{tag}'] must be a non-empty display name"
            )
        skill_tags[tag] = label

    return Constants(
        level_coefficients=level_table("level_coefficients"),
        anomaly_level_multipliers=level_table("anomaly_level_multipliers"),
        base_crit_rate=_require_fraction(raw, "base_crit_rate"),
        base_crit_dmg=_require_fraction(raw, "base_crit_dmg"),
        crit_rate_cap=_require_fraction(raw, "crit_rate_cap"),
        skill_tags=skill_tags,
    )
