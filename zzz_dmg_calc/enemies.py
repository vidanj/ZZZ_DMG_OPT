"""Boss database loader (bosses only — no mobs, per plan §1/§5).

All boss data lives in ``data/bosses.json`` (see DOCS/sources.md for where
values were verified); this module loads, validates, and exposes it.

Data model per boss (plan §5):

- ``level`` / ``base_def``: enemy DEF stops growing after Lv. 60, so a single
  DEF value covers max-level content (952.8 for bosses).
- ``res``: fraction of damage resisted per attribute — a weakness is
  negative (−0.20 → ×1.20 damage), a resistance positive (+0.20 → ×0.80).
  Feeds directly into :func:`zzz_dmg_calc.formulas.res_mult`.
- ``stun_dmg_multiplier``: shown under the boss's daze bar in-game; feeds
  :func:`zzz_dmg_calc.formulas.stun_mult`.

The JSON has a ``defaults`` block (level, base_def, stun multiplier) that
individual bosses may override — bosses only need to state what differs.

Usage::

    from zzz_dmg_calc.enemies import load_bosses

    db = load_bosses()
    boss = db["Miasma Priest"]
    boss.res_for("ether")   # -> -0.2 (weakness)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: Default location of the boss data file, relative to this package.
DATA_FILE = Path(__file__).parent / "data" / "bosses.json"

#: Attack attributes every boss must declare a RES entry for.
ELEMENTS = ("physical", "fire", "ice", "electric", "ether", "wind")


class EnemyError(ValueError):
    """Raised when the boss data file is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class Boss:
    """Validated boss entry.

    Attributes:
        name: Display name, also the lookup key.
        level: Boss level (max-level content).
        base_def: DEF at that level (pre-penetration).
        res: attribute -> RES fraction (negative = weakness).
        stun_dmg_multiplier: Damage multiplier while stunned.
    """

    name: str
    level: int
    base_def: float
    res: dict[str, float]
    stun_dmg_multiplier: float

    def res_for(self, element: str) -> float:
        """RES fraction against ``element`` (case-insensitive).

        Raises:
            EnemyError: if ``element`` is not a known attribute.
        """
        key = element.lower()
        if key not in self.res:
            raise EnemyError(
                f"Unknown attack attribute '{element}'; options: {list(ELEMENTS)}"
            )
        return self.res[key]


def _number(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EnemyError(f"{what} must be a number, got {value!r}")
    return float(value)


def load_bosses(path: Path = DATA_FILE) -> dict[str, Boss]:
    """Load and validate the boss database.

    Returns:
        Mapping of boss name -> :class:`Boss`, in file order (dicts preserve
        insertion order, so the CLI can list them as authored).

    Raises:
        EnemyError: if the file is missing, malformed, or a boss entry is
            invalid (missing name, bad numbers, missing/unknown RES elements,
            duplicate names).
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EnemyError(f"Boss data file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise EnemyError(f"Boss data file is not valid JSON: {exc}") from None

    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict):
        raise EnemyError("'defaults' must be an object")

    entries = raw.get("bosses")
    if not isinstance(entries, list) or not entries:
        raise EnemyError("'bosses' must be a non-empty list of boss entries")

    db: dict[str, Boss] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise EnemyError(f"Boss entry must be an object, got {entry!r}")

        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise EnemyError(f"Boss entry is missing a valid 'name': {entry!r}")
        if name in db:
            raise EnemyError(f"Duplicate boss name: '{name}'")

        def field(key: str):
            """Boss value with fallback to the file-level defaults."""
            if key in entry:
                return entry[key]
            if key in defaults:
                return defaults[key]
            raise EnemyError(
                f"Boss '{name}' is missing '{key}' and no default is defined"
            )

        level_raw = field("level")
        if isinstance(level_raw, bool) or not isinstance(level_raw, int) or level_raw < 1:
            raise EnemyError(f"Boss '{name}': 'level' must be a positive integer")

        base_def = _number(field("base_def"), f"Boss '{name}': 'base_def'")
        if base_def < 0:
            raise EnemyError(f"Boss '{name}': 'base_def' must be >= 0")

        stun = _number(
            field("stun_dmg_multiplier"), f"Boss '{name}': 'stun_dmg_multiplier'"
        )
        if stun < 1.0:
            raise EnemyError(
                f"Boss '{name}': 'stun_dmg_multiplier' must be >= 1.0, got {stun}"
            )

        res_raw = entry.get("res")
        if not isinstance(res_raw, dict):
            raise EnemyError(f"Boss '{name}': 'res' must be an object")
        unknown = sorted(set(res_raw) - set(ELEMENTS))
        if unknown:
            raise EnemyError(f"Boss '{name}': unknown RES elements {unknown}")
        missing = [e for e in ELEMENTS if e not in res_raw]
        if missing:
            raise EnemyError(f"Boss '{name}': missing RES elements {missing}")
        res = {
            element: _number(res_raw[element], f"Boss '{name}': res['{element}']")
            for element in ELEMENTS
        }

        db[name] = Boss(
            name=name,
            level=level_raw,
            base_def=base_def,
            res=res,
            stun_dmg_multiplier=stun,
        )

    return db
