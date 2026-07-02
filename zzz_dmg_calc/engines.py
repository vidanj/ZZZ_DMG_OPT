"""W-Engine database loader.

Engines live in their own data file (``data/engines.json``) and are combined
with an agent only at build-aggregation time (``agent.aggregate_build``), so
the same agent can be tested with different engines by swapping one argument.

Scope (plan §1): engines are Lv. 60 / R1; refinement and passive effects are
never auto-applied — each engine's ``passive_note`` documents what the user
may enter manually as external buffs.

Usage::

    from zzz_dmg_calc.engines import load_engines

    engines = load_engines()
    engine = engines["dummy_engine"]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: Default location of the engine data file, relative to this package.
DATA_FILE = Path(__file__).parent / "data" / "engines.json"


class EngineError(ValueError):
    """Raised when the engine data file is missing, malformed, or invalid."""


@dataclass(frozen=True)
class Engine:
    """W-Engine at Lv. 60 / R1.

    Attributes:
        key: Lookup key in the database.
        name: Display name.
        base_atk: Engine base ATK — the second ATK%-scaling bucket
            (plan §2's ``ATK_engine_base``).
        advanced_stat: stat name -> value, same naming/fraction conventions
            as disc stats (e.g. ``{"CRIT Rate": 0.24}``).
        passive_note: Human-readable reminder of non-auto-applied effects.
    """

    key: str
    name: str
    base_atk: float
    advanced_stat: dict[str, float]
    passive_note: str = ""


def load_engines(path: Path = DATA_FILE) -> dict[str, Engine]:
    """Load and validate the engine database.

    Raises:
        EngineError: if the file is missing, malformed, or an entry is
            invalid (missing name, negative base ATK, bad advanced stat).
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EngineError(f"Engine data file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise EngineError(f"Engine data file is not valid JSON: {exc}") from None

    entries = raw.get("engines")
    if not isinstance(entries, dict) or not entries:
        raise EngineError("'engines' must be a non-empty object of key -> engine")

    engines: dict[str, Engine] = {}
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            raise EngineError(f"Engine '{key}' must be an object")

        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise EngineError(f"Engine '{key}' is missing a valid 'name'")

        base_atk = entry.get("base_atk")
        if isinstance(base_atk, bool) or not isinstance(base_atk, (int, float)) or base_atk < 0:
            raise EngineError(
                f"Engine '{key}': 'base_atk' must be a non-negative number"
            )

        advanced = entry.get("advanced_stat", {})
        if not isinstance(advanced, dict):
            raise EngineError(f"Engine '{key}': 'advanced_stat' must be an object")
        for stat, value in advanced.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise EngineError(
                    f"Engine '{key}': advanced stat '{stat}' must be a "
                    f"non-negative number"
                )

        engines[key] = Engine(
            key=key,
            name=name,
            base_atk=float(base_atk),
            advanced_stat={s: float(v) for s, v in advanced.items()},
            passive_note=str(entry.get("passive_note", "")),
        )
    return engines
