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
class EngineBuff:
    """A refinement-scaled conditional stacking buff (pilot feature).

    The buff grants ``per_stack × active stacks`` of DMG% applied only to
    damage of ``element`` (``None`` = any element). Active stacks are a
    runtime input, consistent with the conditional-effects policy.
    """

    name: str
    element: str | None
    per_stack: float          # value at the engine's refinement rank
    max_stacks: int
    note: str = ""


@dataclass(frozen=True)
class Engine:
    """W-Engine at Lv. 60.

    Attributes:
        key: Lookup key in the database.
        name: Display name.
        base_atk: Engine base ATK — the second ATK%-scaling bucket
            (plan §2's ``ATK_engine_base``).
        advanced_stat: stat name -> value, same naming/fraction conventions
            as disc stats (e.g. ``{"CRIT Rate": 0.24}``). Does not scale
            with refinement.
        refinement_rank: R1-R5 (1 when no refinement data is modeled).
        conditional_buff: Rank-scaled stacking buff, or ``None`` if the
            engine's passive isn't modeled (see ``passive_note``).
        passive_note: Human-readable reminder of non-auto-applied effects.
    """

    key: str
    name: str
    base_atk: float
    advanced_stat: dict[str, float]
    refinement_rank: int = 1
    conditional_buff: EngineBuff | None = None
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

        refinement_rank = 1
        conditional_buff = None
        refinement = entry.get("refinement")
        if refinement is not None:
            if not isinstance(refinement, dict):
                raise EngineError(f"Engine '{key}': 'refinement' must be an object")
            max_rank = refinement.get("max_rank", 5)
            rank = refinement.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= max_rank:
                raise EngineError(
                    f"Engine '{key}': refinement 'rank' must be an integer "
                    f"1..{max_rank}"
                )
            refinement_rank = rank

            buff_raw = refinement.get("conditional_buff")
            if buff_raw is not None:
                if not isinstance(buff_raw, dict):
                    raise EngineError(
                        f"Engine '{key}': 'conditional_buff' must be an object"
                    )
                per_rank = buff_raw.get("per_stack_by_rank")
                if (not isinstance(per_rank, list) or len(per_rank) != max_rank
                        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0
                               for v in per_rank)):
                    raise EngineError(
                        f"Engine '{key}': 'per_stack_by_rank' must be a list "
                        f"of {max_rank} positive numbers"
                    )
                max_stacks = buff_raw.get("max_stacks")
                if isinstance(max_stacks, bool) or not isinstance(max_stacks, int) or max_stacks < 1:
                    raise EngineError(
                        f"Engine '{key}': buff 'max_stacks' must be an "
                        f"integer >= 1"
                    )
                buff_name = buff_raw.get("name")
                if not isinstance(buff_name, str) or not buff_name.strip():
                    raise EngineError(
                        f"Engine '{key}': buff is missing a valid 'name'"
                    )
                conditional_buff = EngineBuff(
                    name=buff_name,
                    element=buff_raw.get("element"),
                    per_stack=float(per_rank[rank - 1]),
                    max_stacks=max_stacks,
                    note=str(buff_raw.get("note", "")),
                )

        engines[key] = Engine(
            key=key,
            name=name,
            base_atk=float(base_atk),
            advanced_stat={s: float(v) for s, v in advanced.items()},
            refinement_rank=refinement_rank,
            conditional_buff=conditional_buff,
            passive_note=str(entry.get("passive_note", "")),
        )
    return engines
