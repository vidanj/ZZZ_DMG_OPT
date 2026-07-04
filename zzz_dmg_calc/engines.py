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
    """A refinement-scaled conditional stacking buff.

    The buff grants ``per_stack(rank) × active stacks`` of DMG% applied
    only to damage of ``element`` (``None`` = any element). Active stacks
    are a runtime input, consistent with the conditional-effects policy;
    ``max_stacks == 1`` means a plain on/off effect (the UI shows a
    checkbox). The refinement rank is a runtime input too — the engine's
    data rank is just the default.

    ``bracket`` routes the value: ``"dmg_bonus"`` (default) joins the
    ordinary DMG% bracket; ``"anomaly_buff"`` joins the separate
    Anomaly/Disorder Buff Multiplier bracket (e.g. Joyau Doré's
    "Vortex and Windswept DMG"). ``modes`` optionally gates the buff to
    calculation modes (same semantics as agent kit effects: the element
    gate is checked in direct/anomaly modes only).
    """

    name: str
    element: str | None
    per_stack_by_rank: tuple[float, ...]
    max_stacks: int
    bracket: str = "dmg_bonus"
    modes: tuple[str, ...] | None = None
    note: str = ""

    def per_stack(self, rank: int) -> float:
        """Per-stack value at refinement ``rank`` (1-based).

        Raises:
            EngineError: rank outside the modeled table.
        """
        if not 1 <= rank <= len(self.per_stack_by_rank):
            raise EngineError(
                f"Buff '{self.name}' has no value for rank {rank}; "
                f"modeled ranks: 1..{len(self.per_stack_by_rank)}"
            )
        return self.per_stack_by_rank[rank - 1]


@dataclass(frozen=True)
class EnginePassiveDmg:
    """An always-on, refinement-scaled DMG% part of an engine passive.

    Unlike :class:`EngineBuff` there is no runtime condition — the bonus
    applies whenever damage matches ``element`` (``None`` = any element),
    e.g. Steel Cushion's "Increases Physical DMG by 20..40%".
    """

    element: str | None
    values_by_rank: tuple[float, ...]
    note: str = ""

    def value(self, rank: int) -> float:
        """DMG% value at refinement ``rank`` (1-based).

        Raises:
            EngineError: rank outside the modeled table.
        """
        if not 1 <= rank <= len(self.values_by_rank):
            raise EngineError(
                f"Engine passive DMG has no value for rank {rank}; "
                f"modeled ranks: 1..{len(self.values_by_rank)}"
            )
        return self.values_by_rank[rank - 1]


@dataclass(frozen=True)
class EngineSquadBuff:
    """A team-facing, refinement-scaled conditional buff on an engine.

    Worn by an (off-field) support, it buffs the whole squad — i.e. the
    on-field agent — while its condition holds (asked at runtime).
    ``kind`` uses the agent-kit effect kinds (crit_dmg, atk_pct, ...).
    """

    name: str
    kind: str
    values_by_rank: tuple[float, ...]
    max_stacks: int = 1
    note: str = ""

    def value(self, rank: int) -> float:
        """Per-stack value at refinement ``rank`` (1-based).

        Raises:
            EngineError: rank outside the modeled table.
        """
        if not 1 <= rank <= len(self.values_by_rank):
            raise EngineError(
                f"Squad buff '{self.name}' has no value for rank {rank}; "
                f"modeled ranks: 1..{len(self.values_by_rank)}"
            )
        return self.values_by_rank[rank - 1]


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
        refinement_rank: The DEFAULT rank R1-R5 (1 when no refinement data
            is modeled); the calculation may override it at runtime
            (``CalcConfig.engine_rank``).
        max_rank: Highest modeled rank (1 when no refinement data).
        passive_dmg: Always-on rank-scaled DMG% parts of the passive.
        passive_ap_by_rank: Always-on rank-scaled flat Anomaly Proficiency
            from the passive (e.g. Joyau Doré's AP +70..110) — applied
            automatically to the equipper's AP total, empty when none.
        conditional_buff: Rank-scaled stacking/on-off buff, or ``None`` if
            no conditional part is modeled (see ``passive_note``).
        squad_buffs: Team-facing rank-scaled conditionals (support
            signature engines, e.g. Half-Sugar Bunny's squad CRIT DMG).
        passive_note: Human-readable reminder of non-auto-applied effects.
    """

    key: str
    name: str
    base_atk: float
    advanced_stat: dict[str, float]
    refinement_rank: int = 1
    max_rank: int = 1
    passive_dmg: tuple[EnginePassiveDmg, ...] = ()
    passive_ap_by_rank: tuple[float, ...] = ()
    conditional_buff: EngineBuff | None = None
    squad_buffs: tuple[EngineSquadBuff, ...] = ()
    passive_note: str = ""

    def passive_ap(self, rank: int) -> float:
        """Always-on flat AP of the passive at ``rank`` (0 when unmodeled).

        Raises:
            EngineError: rank outside the modeled table.
        """
        if not self.passive_ap_by_rank:
            return 0.0
        if not 1 <= rank <= len(self.passive_ap_by_rank):
            raise EngineError(
                f"Engine '{self.name}' passive AP has no value for rank "
                f"{rank}; modeled ranks: 1..{len(self.passive_ap_by_rank)}"
            )
        return self.passive_ap_by_rank[rank - 1]


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
        max_rank = 1
        passive_dmg: list[EnginePassiveDmg] = []
        passive_ap: tuple[float, ...] = ()
        conditional_buff = None
        refinement = entry.get("refinement")
        if refinement is not None:
            if not isinstance(refinement, dict):
                raise EngineError(f"Engine '{key}': 'refinement' must be an object")
            max_rank = refinement.get("max_rank", 5)
            rank = refinement.get("rank", 1)
            if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= max_rank:
                raise EngineError(
                    f"Engine '{key}': refinement 'rank' must be an integer "
                    f"1..{max_rank}"
                )
            refinement_rank = rank

            def rank_values(values, what: str) -> tuple[float, ...]:
                """Validate a per-rank list of ``max_rank`` positive numbers."""
                if (not isinstance(values, list) or len(values) != max_rank
                        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0
                               for v in values)):
                    raise EngineError(
                        f"Engine '{key}': '{what}' must be a list of "
                        f"{max_rank} positive numbers"
                    )
                return tuple(float(v) for v in values)

            for item in refinement.get("passive_dmg_by_rank", []):
                if not isinstance(item, dict):
                    raise EngineError(
                        f"Engine '{key}': 'passive_dmg_by_rank' entries must "
                        f"be objects"
                    )
                passive_dmg.append(EnginePassiveDmg(
                    element=item.get("element"),
                    values_by_rank=rank_values(item.get("values"), "values"),
                    note=str(item.get("note", "")),
                ))

            passive_ap_raw = refinement.get("passive_ap_by_rank")
            if passive_ap_raw is not None:
                passive_ap = rank_values(passive_ap_raw, "passive_ap_by_rank")
            else:
                passive_ap = ()

            buff_raw = refinement.get("conditional_buff")
            if buff_raw is not None:
                if not isinstance(buff_raw, dict):
                    raise EngineError(
                        f"Engine '{key}': 'conditional_buff' must be an object"
                    )
                per_rank = rank_values(
                    buff_raw.get("per_stack_by_rank"), "per_stack_by_rank"
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
                bracket = buff_raw.get("bracket", "dmg_bonus")
                if bracket not in ("dmg_bonus", "anomaly_buff"):
                    raise EngineError(
                        f"Engine '{key}': buff 'bracket' must be 'dmg_bonus' "
                        f"or 'anomaly_buff', got {bracket!r}"
                    )
                modes = buff_raw.get("modes")
                if modes is not None:
                    if (not isinstance(modes, list) or not modes
                            or any(m not in ("direct", "anomaly", "disorder",
                                             "vortex") for m in modes)):
                        raise EngineError(
                            f"Engine '{key}': buff 'modes' must be a "
                            f"non-empty list of calculation modes"
                        )
                    modes = tuple(modes)
                conditional_buff = EngineBuff(
                    name=buff_name,
                    element=buff_raw.get("element"),
                    per_stack_by_rank=per_rank,
                    max_stacks=max_stacks,
                    bracket=bracket,
                    modes=modes,
                    note=str(buff_raw.get("note", "")),
                )

        squad_buffs: list[EngineSquadBuff] = []
        if refinement is not None:
            for item in refinement.get("squad_buffs", []):
                if not isinstance(item, dict):
                    raise EngineError(
                        f"Engine '{key}': 'squad_buffs' entries must be objects"
                    )
                buff_name = item.get("name")
                if not isinstance(buff_name, str) or not buff_name.strip():
                    raise EngineError(
                        f"Engine '{key}': squad buff is missing a 'name'"
                    )
                kind = item.get("kind")
                if not isinstance(kind, str) or not kind.strip():
                    raise EngineError(
                        f"Engine '{key}': squad buff '{buff_name}' needs a 'kind'"
                    )
                max_stacks = item.get("max_stacks", 1)
                if isinstance(max_stacks, bool) or not isinstance(max_stacks, int) or max_stacks < 1:
                    raise EngineError(
                        f"Engine '{key}': squad buff 'max_stacks' must be an "
                        f"integer >= 1"
                    )
                squad_buffs.append(EngineSquadBuff(
                    name=buff_name,
                    kind=kind,
                    values_by_rank=rank_values(item.get("values"), "values"),
                    max_stacks=max_stacks,
                    note=str(item.get("note", "")),
                ))

        engines[key] = Engine(
            key=key,
            name=name,
            base_atk=float(base_atk),
            advanced_stat={s: float(v) for s, v in advanced.items()},
            refinement_rank=refinement_rank,
            max_rank=max_rank,
            passive_dmg=tuple(passive_dmg),
            passive_ap_by_rank=passive_ap,
            conditional_buff=conditional_buff,
            squad_buffs=tuple(squad_buffs),
            passive_note=str(entry.get("passive_note", "")),
        )
    return engines
