"""W-Engine database loader.

Engines live in their own data file (``data/engines.json``) and are combined
with an agent only at build-aggregation time (``agent.aggregate_build``), so
the same agent can be tested with different engines by swapping one argument.

Scope: engines are Lv. 60 with a runtime refinement rank. Every
damage-relevant passive part is modeled and applied automatically (always-on
parts, auto-defaulted conditionals, squad buffs); manual external inputs are
reserved for effects outside the kit (Deadly Assault, boss mechanics). Each
engine's ``passive_note`` documents its source and what is resource-only.

Usage::

    from zzz_dmg_calc.engines import load_engines

    engines = load_engines()
    engine = engines["deep_sea_visitor"]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .enemies import ELEMENTS

#: Default location of the engine data file, relative to this package.
DATA_FILE = Path(__file__).parent / "data" / "engines.json"

#: Conditional-buff brackets that land in the damage formula. The first
#: six are read per mode by each calculation; ``atk_pct`` / ``res_shred``
#: / ``pen_ratio`` reach every mode through the kit totals (the
#: equipper's own engine is folded into ``api._kit_contributions``).
DAMAGE_BRACKETS = ("dmg_bonus", "anomaly_buff", "sheer_dmg", "crit_dmg",
                   "crit_rate", "anomaly_proficiency",
                   "atk_pct", "res_shred", "pen_ratio")

#: Conditional-buff brackets with NO damage term of their own: they raise a
#: stat of the equipper that one of THEIR kit buffs reads in real time
#: (Ice-Jade Teapot's Impact -> Qingyi's Impact -> ATK ability). Read only
#: by ``api._realtime_stat_bonus``. ``impact_pct`` is a fraction of the
#: equipper's base Impact; the other two are flat.
SCALING_BRACKETS = ("impact", "impact_pct", "anomaly_mastery")

#: Calculation modes a buff may be gated to (agent kit effects use the
#: same list).
BUFF_MODES = ("direct", "anomaly", "disorder", "vortex", "sheer")


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
    "Vortex and Windswept DMG"); ``"sheer_dmg"`` joins the dedicated
    Sheer DMG bracket (Qingming Birdcage); ``"crit_dmg"`` adds to the
    equipper's CRIT DMG total in crit-capable modes (Heartstring
    Nocturne's unconditional +50..80%); ``"crit_rate"`` likewise adds to
    the equipper's CRIT Rate total (Deep Sea Visitor's Basic / Ice-Dash
    stacks); ``"anomaly_proficiency"`` adds
    FLAT AP per stack to the equipper's AP total in anomaly modes only
    (Weeping Gemini's per-anomaly stacks — per-stack values are plain
    numbers, not fractions). ``"atk_pct"`` joins the combat ATK% bracket,
    ``"res_shred"`` the enemy RES ignore and ``"pen_ratio"`` the
    attacker's PEN Ratio ("DMG ignores X% DEF" — Cordis Germina); those
    three reach every mode through the kit totals. The scaling-only
    brackets (``SCALING_BRACKETS``) never touch damage directly — they
    raise a stat one of the equipper's own kit buffs reads in real time.
    ``modes`` optionally gates the
    buff to calculation modes (same semantics as agent kit effects: the
    element gate is checked in direct/anomaly/sheer modes only).
    ``skill_tags`` optionally gates it to hits of those damage types
    (e.g. Qingming Birdcage's Sheer DMG applies to Ultimate/EX Special
    hits only).

    ``always_on`` marks an UNCONDITIONAL part listed beside a conditional
    one: it applies at its rank value whatever the stack counter says,
    which leaves the counter free for the conditional part (Heartstring
    Nocturne: CRIT DMG always, Fire RES ignore per Heartstring stack).

    An engine may model SEVERAL conditional buffs that share one stack
    counter (``Engine.conditional_buffs``) — e.g. Qingming Companion
    grants Ether DMG% *and* Ult/EX Sheer DMG% per stack.

    ``at_max_only`` marks an all-or-nothing part of such a shared counter:
    the buff grants ``per_stack(rank)`` **once**, and only once the counter
    has reached ``max_stacks`` (Severed Innocence's Electric DMG needs all
    3 Tranquil Blossom stacks, while its CRIT DMG part scales per stack).
    """

    name: str
    element: str | None
    per_stack_by_rank: tuple[float, ...]
    max_stacks: int
    bracket: str = "dmg_bonus"
    modes: tuple[str, ...] | None = None
    skill_tags: tuple[str, ...] | None = None
    at_max_only: bool = False
    auto: bool = False
    always_on: bool = False
    note: str = ""

    def value_at(self, rank: int, stacks: float) -> float:
        """Contribution at ``rank`` for a counter standing at ``stacks``.

        Ordinary parts scale linearly and clamp at their own
        ``max_stacks``; ``at_max_only`` parts are all-or-nothing;
        ``always_on`` parts ignore the counter.
        """
        if self.always_on:
            return self.per_stack(rank)
        if self.at_max_only:
            return (self.per_stack(rank) if stacks >= self.max_stacks
                    else 0.0)
        return self.per_stack(rank) * min(stacks, self.max_stacks)

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

    Optional gates, checked against the calculated hit like kit effects:
    ``elements`` (the dealt element must be one of them — Blazing
    Laurel's Wilt only raises Ice and Fire CRIT DMG), ``modes`` and
    ``skill_tags``. A buff whose ``modes`` explicitly lists disorder or
    vortex skips the element check there, as everywhere else.
    """

    name: str
    kind: str
    values_by_rank: tuple[float, ...]
    max_stacks: int = 1
    auto: bool = False
    elements: tuple[str, ...] | None = None
    modes: tuple[str, ...] | None = None
    skill_tags: tuple[str, ...] | None = None
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
        passive_crit_by_rank: Always-on rank-scaled CRIT Rate from the
            passive (Qingming Birdcage's +20..32%) — applied automatically
            to the equipper's CRIT Rate in crit-capable modes, empty when
            none.
        passive_crit_dmg_by_rank: Always-on rank-scaled CRIT DMG from the
            passive (Severed Innocence's unconditional +30..48%) — applied
            automatically to the equipper's CRIT DMG in crit-capable
            modes, empty when none. The *conditional* CRIT DMG parts of
            the same passive belong in ``conditional_buffs`` with
            ``bracket="crit_dmg"``.
        conditional_buffs: Rank-scaled stacking/on-off buffs sharing ONE
            stack counter (``CalcConfig.engine_buff_stacks``); empty when
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
    passive_crit_by_rank: tuple[float, ...] = ()
    passive_crit_dmg_by_rank: tuple[float, ...] = ()
    conditional_buffs: tuple[EngineBuff, ...] = ()
    squad_buffs: tuple[EngineSquadBuff, ...] = ()
    passive_note: str = ""

    @property
    def conditional_buff(self) -> EngineBuff | None:
        """First modeled conditional buff (back-compat accessor).

        Most engines model a single conditional; multi-part conditionals
        (Qingming Birdcage) should be iterated via ``conditional_buffs``.
        """
        return self.conditional_buffs[0] if self.conditional_buffs else None

    def _rank_value(self, table: tuple[float, ...], rank: int,
                    what: str) -> float:
        if not table:
            return 0.0
        if not 1 <= rank <= len(table):
            raise EngineError(
                f"Engine '{self.name}' {what} has no value for rank "
                f"{rank}; modeled ranks: 1..{len(table)}"
            )
        return table[rank - 1]

    def passive_ap(self, rank: int) -> float:
        """Always-on flat AP of the passive at ``rank`` (0 when unmodeled).

        Raises:
            EngineError: rank outside the modeled table.
        """
        return self._rank_value(self.passive_ap_by_rank, rank, "passive AP")

    def passive_crit(self, rank: int) -> float:
        """Always-on CRIT Rate of the passive at ``rank`` (0 when unmodeled).

        Raises:
            EngineError: rank outside the modeled table.
        """
        return self._rank_value(
            self.passive_crit_by_rank, rank, "passive CRIT Rate"
        )

    def passive_crit_dmg(self, rank: int) -> float:
        """Always-on CRIT DMG of the passive at ``rank`` (0 when unmodeled).

        Raises:
            EngineError: rank outside the modeled table.
        """
        return self._rank_value(
            self.passive_crit_dmg_by_rank, rank, "passive CRIT DMG"
        )


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

        def gate_list(raw: dict, field: str,
                      allowed: tuple[str, ...] | None = None):
            """Validate an optional non-empty gate list (``modes``,
            ``skill_tags``, ``elements``); ``None`` when absent."""
            values = raw.get(field)
            if values is None:
                return None
            if (not isinstance(values, list) or not values
                    or any(not isinstance(v, str) or not v.strip()
                           for v in values)
                    or (allowed is not None
                        and any(v not in allowed for v in values))):
                expected = (f"values from {list(allowed)}" if allowed
                            else "keys")
                raise EngineError(
                    f"Engine '{key}': buff '{field}' must be a non-empty "
                    f"list of {expected}"
                )
            return tuple(values)

        refinement_rank = 1
        max_rank = 1
        passive_dmg: list[EnginePassiveDmg] = []
        passive_ap: tuple[float, ...] = ()
        passive_crit: tuple[float, ...] = ()
        passive_crit_dmg: tuple[float, ...] = ()
        conditional_buffs: list[EngineBuff] = []
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

            passive_crit_raw = refinement.get("passive_crit_by_rank")
            if passive_crit_raw is not None:
                passive_crit = rank_values(
                    passive_crit_raw, "passive_crit_by_rank"
                )
            else:
                passive_crit = ()

            passive_cd_raw = refinement.get("passive_crit_dmg_by_rank")
            if passive_cd_raw is not None:
                passive_crit_dmg = rank_values(
                    passive_cd_raw, "passive_crit_dmg_by_rank"
                )
            else:
                passive_crit_dmg = ()

            def parse_buff(buff_raw) -> EngineBuff:
                """Validate one conditional-buff object."""
                if not isinstance(buff_raw, dict):
                    raise EngineError(
                        f"Engine '{key}': conditional buff entries must be "
                        f"objects"
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
                if bracket not in DAMAGE_BRACKETS + SCALING_BRACKETS:
                    raise EngineError(
                        f"Engine '{key}': buff 'bracket' must be one of "
                        f"{list(DAMAGE_BRACKETS + SCALING_BRACKETS)}, "
                        f"got {bracket!r}"
                    )
                element = buff_raw.get("element")
                if element is not None and element not in ELEMENTS:
                    raise EngineError(
                        f"Engine '{key}': buff 'element' must be one of "
                        f"{list(ELEMENTS)}, got {element!r}"
                    )
                always_on = bool(buff_raw.get("always_on", False))
                at_max_only = bool(buff_raw.get("at_max_only", False))
                if always_on and (max_stacks != 1 or at_max_only):
                    raise EngineError(
                        f"Engine '{key}': always_on buff '{buff_name}' must "
                        f"have max_stacks 1 and no at_max_only - it ignores "
                        f"the stack counter"
                    )
                return EngineBuff(
                    name=buff_name,
                    element=element,
                    per_stack_by_rank=per_rank,
                    max_stacks=max_stacks,
                    bracket=bracket,
                    modes=gate_list(buff_raw, "modes", BUFF_MODES),
                    skill_tags=gate_list(buff_raw, "skill_tags"),
                    at_max_only=at_max_only,
                    auto=bool(buff_raw.get("auto", False)),
                    always_on=always_on,
                    note=str(buff_raw.get("note", "")),
                )

            # One stack counter may drive several buff parts: accept the
            # single 'conditional_buff' object (existing data) and/or the
            # 'conditional_buffs' list (multi-part engines).
            single_raw = refinement.get("conditional_buff")
            if single_raw is not None:
                conditional_buffs.append(parse_buff(single_raw))
            multi_raw = refinement.get("conditional_buffs")
            if multi_raw is not None:
                if not isinstance(multi_raw, list) or not multi_raw:
                    raise EngineError(
                        f"Engine '{key}': 'conditional_buffs' must be a "
                        f"non-empty list"
                    )
                conditional_buffs.extend(parse_buff(b) for b in multi_raw)

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
                    auto=bool(item.get("auto", False)),
                    elements=gate_list(item, "elements", ELEMENTS),
                    modes=gate_list(item, "modes", BUFF_MODES),
                    skill_tags=gate_list(item, "skill_tags"),
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
            passive_crit_by_rank=passive_crit,
            passive_crit_dmg_by_rank=passive_crit_dmg,
            conditional_buffs=tuple(conditional_buffs),
            squad_buffs=tuple(squad_buffs),
            passive_note=str(entry.get("passive_note", "")),
        )
    return engines
