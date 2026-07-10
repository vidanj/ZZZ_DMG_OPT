"""Agent loading and full build stat aggregation.

Loads the preset agent list from ``data/agents.json`` (v1: only the DUMMY
agent — Ellen's values, per Decision #1) and aggregates a complete build:

    agent base stats (incl. max core skill bonuses)
    + W-Engine (separate database, passed in explicitly — swap freely)
    + drive discs (validated mains + substat rolls)
    -> BuildStats (plain numbers ready for formulas.py)

Engines are deliberately *not* embedded in the agent: an agent only names a
``default_engine`` key, and :func:`aggregate_build` takes the engine as its
own argument so the same agent is trivially tested with different engines.

Bucket rules enforced here (plan §2):

- ATK% scales only (agent base + core flat ATK + engine base); flat ATK from
  discs is added after — the buckets never mix.
- "Attribute DMG%" from a slot-5 disc main lands in the DMG% bonus bracket
  only when the disc's ``element`` matches the agent's attack attribute
  (optimizer_plan.md §11 E1). A disc without an element (legacy data) keeps
  the original assume-it-matches behavior.
- Stats irrelevant to direct-hit damage (HP, DEF, Anomaly, Impact, Energy
  Regen) are aggregated into ``BuildStats.other`` so nothing is silently
  dropped, but they do not affect the damage math in v1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .constants import Constants
from .discs import Disc, DiscData, DiscSet, disc_stats, set_bonus_stats
from .enemies import ELEMENTS
from .engines import Engine
from . import formulas

#: Default location of the agent data file, relative to this package.
DATA_FILE = Path(__file__).parent / "data" / "agents.json"


class AgentError(ValueError):
    """Raised for invalid agent data files or invalid build descriptions."""


#: Effect kinds a kit buff (core passive / mindscape / team buff) may carry.
#: Only damage-relevant effects are modeled (project convention): stat buffs
#: and enemy debuffs. Motion-value increases (skill Lv. +N) are documented in
#: notes but never modeled — the user enters multipliers manually.
#: Anomaly-mode kinds (Phase 5e): ``anomaly_buff`` joins the separate
#: Anomaly/Disorder "Buff Multiplier" bracket, ``disorder_mult_add`` adds to
#: the Disorder/Vortex burst base multiplier, ``anomaly_proficiency`` adds
#: flat AP.
#: Rupture kinds (Phase 5 Rupture): ``sheer_force`` adds flat Sheer Force,
#: ``hp_pct`` joins the combat Max-HP% bracket (multiplies the finished HP
#: panel, like Combat ATK%), ``sheer_dmg`` joins the dedicated Sheer DMG
#: bracket. ``def_red`` reduces enemy DEF (multiplicative with PEN Ratio in
#: the DEF zone — Sheer damage ignores DEF, so it matters in direct/anomaly
#: modes only).
KIT_EFFECT_KINDS = ("crit_rate", "crit_dmg", "dmg_bonus", "res_shred",
                    "dmg_taken", "flat_atk", "atk_pct", "daze_bonus",
                    "anomaly_buff", "disorder_mult_add",
                    "anomaly_proficiency", "pen_ratio",
                    "sheer_force", "hp_pct", "sheer_dmg", "def_red")

#: Calculation modes an effect may be gated to via its ``modes`` list.
#: An effect without ``modes`` applies in every mode (original behavior).
KIT_EFFECT_MODES = ("direct", "anomaly", "disorder", "vortex", "sheer")


@dataclass(frozen=True)
class EffectScaling:
    """A kit effect whose value scales with a build stat of its OWNER.

    Pilot: Zhao — her squad buffs scale with her initial Max HP, so the
    support entry asks for that number (``input`` names it) and the value
    resolves as ``base + floor(max(x - threshold, 0) / per_step) ×
    per_value``, clamped to ``cap`` when one is set.
    """

    input: str                # e.g. "initial_max_hp" (asked at runtime)
    base: float = 0.0
    threshold: float = 0.0
    per_step: float = 1.0
    per_value: float = 0.0
    cap: float | None = None

    def resolve(self, x: float) -> float:
        # 1e-9 guard: fractional steps (Velina: 0.01 Energy Regen) would
        # otherwise lose a step to float error ((1.92-1.2)//0.01 -> 71).
        steps = (max(x - self.threshold, 0.0) + 1e-9) // self.per_step
        value = self.base + steps * self.per_value
        return min(value, self.cap) if self.cap is not None else value


@dataclass(frozen=True)
class KitEffect:
    """One damage-relevant effect of a kit buff, per active stack.

    ``kind`` routes the value into the damage formula: ``crit_rate`` /
    ``crit_dmg`` add to the crit totals, ``dmg_bonus`` joins the additive
    DMG% bracket, ``res_shred`` adds enemy RES ignore, ``dmg_taken`` adds
    a "DMG taken" bracket entry, ``flat_atk`` adds flat ATK to the final
    panel, ``daze_bonus`` adds to the stunned-state multiplier. Values are
    fractions per stack (``flat_atk`` is a plain number).

    ``skill_tag`` gates the effect to hits of that damage type (e.g.
    Nekomata's Catwalk applies only to EX Specials). ``scaling`` replaces
    the fixed ``value`` with an owner-stat-scaled one (see
    :class:`EffectScaling`).

    ``modes`` gates the effect to calculation modes (``direct`` /
    ``anomaly`` / ``disorder`` / ``vortex``); ``None`` applies everywhere.
    ``element`` gates it to damage dealt as that element. An effect that
    EXPLICITLY lists a disorder/vortex mode skips the element check there
    (those bursts deal another element on the kit owner's behalf); an
    effect without ``modes`` stays element-checked in every mode.
    """

    kind: str
    value: float = 0.0
    skill_tag: str | None = None
    scaling: EffectScaling | None = None
    modes: tuple[str, ...] | None = None
    element: str | None = None


@dataclass(frozen=True)
class KitBuff:
    """A conditional kit effect group (core passive, mindscape, additional
    ability, or team buff).

    Active stacks are a runtime input (0..``max_stacks``); ``max_stacks``
    of 1 means a plain on/off effect (the UI shows a checkbox).
    ``condition`` optionally describes a team-composition requirement as
    data (e.g. ``{"teammate_specialty_any_of": ["attack"]}`` or
    ``{"teammate_shares_attribute_or_faction": true}``) — the front ends
    use it to pre-check the toggle; the calculation itself always obeys
    the user's runtime choice.
    """

    name: str
    max_stacks: int
    effects: tuple[KitEffect, ...]
    note: str = ""
    condition: dict | None = None


@dataclass(frozen=True)
class Mindscape:
    """One Mindscape Cinema level (M1-M6), modeled or documentation-only.

    ``buff`` is ``None`` for levels with no damage-relevant effect (energy,
    motion values, QoL) — they stay in the data so the kit is complete, but
    the calculation ignores them (see ``note`` for why).
    """

    level: int
    name: str
    buff: KitBuff | None = None
    note: str = ""


@dataclass(frozen=True)
class Agent:
    """Preset agent at Lv. 60 with max core skill.

    ``base_atk`` already *excludes* core bonuses; use :meth:`total_base_atk`
    for the ATK%-scaling agent bucket (same for ``base_hp`` /
    :meth:`total_base_hp`). The engine is *not* part of the agent —
    ``default_engine`` only names a key in ``data/engines.json``.

    ``sheer_force_hp_conversion`` is the Rupture HP → Sheer Force rate:
    only Yixuan converts HP (0.10); it is 0 for everyone else, so the HP
    term of :func:`~.formulas.sheer_force` vanishes.
    """

    key: str
    name: str
    attribute: str
    level: int
    base_hp: float
    base_atk: float
    base_def: float
    base_anomaly_mastery: float
    base_anomaly_proficiency: float
    core_bonus_atk: float
    core_bonus_crit_rate: float
    core_bonus_anomaly_proficiency: float
    core_bonus_anomaly_mastery: float
    default_engine: str
    core_bonus_hp: float = 0.0
    sheer_force_hp_conversion: float = 0.0
    specialty: str = ""          # attack / stun / anomaly / support / defense / rupture
    faction: str | None = None
    core_passive: KitBuff | None = None
    additional_ability: KitBuff | None = None
    mindscapes: dict[int, Mindscape] = field(default_factory=dict)
    team_buffs: tuple[KitBuff, ...] = ()

    def total_base_atk(self) -> float:
        """Agent-side base ATK bucket: own base + flat core skill ATK."""
        return self.base_atk + self.core_bonus_atk

    def total_base_hp(self) -> float:
        """Agent-side base HP bucket: own base + flat core skill HP.

        Core flat HP (Yixuan's +420) is *base* HP — it scales with HP%,
        exactly like core flat ATK scales with ATK% (datamine: core flat
        hp/atk enter the base bucket; only hp_/atk_ core boosts are
        initial-bracket).
        """
        return self.base_hp + self.core_bonus_hp


@dataclass
class BuildStats:
    """Aggregated build totals, ready to feed ``formulas.py``.

    Attributes:
        atk: Final ATK (all buckets combined).
        crit_rate: Total CRIT Rate (uncapped — the formula layer caps it).
        crit_dmg: Total CRIT DMG.
        pen_ratio / pen_flat: Penetration stats for the DEF zone.
        anomaly_proficiency: Total AP (agent base + core + engine + discs).
        anomaly_mastery: Total AM (buildup speed — informational; not part
            of the per-proc damage formula).
        dmg_bonuses: DMG% bracket contributions (attribute DMG% from discs;
            external buffs are appended by the API layer).
        hp: Panel Max HP: ``hp_base × (1 + hp_pct) + hp_flat``. Team HP%
            buffs multiply this finished panel at the API layer (combat
            bracket, like Combat ATK%). Feeds Sheer Force (Phase 5 Rupture).
        hp_base: Agent base HP + core flat HP (the HP%-scaling bucket).
        hp_pct: Sheet HP% total (disc mains/subs + engine advanced stat).
        hp_flat: Flat HP total (disc mains/subs) — added after HP%.
        sheer_dmg_bonuses: Dedicated Sheer DMG bracket entries from the
            build itself (Yunkui Tales 4pc's "at 3 stacks Sheer DMG +10%"
            — calibrated in-game 2026-07-10); the API adds kit/engine/
            external entries on top. Only the sheer mode reads it.
        other: Aggregated stats that don't affect damage in v1.
    """

    atk: float
    crit_rate: float
    crit_dmg: float
    pen_ratio: float
    pen_flat: float
    anomaly_proficiency: float = 0.0
    anomaly_mastery: float = 0.0
    dmg_bonuses: list[float] = field(default_factory=list)
    hp: float = 0.0
    hp_base: float = 0.0
    hp_pct: float = 0.0
    hp_flat: float = 0.0
    sheer_dmg_bonuses: list[float] = field(default_factory=list)
    other: dict[str, float] = field(default_factory=dict)
    # Panel ATK before the combat bracket, and that bracket's sum — kept
    # separate so team ATK% buffs can join the same additive bracket
    # (in-game finding 2026-07-03): atk = atk_pre_combat × (1 + bracket).
    atk_pre_combat: float = 0.0
    combat_atk_pct: float = 0.0


def _parse_kit_buff(raw, where: str) -> KitBuff:
    """Validate and build one kit buff (core passive / modeled mindscape).

    Raises:
        AgentError: missing name, bad stacks, unknown effect kind, or a
            non-positive value.
    """
    if not isinstance(raw, dict):
        raise AgentError(f"{where} must be an object")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise AgentError(f"{where} is missing a valid 'name'")
    max_stacks = raw.get("max_stacks", 1)
    if isinstance(max_stacks, bool) or not isinstance(max_stacks, int) or max_stacks < 1:
        raise AgentError(f"{where}: 'max_stacks' must be an integer >= 1")
    effects_raw = raw.get("effects")
    if not isinstance(effects_raw, list) or not effects_raw:
        raise AgentError(f"{where}: 'effects' must be a non-empty list")
    effects: list[KitEffect] = []
    for item in effects_raw:
        if not isinstance(item, dict):
            raise AgentError(f"{where}: effect entries must be objects")
        kind = item.get("kind")
        if kind not in KIT_EFFECT_KINDS:
            raise AgentError(
                f"{where}: unknown effect kind {kind!r}; "
                f"options: {list(KIT_EFFECT_KINDS)}"
            )
        scaling = None
        scaling_raw = item.get("scaling")
        if scaling_raw is not None:
            if not isinstance(scaling_raw, dict) or not isinstance(
                    scaling_raw.get("input"), str):
                raise AgentError(
                    f"{where}: effect 'scaling' must be an object with an "
                    f"'input' name"
                )
            def scale_num(key: str, default: float) -> float:
                v = scaling_raw.get(key, default)
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise AgentError(
                        f"{where}: scaling '{key}' must be a number"
                    )
                return float(v)
            per_step = scale_num("per_step", 1.0)
            if per_step <= 0:
                raise AgentError(f"{where}: scaling 'per_step' must be > 0")
            cap_raw = scaling_raw.get("cap")
            scaling = EffectScaling(
                input=scaling_raw["input"],
                base=scale_num("base", 0.0),
                threshold=scale_num("threshold", 0.0),
                per_step=per_step,
                per_value=scale_num("per_value", 0.0),
                cap=None if cap_raw is None else scale_num("cap", 0.0),
            )
            value = 0.0
        else:
            value = item.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise AgentError(f"{where}: effect 'value' must be positive")
        skill_tag = item.get("skill_tag")
        if skill_tag is not None and (not isinstance(skill_tag, str) or not skill_tag.strip()):
            raise AgentError(f"{where}: effect 'skill_tag' must be a string")
        modes = item.get("modes")
        if modes is not None:
            if (not isinstance(modes, list) or not modes
                    or any(m not in KIT_EFFECT_MODES for m in modes)):
                raise AgentError(
                    f"{where}: effect 'modes' must be a non-empty list from "
                    f"{list(KIT_EFFECT_MODES)}"
                )
            modes = tuple(modes)
        element = item.get("element")
        if element is not None and element not in ELEMENTS:
            raise AgentError(
                f"{where}: effect 'element' must be one of {list(ELEMENTS)}, "
                f"got {element!r}"
            )
        effects.append(KitEffect(kind=kind, value=float(value),
                                 skill_tag=skill_tag, scaling=scaling,
                                 modes=modes, element=element))
    condition = raw.get("condition")
    if condition is not None and not isinstance(condition, dict):
        raise AgentError(f"{where}: 'condition' must be an object")
    return KitBuff(
        name=name,
        max_stacks=max_stacks,
        effects=tuple(effects),
        note=str(raw.get("note", "")),
        condition=condition,
    )


def _parse_mindscapes(raw, agent_key: str) -> dict[int, Mindscape]:
    """Validate the optional 'mindscapes' block (levels "1".."6").

    A level with an 'effects' list is modeled (parsed as a kit buff);
    a level without one is documentation-only (name + note).
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise AgentError(f"Agent '{agent_key}': 'mindscapes' must be an object")
    mindscapes: dict[int, Mindscape] = {}
    for level_str, entry in raw.items():
        if level_str not in {"1", "2", "3", "4", "5", "6"}:
            raise AgentError(
                f"Agent '{agent_key}': mindscape level must be '1'..'6', "
                f"got {level_str!r}"
            )
        level = int(level_str)
        if not isinstance(entry, dict):
            raise AgentError(
                f"Agent '{agent_key}': mindscape {level} must be an object"
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise AgentError(
                f"Agent '{agent_key}': mindscape {level} is missing a 'name'"
            )
        buff = None
        if entry.get("effects") is not None:
            buff = _parse_kit_buff(
                entry, f"Agent '{agent_key}' mindscape {level}"
            )
        mindscapes[level] = Mindscape(
            level=level,
            name=name,
            buff=buff,
            note=str(entry.get("note", "")),
        )
    return mindscapes


def load_agents(path: Path = DATA_FILE) -> dict[str, Agent]:
    """Load and validate the preset agent list.

    Raises:
        AgentError: if the file is missing, malformed, or an entry is invalid.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise AgentError(f"Agent data file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise AgentError(f"Agent data file is not valid JSON: {exc}") from None

    entries = raw.get("agents")
    if not isinstance(entries, dict) or not entries:
        raise AgentError("'agents' must be a non-empty object of key -> agent")

    def number(entry: dict, key: str, agent_key: str, minimum: float = 0.0) -> float:
        value = entry.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AgentError(f"Agent '{agent_key}': '{key}' must be a number")
        if value < minimum:
            raise AgentError(f"Agent '{agent_key}': '{key}' must be >= {minimum}")
        return float(value)

    agents: dict[str, Agent] = {}
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            raise AgentError(f"Agent '{key}' must be an object")

        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise AgentError(f"Agent '{key}' is missing a valid 'name'")

        attribute = entry.get("attribute")
        if attribute not in ELEMENTS:
            raise AgentError(
                f"Agent '{key}': 'attribute' must be one of {list(ELEMENTS)}, "
                f"got {attribute!r}"
            )

        core = entry.get("core_skill_bonuses", {})
        if not isinstance(core, dict):
            raise AgentError(f"Agent '{key}': 'core_skill_bonuses' must be an object")

        default_engine = entry.get("default_engine")
        if not isinstance(default_engine, str) or not default_engine.strip():
            raise AgentError(
                f"Agent '{key}' is missing 'default_engine' (a key into "
                f"data/engines.json)"
            )

        core_passive = None
        if entry.get("core_passive") is not None:
            core_passive = _parse_kit_buff(
                entry["core_passive"], f"Agent '{key}' core_passive"
            )
        additional_ability = None
        if entry.get("additional_ability") is not None:
            additional_ability = _parse_kit_buff(
                entry["additional_ability"], f"Agent '{key}' additional_ability"
            )
        mindscapes = _parse_mindscapes(entry.get("mindscapes"), key)

        team_buffs_raw = entry.get("team_buffs", [])
        if not isinstance(team_buffs_raw, list):
            raise AgentError(f"Agent '{key}': 'team_buffs' must be a list")
        team_buffs = tuple(
            _parse_kit_buff(item, f"Agent '{key}' team_buffs[{i}]")
            for i, item in enumerate(team_buffs_raw)
        )
        seen_names = [b.name for b in team_buffs]
        if len(seen_names) != len(set(seen_names)):
            raise AgentError(f"Agent '{key}': duplicate team buff names")

        hp_conversion = entry.get("sheer_force_hp_conversion", 0.0)
        if isinstance(hp_conversion, bool) or not isinstance(
                hp_conversion, (int, float)) or hp_conversion < 0:
            raise AgentError(
                f"Agent '{key}': 'sheer_force_hp_conversion' must be a "
                f"number >= 0"
            )

        agents[key] = Agent(
            key=key,
            name=name,
            attribute=attribute,
            level=int(number(entry, "level", key, minimum=1)),
            base_hp=number(entry, "base_hp", key),
            base_atk=number(entry, "base_atk", key),
            base_def=number(entry, "base_def", key),
            base_anomaly_mastery=number(entry, "anomaly_mastery", key),
            base_anomaly_proficiency=number(entry, "anomaly_proficiency", key),
            core_bonus_atk=float(core.get("base_atk", 0.0)),
            core_bonus_crit_rate=float(core.get("crit_rate", 0.0)),
            core_bonus_anomaly_proficiency=float(
                core.get("anomaly_proficiency", 0.0)
            ),
            core_bonus_anomaly_mastery=float(core.get("anomaly_mastery", 0.0)),
            core_bonus_hp=float(core.get("base_hp", 0.0)),
            sheer_force_hp_conversion=float(hp_conversion),
            default_engine=default_engine,
            specialty=str(entry.get("specialty", "")),
            faction=entry.get("faction"),
            core_passive=core_passive,
            additional_ability=additional_ability,
            mindscapes=mindscapes,
            team_buffs=team_buffs,
        )
    return agents


def _fold_stat(build: BuildStats, stat: str, value: float) -> None:
    """Route one named stat total into the right BuildStats bucket.

    Shared by disc totals and the engine's advanced stat so both use the
    exact same stat names (defined in discs.json / agents.json).

    Percent-ATK and flat-ATK are accumulated in ``other`` under internal keys
    first; :func:`aggregate_build` combines them into final ATK at the end.
    """
    if stat == "ATK%":
        build.other["_atk_pct"] = build.other.get("_atk_pct", 0.0) + value
    elif stat == "HP%":
        build.hp_pct += value
    elif stat == "HP":
        build.hp_flat += value
    elif stat == "Combat ATK%":
        # In-combat ATK buffs multiply final (panel) ATK — measured in-game
        # 2026-07-02 (Woodpecker 4pc: +300/+599 on a 3330 panel) — unlike
        # sheet ATK% which only scales base ATK.
        build.other["_combat_atk_pct"] = (
            build.other.get("_combat_atk_pct", 0.0) + value
        )
    elif stat == "ATK":
        build.other["_atk_flat"] = build.other.get("_atk_flat", 0.0) + value
    elif stat == "CRIT Rate":
        build.crit_rate += value
    elif stat == "CRIT DMG":
        build.crit_dmg += value
    elif stat == "PEN Ratio":
        build.pen_ratio += value
    elif stat == "PEN":
        build.pen_flat += value
    elif stat == "Anomaly Proficiency":
        build.anomaly_proficiency += value
    elif stat == "Anomaly Mastery":
        build.anomaly_mastery += value
    elif stat == "Attribute DMG%":
        build.dmg_bonuses.append(value)
    elif stat == "DMG%":
        # Generic (element-agnostic) DMG bonus, e.g. Fanged Metal 4pc —
        # same additive bracket as Attribute DMG%.
        build.dmg_bonuses.append(value)
    elif stat == "Sheer DMG%":
        # Dedicated Sheer DMG bracket (Yunkui Tales 4pc at max stacks) —
        # separate multiplicative zone, calibrated in-game 2026-07-10.
        build.sheer_dmg_bonuses.append(value)
    else:
        # HP, DEF, HP%, DEF%, Impact%, Energy Regen%, ... — tracked but not
        # part of damage in v1.
        build.other[stat] = build.other.get(stat, 0.0) + value


def aggregate_build(
    agent: Agent,
    engine: Engine,
    discs: list[Disc],
    disc_data: DiscData,
    consts: Constants,
    disc_sets: dict[str, DiscSet] | None = None,
    set_stacks: dict[str, int] | None = None,
) -> BuildStats:
    """Aggregate agent + engine + discs (+ set bonuses) into build stats.

    Args:
        agent: Preset agent from :func:`load_agents`.
        engine: W-Engine from :func:`~.engines.load_engines` — passed
            explicitly so the same agent can be aggregated with different
            engines (e.g. in comparison tests).
        discs: 0-6 user discs; slots must be unique. Each disc is validated
            against ``disc_data`` (raises :class:`~.discs.DiscError` if bad).
        disc_data: Loaded disc tables.
        consts: Loaded constants (base crit values).
        disc_sets: Set registry; when given, 2-piece bonuses apply
            automatically and modeled 4-piece effects apply at
            ``set_stacks`` active stacks (see :func:`~.discs.set_bonus_stats`).
            ``None`` skips set bonuses entirely.
        set_stacks: set key -> active stacks for 4-piece effects.

    Raises:
        AgentError: on duplicate disc slots.
        DiscError: if any disc fails validation, or on set/stack errors.
    """
    seen_slots: set[int] = set()
    for disc in discs:
        if disc.slot in seen_slots:
            raise AgentError(f"Duplicate disc slot {disc.slot}")
        seen_slots.add(disc.slot)

    build = BuildStats(
        atk=0.0,
        crit_rate=consts.base_crit_rate + agent.core_bonus_crit_rate,
        crit_dmg=consts.base_crit_dmg,
        pen_ratio=0.0,
        pen_flat=0.0,
        anomaly_proficiency=(
            agent.base_anomaly_proficiency
            + agent.core_bonus_anomaly_proficiency
        ),
        anomaly_mastery=(
            agent.base_anomaly_mastery + agent.core_bonus_anomaly_mastery
        ),
        hp_base=agent.total_base_hp(),
    )

    for stat, value in engine.advanced_stat.items():
        _fold_stat(build, stat, value)

    for disc in discs:
        for stat, value in disc_stats(disc, disc_data).items():
            if (stat == "Attribute DMG%" and disc.element is not None
                    and disc.element != agent.attribute):
                # Off-element disc: the DMG% buffs a different element, so
                # it contributes nothing to this agent (tracked, not lost).
                build.other["Attribute DMG% (off-element)"] = (
                    build.other.get("Attribute DMG% (off-element)", 0.0)
                    + value
                )
                continue
            _fold_stat(build, stat, value)

    if disc_sets is not None:
        for stat, value in set_bonus_stats(discs, disc_sets, set_stacks).items():
            _fold_stat(build, stat, value)

    build.atk_pre_combat = formulas.atk_final(
        agent_base_atk=agent.total_base_atk(),
        engine_base_atk=engine.base_atk,
        atk_pct_total=build.other.pop("_atk_pct", 0.0),
        atk_flat_total=build.other.pop("_atk_flat", 0.0),
    )
    # Combat ATK% buffs (set 4pc stacks, team ATK% buffs) multiply the
    # finished panel ATK in ONE additive bracket — measured in-game
    # 2026-07-03 (validation #4: Puffer 15% + Half-Sugar Bunny 10% each
    # applied to the unbuffed panel, not compounding). The API adds team
    # ATK% entries into the same bracket.
    build.combat_atk_pct = build.other.pop("_combat_atk_pct", 0.0)
    build.atk = build.atk_pre_combat * (1.0 + build.combat_atk_pct)
    # Max HP panel: base bucket × (1 + sheet HP%) + flat HP. Team HP% buffs
    # (Lucia, Dreamlit Hearth) multiply this finished panel at the API layer
    # — the combat HP bracket mirrors the combat ATK% structure (datamine:
    # final hp = initial hp × (1 + combat hp_) + combat hp).
    build.hp = build.hp_base * (1.0 + build.hp_pct) + build.hp_flat
    return build
