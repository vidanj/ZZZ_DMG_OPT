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
- "Attribute DMG%" from a slot-5 disc main is assumed to match the agent's
  attack attribute and lands in the DMG% bonus bracket.
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


@dataclass(frozen=True)
class Agent:
    """Preset agent at Lv. 60 with max core skill.

    ``base_atk`` already *excludes* core bonuses; use :meth:`total_base_atk`
    for the ATK%-scaling agent bucket. The engine is *not* part of the agent —
    ``default_engine`` only names a key in ``data/engines.json``.
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

    def total_base_atk(self) -> float:
        """Agent-side base ATK bucket: own base + flat core skill ATK."""
        return self.base_atk + self.core_bonus_atk


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
    other: dict[str, float] = field(default_factory=dict)


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
            default_engine=default_engine,
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
    )

    for stat, value in engine.advanced_stat.items():
        _fold_stat(build, stat, value)

    for disc in discs:
        for stat, value in disc_stats(disc, disc_data).items():
            _fold_stat(build, stat, value)

    if disc_sets is not None:
        for stat, value in set_bonus_stats(discs, disc_sets, set_stacks).items():
            _fold_stat(build, stat, value)

    build.atk = formulas.atk_final(
        agent_base_atk=agent.total_base_atk(),
        engine_base_atk=engine.base_atk,
        atk_pct_total=build.other.pop("_atk_pct", 0.0),
        atk_flat_total=build.other.pop("_atk_flat", 0.0),
    )
    # Combat ATK% buffs (e.g. set 4pc stacks) multiply the finished panel ATK.
    build.atk *= 1.0 + build.other.pop("_combat_atk_pct", 0.0)
    return build
