"""UI-agnostic calculation API: ``calculate(config) -> CalcResults``.

This is the single entry point every front end uses (Decision #4). The CLI
in ``main.py`` is one thin consumer; a future web UI calls this module the
same way. No I/O happens here beyond loading the JSON data files (which can
be injected for tests).

Flow::

    CalcConfig ──> aggregate_build() ──> formulas.* ──> CalcResults
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import formulas
from .agent import Agent, aggregate_build, load_agents
from .constants import Constants, load_constants
from .discs import Disc, DiscData, DiscSet, load_disc_data, load_disc_sets
from .enemies import Boss, load_bosses
from .engines import Engine, load_engines


@dataclass
class CalcConfig:
    """Everything the calculation needs, gathered by the front end.

    Attributes:
        agent_key: Key into the preset agent list (v1: ``"dummy"``).
        engine_key: Key into the engine database; ``None`` uses the agent's
            ``default_engine``.
        boss_name: Boss name as listed in the boss database.
        skill_multiplier: Skill's % of ATK as a fraction (2.5 = 250%).
            **Convention:** this is the move's TOTAL motion value — the
            calculation assumes the move lands in its entirety as one number.
            In-game damage popups are per *hit*, so a multi-hit move shows
            smaller individual numbers that sum to this result.
        discs: 0-6 discs with unique slots (validated during calculation).
            Discs carrying a ``disc_set`` key contribute set bonuses:
            2-piece bonuses automatically, 4-piece effects per ``set_stacks``.
        set_stacks: set key -> active stacks for modeled 4-piece effects
            (e.g. ``{"TECHNO": 3}`` = +27% ATK). Default: no stacks — 4-piece
            effects are combat-conditional, like engine/core passives.
        external_dmg_bonuses: Extra DMG% bracket entries (fractions) — e.g.
            the engine passive's Ice DMG +25% when active, team buffs.
        external_crit_rate: Extra CRIT Rate (fraction) from conditional
            buffs; added on top of the build's total before capping.
        external_crit_dmg: Extra CRIT DMG (fraction) from conditional,
            often skill-specific buffs — e.g. Ellen's core passive adds
            +100% CRIT DMG to her charged/Flash Freeze hits.
        external_res_shred: Total enemy RES shred/ignore (fraction).
        external_dmg_taken: "DMG taken" bracket entries (fractions) —
            composable per the future-proofing note (plan §7).
        external_daze_bonuses: Additive daze-vulnerability entries applied
            while the boss is stunned (fractions).
        stun_multiplier_override: Total stun DMG multiplier as displayed
            under the boss's daze bar in-game (fraction: 2.35 = 235%).
            ``None`` uses the boss database value. This REPLACES the boss's
            base multiplier (daze bonuses still add on top) — the in-game
            display already includes every active vulnerability effect, so
            reading it off the screen needs no arithmetic.
    """

    agent_key: str
    boss_name: str
    skill_multiplier: float
    engine_key: str | None = None
    discs: list[Disc] = field(default_factory=list)
    set_stacks: dict[str, int] = field(default_factory=dict)
    external_dmg_bonuses: list[float] = field(default_factory=list)
    external_crit_rate: float = 0.0
    external_crit_dmg: float = 0.0
    external_res_shred: float = 0.0
    external_dmg_taken: list[float] = field(default_factory=list)
    external_daze_bonuses: list[float] = field(default_factory=list)
    stun_multiplier_override: float | None = None


@dataclass(frozen=True)
class CalcResults:
    """Output table (plan §6) plus the intermediate values for transparency.

    Damage values cover every scenario combination: crit outcome
    (non-crit / crit / average) × boss state (normal / stunned).
    """

    # Damage table
    non_crit: float
    crit: float
    average: float
    non_crit_stunned: float
    crit_stunned: float
    average_stunned: float
    # Breakdown (for output formatting / debugging / cross-checking)
    atk_final: float
    crit_rate: float          # uncapped total; capping happens in the formula
    crit_dmg: float
    base_dmg: float
    dmg_bonus_mult: float
    def_mult: float
    res_mult: float
    dmg_taken_mult: float
    stun_mult: float


class CalcError(ValueError):
    """Raised when a config references unknown data (agent, engine, boss)."""


def calculate(
    config: CalcConfig,
    *,
    consts: Constants | None = None,
    disc_data: DiscData | None = None,
    bosses: dict[str, Boss] | None = None,
    agents: dict[str, Agent] | None = None,
    engines: dict[str, Engine] | None = None,
    disc_sets: dict[str, DiscSet] | None = None,
) -> CalcResults:
    """Run the full direct-hit damage calculation for one configuration.

    The keyword arguments allow tests (or a long-running UI) to inject
    already-loaded data; by default each data file is loaded from the
    package's ``data/`` directory.

    Raises:
        CalcError: unknown agent key or boss name.
        DiscError / AgentError: invalid discs (bad slot/main/substats/rolls).
        ConstantsError / EnemyError: malformed data files.
    """
    consts = consts if consts is not None else load_constants()
    disc_data = disc_data if disc_data is not None else load_disc_data()
    bosses = bosses if bosses is not None else load_bosses()
    agents = agents if agents is not None else load_agents()
    engines = engines if engines is not None else load_engines()
    disc_sets = disc_sets if disc_sets is not None else load_disc_sets()

    if config.agent_key not in agents:
        raise CalcError(
            f"Unknown agent '{config.agent_key}'; options: {sorted(agents)}"
        )
    agent = agents[config.agent_key]

    engine_key = config.engine_key if config.engine_key is not None else agent.default_engine
    if engine_key not in engines:
        raise CalcError(
            f"Unknown engine '{engine_key}'; options: {sorted(engines)}"
        )
    engine = engines[engine_key]

    if config.boss_name not in bosses:
        raise CalcError(
            f"Unknown boss '{config.boss_name}'; options: {sorted(bosses)}"
        )
    boss = bosses[config.boss_name]

    if config.skill_multiplier <= 0:
        raise CalcError("skill_multiplier must be > 0 (fraction: 2.5 = 250%)")

    build = aggregate_build(
        agent, engine, config.discs, disc_data, consts,
        disc_sets=disc_sets, set_stacks=config.set_stacks,
    )

    # --- Multiplier zones (formulas.py, plain numbers only) ---------------
    base = formulas.base_dmg(config.skill_multiplier, build.atk)

    crit_rate = build.crit_rate + config.external_crit_rate
    crit_dmg = build.crit_dmg + config.external_crit_dmg
    crit_none = formulas.crit_mult_non_crit()
    crit_full = formulas.crit_mult_crit(crit_dmg)
    crit_avg = formulas.crit_mult_average(
        crit_rate, crit_dmg, consts.crit_rate_cap
    )

    bonus = formulas.dmg_bonus_mult(
        build.dmg_bonuses + list(config.external_dmg_bonuses)
    )

    eff_def = formulas.effective_def(boss.base_def, build.pen_ratio, build.pen_flat)
    defense = formulas.def_mult(consts.level_coefficient(agent.level), eff_def)

    res = formulas.res_mult(
        boss.res_for(agent.attribute), res_ignore=config.external_res_shred
    )

    taken = formulas.dmg_taken_mult(config.external_dmg_taken)

    stun_base = (
        config.stun_multiplier_override
        if config.stun_multiplier_override is not None
        else boss.stun_dmg_multiplier
    )
    if stun_base < 1.0:
        raise CalcError(
            f"stun multiplier must be >= 1.0 (fraction: 2.35 = 235%), "
            f"got {stun_base}"
        )
    stun_off = formulas.stun_mult(False, stun_base)
    stun_on = formulas.stun_mult(True, stun_base, config.external_daze_bonuses)

    def dmg(crit_mult: float, stun: float) -> float:
        return formulas.total_dmg(base, crit_mult, bonus, defense, res, taken, stun)

    return CalcResults(
        non_crit=dmg(crit_none, stun_off),
        crit=dmg(crit_full, stun_off),
        average=dmg(crit_avg, stun_off),
        non_crit_stunned=dmg(crit_none, stun_on),
        crit_stunned=dmg(crit_full, stun_on),
        average_stunned=dmg(crit_avg, stun_on),
        atk_final=build.atk,
        crit_rate=crit_rate,
        crit_dmg=crit_dmg,
        base_dmg=base,
        dmg_bonus_mult=bonus,
        def_mult=defense,
        res_mult=res,
        dmg_taken_mult=taken,
        stun_mult=stun_on,
    )
