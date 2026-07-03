"""UI-agnostic calculation API.

This is the single entry point layer every front end uses (Decision #4).
The CLI in ``main.py`` is one thin consumer; a future web UI calls this
module the same way. No I/O happens here beyond loading the JSON data files
(which can be injected for tests).

Entry points::

    calculate(config)         -> CalcResults      # direct hits (v1)
    calculate_anomaly(config) -> AnomalyResults   # anomaly proc / Disorder

``calculate_anomaly`` computes the agent's own attribute anomaly, or — when
``config.disorder_replaced`` is set — the Disorder triggered by replacing
that element's anomaly (damage dealt as the REPLACED element).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import formulas
from .agent import Agent, aggregate_build, load_agents
from .anomalies import AnomalyData, load_anomalies
from .constants import Constants, load_constants
from .discs import (
    Disc, DiscData, DiscSet, load_disc_data, load_disc_sets,
    set_tagged_dmg_bonuses,
)
from .enemies import Boss, load_bosses
from .engines import Engine, load_engines


@dataclass
class BuildupSegment:
    """One slice of an anomaly's buildup, with its attacker-side state.

    Anomaly procs snapshot attacker-side values as a buildup-weighted
    average (mechanic discovery #2, DOCS/sources.md). A segment says
    "``share`` of the buildup happened under these buffs". Segments in
    ``CalcConfig.buildup_segments`` may sum to less than 1 — the remainder
    implicitly uses the config's top-level buff state.

    Attributes:
        share: Fraction of total buildup (0 < share <= 1).
        engine_buff_stacks: Engine conditional-buff stacks during this slice.
        set_stacks: Disc-set stacks (e.g. Combat ATK%) during this slice.
        external_dmg_bonuses: Attacker DMG% buffs active during this slice.
        external_anomaly_proficiency: Extra flat AP during this slice.
        atk_override: Replace the agent's computed ATK entirely — for
            **teammate** segments, enter that teammate's panel ATK.
        anomaly_proficiency_override: Replace total AP entirely — for
            teammate segments, their AP.
    """

    share: float
    engine_buff_stacks: float = 0.0
    set_stacks: dict[str, int] = field(default_factory=dict)
    external_dmg_bonuses: list[float] = field(default_factory=list)
    external_anomaly_proficiency: float = 0.0
    atk_override: float | None = None
    anomaly_proficiency_override: float | None = None


@dataclass
class SupportConfig:
    """One off-field teammate and the state of their team-facing effects.

    References a full roster ``agent_key`` (not a stripped copy) so that a
    future "off-field damage" feature can run :func:`calculate` with the
    same agent as the attacker.

    Attributes:
        agent_key: Roster key of the support.
        buffs: team buff name -> active stacks (0/1 for on-off effects).
            Names must match the agent's ``team_buffs`` entries.
        scaling_inputs: Values for owner-stat-scaled effects, e.g.
            ``{"initial_max_hp": 27000}`` for Zhao — her squad CRIT/DMG
            buffs scale with HER panel Max HP.
        squad_set: Key of a squad-facing 4-piece set the support wears
            (e.g. ``"SWING_JAZZ"``), or ``None``.
        squad_set_stacks: Active stacks of that squad effect (1/0 for
            on-off sets).
        engine_key: Engine the support wears, for its team-facing
            ``squad_buffs`` (e.g. Half-Sugar Bunny); ``None`` = the
            support's default engine.
        engine_rank: Refinement rank of that engine (``None`` = default).
        engine_buffs: squad buff name -> active stacks (1/0 for on-off).
    """

    agent_key: str
    buffs: dict[str, int] = field(default_factory=dict)
    scaling_inputs: dict[str, float] = field(default_factory=dict)
    squad_set: str | None = None
    squad_set_stacks: int = 0
    engine_key: str | None = None
    engine_rank: int | None = None
    engine_buffs: dict[str, int] = field(default_factory=dict)


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
        skill_tag: The hit's skill-type tag (key into the ``skill_tags``
            table in constants.json, e.g. ``"ultimate"``). Gates
            skill-type-conditional bonuses — e.g. Puffer Electro 4pc's
            Ultimate DMG +20% applies only to ``"ultimate"`` hits.
            ``None`` = untyped: no tagged bonus applies. Direct hits only.
        engine_rank: Refinement rank override (1..engine.max_rank);
            ``None`` uses the engine's data-file default. Scales the
            engine's modeled passive parts (always-on DMG% and the
            conditional buff's per-stack value).
        core_passive_active: Apply the agent's modeled core passive
            conditional (e.g. Nekomata's Stealthy Paws DMG +60%).
            Direct hits only.
        mindscapes: Mindscape level -> active stacks for the agent's
            MODELED mindscapes (e.g. ``{1: 1, 6: 3}``). On/off effects
            (max_stacks 1) use stacks 1/0. Unmodeled or unknown levels
            are rejected. Direct hits only.
        additional_ability_stacks: Active stacks of the on-field agent's
            Additional Ability (e.g. Nekomata's Catwalk, 0-2). The team
            condition is the user's responsibility (the UI pre-checks it);
            skill-tag-gated effects apply only to matching hits.
        scaling_inputs: Owner-stat values for the ON-FIELD agent's scaled
            kit effects (e.g. ``{"initial_max_hp": 27000}`` when Zhao is
            the attacker — her core passive CRIT scales with her HP).
        supports: Up to two off-field teammates (:class:`SupportConfig`)
            contributing team buffs / enemy debuffs / squad set effects.
            Direct hits only.
        discs: 0-6 discs with unique slots (validated during calculation).
            Discs carrying a ``disc_set`` key contribute set bonuses:
            2-piece bonuses automatically, 4-piece effects per ``set_stacks``.
        set_stacks: set key -> active stacks for modeled 4-piece effects
            (e.g. ``{"WOODPECKER_ELECTRO": 3}`` = +27% ATK). Default: no
            stacks — 4-piece
            effects are combat-conditional, like engine/core passives.
        engine_buff_stacks: Stacks of the engine's refinement-scaled
            conditional buff (pilot: Sharpened Stinger's Predatory Instinct).
            The per-stack value comes from the engine's refinement rank; the
            buff only applies to damage matching its element. **Fractions
            allowed for anomaly modes**: anomaly damage snapshots buffs as a
            buildup-weighted average (measured in-game 2026-07-02), so enter
            the average stack count *during buildup* — for clean numbers,
            keep the state constant while building (stacks auto-refresh on
            dash, so 3 is the realistic steady state).
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
        external_anomaly_proficiency: Extra AP from conditional buffs (e.g.
            Timeweaver's +75 on EX, Flight of Fancy stacks). Anomaly modes
            only.
        disorder_replaced: For Disorder calculations: the element of the
            anomaly being REPLACED (``None`` = plain anomaly proc).
        disorder_remaining_seconds: Time left on the replaced anomaly when
            Disorder triggers (used by proc-based conversions).
        disorder_elapsed_seconds: Time since the replaced anomaly was
            applied (used by decaying one-shot conversions).
    """

    agent_key: str
    boss_name: str
    skill_multiplier: float
    skill_tag: str | None = None
    engine_key: str | None = None
    engine_rank: int | None = None
    core_passive_active: bool = False
    mindscapes: dict[int, int] = field(default_factory=dict)
    additional_ability_stacks: int = 0
    scaling_inputs: dict[str, float] = field(default_factory=dict)
    supports: list[SupportConfig] = field(default_factory=list)
    discs: list[Disc] = field(default_factory=list)
    set_stacks: dict[str, int] = field(default_factory=dict)
    engine_buff_stacks: float = 0.0
    external_dmg_bonuses: list[float] = field(default_factory=list)
    external_crit_rate: float = 0.0
    external_crit_dmg: float = 0.0
    external_res_shred: float = 0.0
    external_dmg_taken: list[float] = field(default_factory=list)
    external_daze_bonuses: list[float] = field(default_factory=list)
    stun_multiplier_override: float | None = None
    external_anomaly_proficiency: float = 0.0
    buildup_segments: list[BuildupSegment] = field(default_factory=list)
    disorder_replaced: str | None = None
    disorder_remaining_seconds: float = 0.0
    disorder_elapsed_seconds: float = 0.0


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
    # Every active conditional buff as {source, kind, value} — the front
    # ends render this as an itemized "Active buffs" summary.
    buff_breakdown: tuple = ()


@dataclass(frozen=True)
class AnomalyResults:
    """Output of :func:`calculate_anomaly` (anomaly proc or Disorder).

    Anomalies cannot crit, so instead of the crit table the results carry
    per-proc and full-duration totals, each in normal and stunned states.
    For Disorder, the burst is a single number: ``per_proc == full``.
    """

    kind: str                 # "anomaly" or "disorder"
    anomaly_name: str         # e.g. "Assault", or "Disorder (Burn)"
    element: str              # element the damage is dealt as
    hits: int                 # procs over full duration (1 for bursts)
    per_proc: float
    per_proc_stunned: float
    full: float               # per_proc x hits (or the Disorder burst)
    full_stunned: float
    # Breakdown
    atk_final: float
    anomaly_proficiency: float
    ap_mult: float
    anomaly_level_mult: float
    anomaly_mult: float       # the composed per-proc/burst multiplier used
    dmg_bonus_mult: float
    def_mult: float
    res_mult: float
    dmg_taken_mult: float
    stun_mult: float


class CalcError(ValueError):
    """Raised when a config references unknown data (agent, engine, boss)."""


def _resolve_engine_rank(engine: Engine, override: int | None) -> int:
    """The refinement rank to calculate with (override or data default).

    Raises:
        CalcError: override outside 1..engine.max_rank.
    """
    if override is None:
        return engine.refinement_rank
    if isinstance(override, bool) or not isinstance(override, int) or not 1 <= override <= engine.max_rank:
        raise CalcError(
            f"Engine '{engine.name}' rank must be an integer "
            f"1..{engine.max_rank}, got {override!r}"
        )
    return override


def _engine_buff_bonus(
    engine: Engine, stacks: float, dealt_element: str, rank: int
) -> float:
    """DMG% contribution of the engine's conditional buff at ``stacks``.

    ``stacks`` may be fractional: anomaly procs snapshot buffs as a
    buildup-weighted average (in-game finding, 2026-07-02), so e.g. 2.35
    expresses "buildup happened at a mix of 2 and 3 stacks". The per-stack
    value comes from the refinement ``rank``.

    Returns 0 when the buff's element doesn't match the dealt element.

    Raises:
        CalcError: stacks requested for an engine without a modeled buff,
            or out of the buff's range.
    """
    if stacks == 0:
        return 0.0
    buff = engine.conditional_buff
    if buff is None:
        raise CalcError(
            f"Engine '{engine.name}' has no modeled conditional buff; "
            f"enter its passive manually as external buffs instead"
        )
    if isinstance(stacks, bool) or not isinstance(stacks, (int, float)) or not 0 <= stacks <= buff.max_stacks:
        raise CalcError(
            f"{buff.name} stacks must be a number 0..{buff.max_stacks}, "
            f"got {stacks!r}"
        )
    if buff.element is not None and buff.element != dealt_element:
        return 0.0
    return buff.per_stack(rank) * stacks


def _engine_passive_bonuses(
    engine: Engine, dealt_element: str, rank: int
) -> list[float]:
    """Always-on DMG% parts of the engine passive that match the element."""
    return [
        pd.value(rank) for pd in engine.passive_dmg
        if pd.element is None or pd.element == dealt_element
    ]


def _kit_contributions(
    agent: Agent, config: CalcConfig,
    agents: dict[str, Agent], disc_sets: dict[str, DiscSet],
    engines: dict[str, Engine],
) -> dict:
    """Aggregate every active conditional: the on-field agent's kit (core
    passive, mindscapes, additional ability) plus the supports' team
    buffs and squad set effects.

    Returns a dict with keys ``crit_rate``, ``crit_dmg``, ``res_shred``,
    ``flat_atk`` (floats) and ``dmg_bonus``, ``dmg_taken``, ``daze_bonus``
    (lists of bracket entries).

    Raises:
        CalcError: unmodeled/unknown buffs or mindscapes, stacks out of
            range, unknown support agents or squad sets, or a missing
            scaling input (e.g. Zhao's initial Max HP).
    """
    totals = {"crit_rate": 0.0, "crit_dmg": 0.0, "res_shred": 0.0,
              "flat_atk": 0.0, "atk_pct": 0.0,
              "dmg_bonus": [], "dmg_taken": [], "daze_bonus": [],
              "items": []}

    def record(source: str, kind: str, value: float) -> None:
        totals["items"].append(
            {"source": source, "kind": kind, "value": value}
        )

    def apply(buff, stacks: float, scaling_inputs: dict | None = None,
              owner: str = "") -> None:
        for effect in buff.effects:
            # Skill-tag-gated effects apply only to matching hits
            # (e.g. Catwalk buffs EX Specials only).
            if effect.skill_tag is not None and effect.skill_tag != config.skill_tag:
                continue
            if effect.scaling is not None:
                inputs = scaling_inputs or {}
                x = inputs.get(effect.scaling.input)
                if not isinstance(x, (int, float)) or isinstance(x, bool) or x <= 0:
                    raise CalcError(
                        f"{owner}{buff.name}: needs a positive "
                        f"'{effect.scaling.input}' value (the buff scales "
                        f"with the owner's stat)"
                    )
                value = effect.scaling.resolve(float(x))
            else:
                value = effect.value
            amount = value * stacks
            if effect.kind in ("dmg_bonus", "dmg_taken", "daze_bonus"):
                totals[effect.kind].append(amount)
            else:
                totals[effect.kind] += amount
            record(f"{owner}{buff.name}", effect.kind, amount)

    def check_stacks(stacks, max_stacks: int, what: str) -> None:
        if isinstance(stacks, bool) or not isinstance(stacks, int) or not 0 <= stacks <= max_stacks:
            raise CalcError(
                f"{what} stacks must be an integer 0..{max_stacks}, "
                f"got {stacks!r}"
            )

    if config.core_passive_active:
        if agent.core_passive is None:
            raise CalcError(
                f"Agent '{agent.name}' has no modeled core passive"
            )
        apply(agent.core_passive, 1, config.scaling_inputs)

    if config.additional_ability_stacks:
        ability = agent.additional_ability
        if ability is None:
            raise CalcError(
                f"Agent '{agent.name}' has no modeled additional ability"
            )
        check_stacks(config.additional_ability_stacks, ability.max_stacks,
                     f"Additional ability {ability.name}")
        apply(ability, config.additional_ability_stacks,
              config.scaling_inputs)

    for level_raw, stacks in config.mindscapes.items():
        level = int(level_raw)
        mindscape = agent.mindscapes.get(level)
        if mindscape is None:
            raise CalcError(
                f"Agent '{agent.name}' has no mindscape {level} in the data"
            )
        if mindscape.buff is None:
            raise CalcError(
                f"Mindscape {level} ({mindscape.name}) has no modeled "
                f"damage effect: {mindscape.note}"
            )
        check_stacks(stacks, mindscape.buff.max_stacks,
                     f"Mindscape {level} ({mindscape.name})")
        if stacks:
            apply(mindscape.buff, stacks, config.scaling_inputs)

    # --- Off-field supports: team buffs + squad set effects ---------------
    if len(config.supports) > 2:
        raise CalcError("A squad has at most 2 supports")
    seen_supports: set[str] = set()
    for support in config.supports:
        if support.agent_key not in agents:
            raise CalcError(
                f"Unknown support agent '{support.agent_key}'; "
                f"options: {sorted(agents)}"
            )
        if support.agent_key == config.agent_key or support.agent_key in seen_supports:
            raise CalcError(
                f"Support '{support.agent_key}' duplicates a squad member"
            )
        seen_supports.add(support.agent_key)
        member = agents[support.agent_key]
        buffs_by_name = {b.name: b for b in member.team_buffs}
        for name, stacks in support.buffs.items():
            buff = buffs_by_name.get(name)
            if buff is None:
                raise CalcError(
                    f"Support '{member.name}' has no team buff named "
                    f"'{name}'; options: {sorted(buffs_by_name)}"
                )
            check_stacks(stacks, buff.max_stacks,
                         f"{member.name}: {buff.name}")
            if stacks:
                apply(buff, stacks, support.scaling_inputs,
                      owner=f"{member.name}: ")
        if support.squad_set is not None:
            entry = disc_sets.get(support.squad_set)
            if entry is None or entry.squad_4pc is None:
                raise CalcError(
                    f"Set '{support.squad_set}' has no squad-facing "
                    f"4-piece effect"
                )
            check_stacks(support.squad_set_stacks, entry.squad_4pc.max_stacks,
                         f"{entry.name} squad 4pc")
            if support.squad_set_stacks:
                amount = entry.squad_4pc.value * support.squad_set_stacks
                if entry.squad_4pc.kind in ("dmg_bonus", "dmg_taken",
                                            "daze_bonus"):
                    totals[entry.squad_4pc.kind].append(amount)
                else:
                    totals[entry.squad_4pc.kind] += amount
                record(f"{member.name}: {entry.name} 4pc",
                       entry.squad_4pc.kind, amount)

        # Team-facing engine buffs (support signature engines)
        if support.engine_buffs:
            engine_key = (support.engine_key if support.engine_key is not None
                          else member.default_engine)
            if engine_key not in engines:
                raise CalcError(
                    f"Unknown support engine '{engine_key}'; "
                    f"options: {sorted(engines)}"
                )
            support_engine = engines[engine_key]
            rank = _resolve_engine_rank(support_engine, support.engine_rank)
            squad_by_name = {b.name: b for b in support_engine.squad_buffs}
            for name, stacks in support.engine_buffs.items():
                buff = squad_by_name.get(name)
                if buff is None:
                    raise CalcError(
                        f"Engine '{support_engine.name}' has no squad buff "
                        f"named '{name}'; options: {sorted(squad_by_name)}"
                    )
                check_stacks(stacks, buff.max_stacks,
                             f"{support_engine.name}: {buff.name}")
                if not stacks:
                    continue
                amount = buff.value(rank) * stacks
                if buff.kind in ("dmg_bonus", "dmg_taken", "daze_bonus"):
                    totals[buff.kind].append(amount)
                elif buff.kind in totals and buff.kind != "items":
                    totals[buff.kind] += amount
                else:
                    raise CalcError(
                        f"Engine '{support_engine.name}': squad buff kind "
                        f"'{buff.kind}' is not supported"
                    )
                record(f"{member.name}: {buff.name} (R{rank})",
                       buff.kind, amount)
    return totals


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

    if config.skill_tag is not None and config.skill_tag not in consts.skill_tags:
        raise CalcError(
            f"Unknown skill tag '{config.skill_tag}'; "
            f"options: {sorted(consts.skill_tags)}"
        )

    build = aggregate_build(
        agent, engine, config.discs, disc_data, consts,
        disc_sets=disc_sets, set_stacks=config.set_stacks,
    )

    rank = _resolve_engine_rank(engine, config.engine_rank)
    kit = _kit_contributions(agent, config, agents, disc_sets, engines)
    buff_items = kit["items"]

    # Team ATK% buffs join the SAME additive panel bracket as Combat ATK%
    # (each applies to the unbuffed panel — in-game validation #4,
    # 2026-07-03); team flat ATK (e.g. Zhao's +1,000) is added after.
    atk_total = (
        build.atk_pre_combat
        * (1.0 + build.combat_atk_pct + kit["atk_pct"])
        + kit["flat_atk"]
    )

    # --- Multiplier zones (formulas.py, plain numbers only) ---------------
    base = formulas.base_dmg(config.skill_multiplier, atk_total)

    crit_rate = build.crit_rate + config.external_crit_rate + kit["crit_rate"]
    crit_dmg = build.crit_dmg + config.external_crit_dmg + kit["crit_dmg"]
    crit_none = formulas.crit_mult_non_crit()
    crit_full = formulas.crit_mult_crit(crit_dmg)
    crit_avg = formulas.crit_mult_average(
        crit_rate, crit_dmg, consts.crit_rate_cap
    )

    bonus_entries = build.dmg_bonuses + list(config.external_dmg_bonuses)
    for passive_value in _engine_passive_bonuses(engine, agent.attribute, rank):
        bonus_entries.append(passive_value)
        buff_items.append({"source": f"{engine.name} passive (R{rank})",
                           "kind": "dmg_bonus", "value": passive_value})
    engine_buff = _engine_buff_bonus(
        engine, config.engine_buff_stacks, agent.attribute, rank
    )
    if engine_buff:
        bonus_entries.append(engine_buff)
        buff_items.append({
            "source": f"{engine.conditional_buff.name} (R{rank})",
            "kind": "dmg_bonus", "value": engine_buff,
        })
    bonus_entries.extend(kit["dmg_bonus"])
    # Skill-type-conditional set bonuses (e.g. Puffer Electro 4pc's Ultimate
    # DMG +20%) join the same additive DMG% bracket when the tag matches.
    tagged = set_tagged_dmg_bonuses(
        config.discs, disc_sets, config.skill_tag,
        valid_tags=frozenset(consts.skill_tags),
    )
    bonus_entries.extend(tagged.values())
    bonus = formulas.dmg_bonus_mult(bonus_entries)

    eff_def = formulas.effective_def(boss.base_def, build.pen_ratio, build.pen_flat)
    defense = formulas.def_mult(consts.level_coefficient(agent.level), eff_def)

    res = formulas.res_mult(
        boss.res_for(agent.attribute),
        res_ignore=config.external_res_shred + kit["res_shred"],
    )

    taken = formulas.dmg_taken_mult(
        list(config.external_dmg_taken) + kit["dmg_taken"]
    )

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
    stun_on = formulas.stun_mult(
        True, stun_base,
        list(config.external_daze_bonuses) + kit["daze_bonus"],
    )

    def dmg(crit_mult: float, stun: float) -> float:
        return formulas.total_dmg(base, crit_mult, bonus, defense, res, taken, stun)

    return CalcResults(
        non_crit=dmg(crit_none, stun_off),
        crit=dmg(crit_full, stun_off),
        average=dmg(crit_avg, stun_off),
        non_crit_stunned=dmg(crit_none, stun_on),
        crit_stunned=dmg(crit_full, stun_on),
        average_stunned=dmg(crit_avg, stun_on),
        atk_final=atk_total,
        crit_rate=crit_rate,
        crit_dmg=crit_dmg,
        base_dmg=base,
        dmg_bonus_mult=bonus,
        def_mult=defense,
        res_mult=res,
        dmg_taken_mult=taken,
        stun_mult=stun_on,
        buff_breakdown=tuple(buff_items),
    )


def calculate_anomaly(
    config: CalcConfig,
    *,
    consts: Constants | None = None,
    disc_data: DiscData | None = None,
    bosses: dict[str, Boss] | None = None,
    agents: dict[str, Agent] | None = None,
    engines: dict[str, Engine] | None = None,
    disc_sets: dict[str, DiscSet] | None = None,
    anomaly_data: AnomalyData | None = None,
) -> AnomalyResults:
    """Compute anomaly-proc or Disorder damage for one configuration.

    Plain anomaly (``config.disorder_replaced is None``): the agent's own
    attribute anomaly — per-proc and full-duration damage.

    Disorder (``config.disorder_replaced`` set): the burst dealt when the
    agent's anomaly replaces the given element's active anomaly. Damage is
    dealt as the REPLACED element (RES uses that element; the build's
    Attribute DMG% bonus only applies when the dealt element matches the
    agent's own attribute). ``config.skill_multiplier`` is ignored in both
    modes.

    ⚠️ Several underlying values are provisional pending in-game
    calibration — see DOCS/sources.md Phase 5.

    ``config.engine_rank`` scales the engine's modeled passive parts here
    too; agent kit conditionals (``core_passive_active`` / ``mindscapes``)
    are DIRECT-HIT ONLY for now and ignored in anomaly modes (their
    buildup-snapshot semantics are future work).

    Raises:
        CalcError: unknown agent/engine/boss, unsupported anomaly (e.g.
            Windswept until measured), or invalid disorder inputs.
    """
    consts = consts if consts is not None else load_constants()
    disc_data = disc_data if disc_data is not None else load_disc_data()
    bosses = bosses if bosses is not None else load_bosses()
    agents = agents if agents is not None else load_agents()
    engines = engines if engines is not None else load_engines()
    disc_sets = disc_sets if disc_sets is not None else load_disc_sets()
    anomaly_data = anomaly_data if anomaly_data is not None else load_anomalies()

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

    build = aggregate_build(
        agent, engine, config.discs, disc_data, consts,
        disc_sets=disc_sets, set_stacks=config.set_stacks,
    )

    rank = _resolve_engine_rank(engine, config.engine_rank)

    # --- Which anomaly's damage, and per-proc/burst multiplier ------------
    if config.disorder_replaced is None:
        anomaly = anomaly_data.anomalies[agent.attribute]
        try:
            anomaly.require_supported()
        except Exception as exc:
            raise CalcError(str(exc)) from None
        kind = "anomaly"
        name = anomaly.name
        dealt_element = agent.attribute
        per_proc_mult = anomaly.mult
        hits = anomaly.hits
    else:
        replaced_key = config.disorder_replaced.lower()
        if replaced_key not in anomaly_data.anomalies:
            raise CalcError(
                f"Unknown replaced anomaly element '{config.disorder_replaced}'"
            )
        if replaced_key == agent.attribute:
            raise CalcError(
                "Disorder needs a different element: the replaced anomaly "
                "matches the agent's own attribute"
            )
        replaced = anomaly_data.anomalies[replaced_key]
        try:
            replaced.require_supported()
        except Exception as exc:
            raise CalcError(str(exc)) from None
        rule = anomaly_data.disorder.get(replaced_key)
        if rule is None:
            raise CalcError(f"No Disorder rule for element '{replaced_key}'")
        kind = "disorder"
        name = f"Disorder ({replaced.name})"
        dealt_element = replaced_key
        if rule.mode == "procs":
            per_proc_mult = formulas.disorder_procs_mult(
                replaced.mult,
                config.disorder_remaining_seconds,
                replaced.interval,
                rule.extra_procs,
            )
        else:   # flat_decay
            per_proc_mult = formulas.disorder_decay_mult(
                replaced.mult,
                config.disorder_elapsed_seconds,
                replaced.duration,
                rule.min_fraction,
            )
        hits = 1

    # --- Attacker-side snapshot: buildup-weighted over segments ------------
    # (mechanic discovery #2: procs average attacker state over buildup)
    segments = list(config.buildup_segments)
    total_share = 0.0
    for seg in segments:
        if not 0 < seg.share <= 1:
            raise CalcError(
                f"Buildup segment share must be in (0, 1], got {seg.share!r}"
            )
        total_share += seg.share
    if total_share > 1 + 1e-9:
        raise CalcError(
            f"Buildup segment shares sum to {total_share:.3f} (> 1)"
        )
    if total_share < 1 - 1e-9:
        # Remainder of the buildup uses the config's top-level buff state.
        segments.append(BuildupSegment(
            share=1 - total_share,
            engine_buff_stacks=config.engine_buff_stacks,
            set_stacks=config.set_stacks,
            external_dmg_bonuses=list(config.external_dmg_bonuses),
            external_anomaly_proficiency=config.external_anomaly_proficiency,
        ))

    level_mult = consts.anomaly_level_multiplier(agent.level)

    weighted_product = 0.0
    weighted_atk = 0.0
    weighted_ap = 0.0
    weighted_bonus = 0.0
    for seg in segments:
        if seg.atk_override is not None:
            seg_atk = seg.atk_override
            seg_ap_base = 0.0
        else:
            seg_build = aggregate_build(
                agent, engine, config.discs, disc_data, consts,
                disc_sets=disc_sets, set_stacks=seg.set_stacks,
            )
            seg_atk = seg_build.atk
            seg_ap_base = seg_build.anomaly_proficiency
        seg_ap_total = (
            seg.anomaly_proficiency_override
            if seg.anomaly_proficiency_override is not None
            else seg_ap_base + seg.external_anomaly_proficiency
        )
        bonus_entries = list(seg.external_dmg_bonuses)
        if dealt_element == agent.attribute:
            bonus_entries = build.dmg_bonuses + bonus_entries
        bonus_entries.extend(
            _engine_passive_bonuses(engine, dealt_element, rank)
        )
        engine_buff = _engine_buff_bonus(
            engine, seg.engine_buff_stacks, dealt_element, rank
        )
        if engine_buff:
            bonus_entries.append(engine_buff)
        seg_bonus = formulas.dmg_bonus_mult(bonus_entries)

        weighted_product += seg.share * (
            seg_atk * formulas.ap_mult(seg_ap_total) * seg_bonus
        )
        weighted_atk += seg.share * seg_atk
        weighted_ap += seg.share * seg_ap_total
        weighted_bonus += seg.share * seg_bonus

    ap = formulas.ap_mult(weighted_ap)
    bonus = weighted_bonus

    eff_def = formulas.effective_def(boss.base_def, build.pen_ratio, build.pen_flat)
    defense = formulas.def_mult(consts.level_coefficient(agent.level), eff_def)
    res = formulas.res_mult(
        boss.res_for(dealt_element), res_ignore=config.external_res_shred
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

    def dmg(mult: float, stun: float) -> float:
        # weighted_product already contains ATK x AP x DMG% per segment —
        # the exact snapshot math; the breakdown fields report weighted
        # averages of each factor for display only.
        return mult * weighted_product * level_mult * defense * res * taken * stun

    return AnomalyResults(
        kind=kind,
        anomaly_name=name,
        element=dealt_element,
        hits=hits,
        per_proc=dmg(per_proc_mult, stun_off),
        per_proc_stunned=dmg(per_proc_mult, stun_on),
        full=dmg(per_proc_mult, stun_off) * hits,
        full_stunned=dmg(per_proc_mult, stun_on) * hits,
        atk_final=weighted_atk,
        anomaly_proficiency=weighted_ap,
        ap_mult=ap,
        anomaly_level_mult=level_mult,
        anomaly_mult=per_proc_mult,
        dmg_bonus_mult=bonus,
        def_mult=defense,
        res_mult=res,
        dmg_taken_mult=taken,
        stun_mult=stun_on,
    )
