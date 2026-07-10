"""UI-agnostic calculation API.

This is the single entry point layer every front end uses (Decision #4).
The CLI in ``main.py`` is one thin consumer; a future web UI calls this
module the same way. No I/O happens here beyond loading the JSON data files
(which can be injected for tests).

Entry points::

    calculate(config)         -> CalcResults      # direct hits (v1)
    calculate_anomaly(config) -> AnomalyResults   # anomaly / Disorder / Vortex
    calculate_sheer(config)   -> SheerResults     # Rupture / Sheer Force

``calculate_sheer`` computes Rupture agents' Sheer damage: base = Sheer
Force × MV, crits normally, dedicated Sheer DMG bracket, and **no DEF
zone** (Sheer damage ignores enemy DEF — see DOCS/rupture_plan.md).

``calculate_anomaly`` computes the agent's own attribute anomaly (optionally
MV-scaled: Velina's Ablooms), or — when ``config.disorder_replaced`` is set —
the Disorder triggered by replacing that element's anomaly (damage dealt as
the REPLACED element), or — when ``config.vortex_infused`` is set — the
Vortex triggered when that element's anomaly meets an active Windswept
(damage dealt as the INFUSED element).
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
        engine_squad_buffs: The ON-FIELD agent's OWN engine squad-buff
            stacks (buff name -> stacks). Squad buffs are squad-wide, so
            the wearer receives them too — e.g. Velina + Joyau Doré's
            +60 AP at 2 stacks. (Off-field supports' squad buffs come via
            ``SupportConfig.engine_buffs``.)
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
        disorder_elapsed_seconds: Time since the converted anomaly was
            applied — the replaced anomaly for Disorder, the infused one
            for Vortex (feeds the burst closed form's time term).
        vortex_infused: For Vortex calculations: the element of the
            non-wind anomaly INFUSED into an active Windswept (damage is
            dealt as this element). Mutually exclusive with
            ``disorder_replaced``.
        abloom: Vivian's ABLOOM — plain anomaly mode. Computes her
            Featherbloom Abloom automatically: a separate instance dealt as
            ``abloom_element`` (default the agent's attribute) equal to
            ``VIVIAN_ABLOOM_MV[element]/100 × (AP/10) × base_mult`` of the
            anomaly's DMG — i.e. AP-scaled (unlike Velina's fixed-MV Ablooms
            via ``anomaly_mult_override``). ×1.30 when her M2 is toggled
            (``mindscapes[2]``). Mutually exclusive with
            ``anomaly_mult_override``.
        abloom_element: The element of the anomaly the Abloom triggers on
            (``None`` = the agent's attribute, e.g. ether for her own
            Corruption). Its base mult and MV set the Abloom size; it is
            dealt as this element.
        anomaly_mult_override: ABSOLUTE anomaly multiplier for plain
            anomaly mode, REPLACING the element's base (``None`` = normal,
            e.g. wind Windswept 12.5). Velina's ABLOOMS are separate
            wind-anomaly instances at their OWN multiplier — Condensed
            cyclone 1.45, Sweeping cyclone 2.55, Ultimate 6.80 — where
            "X% of Wind Anomaly DMG" IS the anomaly multiplier, NOT a
            scale on 12.5 (confirmed in-game 2026-07-04: Ultimate Abloom =
            6.80/12.5 of the Windswept popup, to the digit). Her M6
            Windswept overlap instead scales the base (12.5 × up to 1.40
            = up to 17.5). Ablooms are separate popups from Windswept/
            Vortex — run one calc per instance, don't sum here.
        external_anomaly_buff: Entries of the separate Anomaly/Disorder
            "Buff Multiplier" bracket (fractions) — "Attribute Anomaly /
            Disorder / Windswept / Vortex DMG +X%" effects not already
            modeled (e.g. Wuthering Salon's +18% on Windswept trigger).
            Anomaly modes only; bracket placement PROVISIONAL.
        external_disorder_mult_add: Additive increase of the
            Disorder/Vortex burst base multiplier (fraction) — e.g.
            Velina's consumed Windbite (+1.50 at max core). Modeled kit
            sources (Yuzuha M6) add on top automatically.
        polarity_disorder: Yanagi's POLARITY DISORDER — Disorder mode only.
            Her downward thrust fires a Disorder on the enemy's EXISTING
            anomaly (including her own electric Shock, so
            ``disorder_replaced`` may equal her attribute) WITHOUT
            consuming it, dealing ``POLARITY_DISORDER_FRACTION`` (0.15) of
            a full Disorder per trigger — repeatable, her main DMG. Set
            ``disorder_replaced`` to the anomaly present on the enemy
            (e.g. ``"electric"`` for her own Shock, which also keeps
            Thunder Metal's Shocked ATK active). One tick = the burst
            result; sum ticks per rotation. Beyond the 15% base it adds an
            AP-scaled term (see ``polarity_special_level``).
        polarity_special_level: Yanagi's Special Attack skill level (1-16,
            default 12 = base max). Sets the Polarity Disorder AP-term
            coefficient ``(5% + level × 2.25%)`` — the second, AP-scaling
            part of her damage (GO's ``anom_flat_dmg``). ⚠️ PROVISIONAL.
        sheer_force_flat: Extra FLAT Sheer Force from external buffs
            (kit sources — Lucia's squad buff — add on top automatically).
            Sheer mode only.
        external_sheer_dmg: Entries of the dedicated Sheer DMG bracket
            (fractions) — "Sheer DMG +X%" effects not already modeled.
            Sheer mode only; bracket placement PROVISIONAL.
        unbuffed_share: Fraction of the buildup treated as UNBUFFED —
            contributed at the attacker's plain panel (kit/team buffs,
            engine conditional stacks, set 4pc stacks, external buffs and
            the anomaly-buff bracket all excluded; unconditional parts —
            disc stats, 2pc bonuses, always-on engine passives — stay).
            Models teammate buildup dilution / buff downtime as a
            pessimistic bound without per-teammate inputs (the UI shows
            the 0.0 vs 0.3 range). Applied on top of
            ``buildup_segments`` (their shares scale by ``1 − share``).
            Enemy-side zones are proc-time and stay full (discovery #2).
            Anomaly modes only.
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
    engine_squad_buffs: dict[str, int] = field(default_factory=dict)
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
    disorder_elapsed_seconds: float = 0.0
    vortex_infused: str | None = None
    anomaly_mult_override: float | None = None
    abloom: bool = False
    abloom_element: str | None = None
    external_anomaly_buff: list[float] = field(default_factory=list)
    external_disorder_mult_add: float = 0.0
    polarity_disorder: bool = False
    polarity_special_level: int = 12
    unbuffed_share: float = 0.0
    sheer_force_flat: float = 0.0
    external_sheer_dmg: list[float] = field(default_factory=list)


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

    kind: str                 # "anomaly", "disorder" or "vortex"
    anomaly_name: str         # e.g. "Assault", "Disorder (Burn)", "Vortex (Burn)"
    element: str              # element the damage is dealt as
    hits: int                 # procs over full duration (1 for bursts)
    per_proc: float
    per_proc_stunned: float
    full: float               # per_proc x hits (or the Disorder/Vortex burst)
    full_stunned: float
    # Breakdown
    atk_final: float
    anomaly_proficiency: float
    ap_mult: float
    anomaly_level_mult: float
    anomaly_mult: float       # the composed per-proc/burst multiplier used
    dmg_bonus_mult: float
    anomaly_buff_mult: float  # separate Anomaly/Disorder Buff Multiplier bracket
    def_mult: float
    res_mult: float
    dmg_taken_mult: float
    stun_mult: float
    # Every active conditional buff as {source, kind, value} — the front
    # ends render this as an itemized "Active buffs" summary.
    buff_breakdown: tuple = ()


@dataclass(frozen=True)
class SheerResults:
    """Output of :func:`calculate_sheer` (Rupture / Sheer damage).

    Same damage table as :class:`CalcResults` (Sheer damage crits and
    reacts to Stun like a direct hit) plus the Sheer Force breakdown.
    ``def_mult`` is always 1.0 — Sheer damage ignores enemy DEF entirely
    (datamine + Yixuan's core text); the field is kept for display
    symmetry with the other result types.
    """

    # Damage table
    non_crit: float
    crit: float
    average: float
    non_crit_stunned: float
    crit_stunned: float
    average_stunned: float
    # Sheer Force breakdown
    sheer_force: float
    sheer_force_atk_part: float   # ATK_final × atk_conversion
    sheer_force_hp_part: float    # HP_final × hp_conversion (Yixuan)
    sheer_force_flat_part: float  # config flat + kit (Lucia) flat SF
    atk_final: float
    hp_final: float
    # Breakdown (display / cross-checking)
    crit_rate: float          # uncapped total; capping happens in the formula
    crit_dmg: float
    base_dmg: float           # SkillMultiplier × Sheer Force
    dmg_bonus_mult: float
    sheer_dmg_mult: float     # dedicated Sheer DMG bracket
    def_mult: float           # always 1.0 — Sheer ignores DEF
    res_mult: float
    dmg_taken_mult: float
    stun_mult: float
    element: str
    # Every active conditional buff as {source, kind, value}.
    buff_breakdown: tuple = ()


#: Polarity Disorder (Yanagi) reduces the BASE Disorder (base + time decay)
#: to this fraction per trigger; the Disorder-mult additives (her +250%) and
#: the AP term apply at FULL on top. It does NOT consume the anomaly, so it
#: repeats — her main DMG. CALIBRATED in-game 2026-07-05 (within 0.3%).
POLARITY_DISORDER_FRACTION = 0.15

#: Vivian's Abloom (Dirge of Destiny) MV per element — % per 10 AP of the
#: original anomaly's DMG, at max core. Her Featherbloom on an anomalous
#: target adds a separate instance = (MV/100 × AP/10) × the anomaly's DMG,
#: dealt as that element. ⚠️ PROVISIONAL (datamine, uncalibrated).
VIVIAN_ABLOOM_MV = {
    "ether": 6.15, "electric": 3.2, "fire": 8.0,
    "physical": 0.75, "ice": 1.08, "wind": 0.32,
}
#: Vivian's M2 raises the Abloom's AP benefit to this fraction.
VIVIAN_ABLOOM_M2_FACTOR = 1.30


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


def _engine_buff_bonuses(
    engine: Engine, stacks: float, dealt_element: str, rank: int,
    mode: str = "direct", bracket: str = "dmg_bonus",
    skill_tag: str | None = None,
) -> list[dict]:
    """Contributions of the engine's conditional buff parts at ``stacks``.

    An engine models one stack counter that may drive SEVERAL buff parts
    (``Engine.conditional_buffs`` — e.g. Qingming Companion grants Ether
    DMG% and Ult/EX Sheer DMG% per stack); each applicable part yields one
    ``{"name", "value"}`` entry for the requested ``bracket``.

    ``stacks`` may be fractional: anomaly procs snapshot buffs as a
    buildup-weighted average (in-game finding, 2026-07-02), so e.g. 2.35
    expresses "buildup happened at a mix of 2 and 3 stacks". The per-stack
    value comes from the refinement ``rank``; a part with a lower stack cap
    than the entered count clamps to its own cap.

    A part is skipped when it doesn't apply: its ``bracket`` differs from
    the requested one, its ``modes`` gate excludes ``mode``, its
    ``skill_tags`` gate excludes the hit's ``skill_tag``, or its element
    doesn't match the dealt element. A part that EXPLICITLY lists a
    disorder/vortex mode skips the element check there (those bursts deal
    another element on the equipper's behalf — Joyau Doré); a plain
    element-gated part (Sharpened Stinger) stays element-checked in every
    mode, including sheer.

    Raises:
        CalcError: stacks requested for an engine without a modeled buff,
            or out of the buffs' range.
    """
    if stacks == 0:
        return []
    buffs = engine.conditional_buffs
    if not buffs:
        raise CalcError(
            f"Engine '{engine.name}' has no modeled conditional buff; "
            f"enter its passive manually as external buffs instead"
        )
    stack_cap = max(b.max_stacks for b in buffs)
    if isinstance(stacks, bool) or not isinstance(stacks, (int, float)) or not 0 <= stacks <= stack_cap:
        raise CalcError(
            f"{buffs[0].name} stacks must be a number 0..{stack_cap}, "
            f"got {stacks!r}"
        )
    entries: list[dict] = []
    for buff in buffs:
        if buff.bracket != bracket:
            continue
        if buff.modes is not None and mode not in buff.modes:
            continue
        if buff.skill_tags is not None and skill_tag not in buff.skill_tags:
            continue
        element_gate_waived = (
            buff.modes is not None and mode in ("disorder", "vortex")
        )
        if (buff.element is not None and not element_gate_waived
                and buff.element != dealt_element):
            continue
        value = buff.per_stack(rank) * min(stacks, buff.max_stacks)
        if value:
            entries.append({"name": buff.name, "value": value})
    return entries


def _engine_buff_bonus(
    engine: Engine, stacks: float, dealt_element: str, rank: int,
    mode: str = "direct", bracket: str = "dmg_bonus",
    skill_tag: str | None = None,
) -> float:
    """Summed contribution of the engine's conditional buff parts.

    Thin wrapper over :func:`_engine_buff_bonuses` for callers that only
    need the bracket total (the optimizer's constant-folding fast paths).
    """
    return sum(
        part["value"]
        for part in _engine_buff_bonuses(
            engine, stacks, dealt_element, rank,
            mode=mode, bracket=bracket, skill_tag=skill_tag,
        )
    )


def _engine_passive_bonuses(
    engine: Engine, dealt_element: str, rank: int
) -> list[float]:
    """Always-on DMG% parts of the engine passive that match the element."""
    return [
        pd.value(rank) for pd in engine.passive_dmg
        if pd.element is None or pd.element == dealt_element
    ]


#: Kit effect kinds collected as bracket-entry LISTS (the rest are sums).
_LIST_KINDS = ("dmg_bonus", "dmg_taken", "daze_bonus", "anomaly_buff",
               "sheer_dmg")


def _kit_contributions(
    agent: Agent, config: CalcConfig,
    agents: dict[str, Agent], disc_sets: dict[str, DiscSet],
    engines: dict[str, Engine],
    mode: str = "direct", dealt_element: str | None = None,
) -> dict:
    """Aggregate every active conditional: the on-field agent's kit (core
    passive, mindscapes, additional ability) plus the supports' team
    buffs and squad set effects.

    ``mode`` (direct/anomaly/disorder/vortex) and ``dealt_element`` drive
    the effect gates: an effect with ``modes`` applies only when ``mode``
    is listed; an effect with ``element`` applies only when the dealt
    element matches — checked in the direct/anomaly modes only, since
    Disorder/Vortex deal another element's damage by design.

    Returns a dict with keys ``crit_rate``, ``crit_dmg``, ``res_shred``,
    ``flat_atk``, ``atk_pct``, ``anomaly_proficiency``,
    ``disorder_mult_add`` (floats) and ``dmg_bonus``, ``dmg_taken``,
    ``daze_bonus``, ``anomaly_buff`` (lists of bracket entries).

    Raises:
        CalcError: unmodeled/unknown buffs or mindscapes, stacks out of
            range, unknown support agents or squad sets, or a missing
            scaling input (e.g. Zhao's initial Max HP).
    """
    totals = {"crit_rate": 0.0, "crit_dmg": 0.0, "res_shred": 0.0,
              "flat_atk": 0.0, "atk_pct": 0.0, "pen_ratio": 0.0,
              "anomaly_proficiency": 0.0, "disorder_mult_add": 0.0,
              "sheer_force": 0.0, "hp_pct": 0.0, "def_red": 0.0,
              "dmg_bonus": [], "dmg_taken": [], "daze_bonus": [],
              "anomaly_buff": [], "sheer_dmg": [],
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
            # Mode gate (anomaly_buff / disorder_mult_add carriers etc.).
            if effect.modes is not None and mode not in effect.modes:
                continue
            # Element gate. An effect that EXPLICITLY lists a
            # disorder/vortex mode skips it there (those bursts deal
            # another element on the owner's behalf); plain element-gated
            # effects stay element-checked in every mode.
            element_gate_waived = (
                effect.modes is not None and mode in ("disorder", "vortex")
            )
            if (effect.element is not None and not element_gate_waived
                    and effect.element != dealt_element):
                continue
            if effect.scaling is not None:
                inputs = scaling_inputs or {}
                x = inputs.get(effect.scaling.input)
                # Zero is a valid input when the effect has a flat base
                # (Piper M2 at 0 Power = +10%); otherwise it means the
                # input was forgotten and the buff would silently vanish.
                zero_ok = effect.scaling.base > 0
                if (not isinstance(x, (int, float)) or isinstance(x, bool)
                        or x < 0 or (x == 0 and not zero_ok)):
                    raise CalcError(
                        f"{owner}{buff.name}: needs a "
                        f"{'non-negative' if zero_ok else 'positive'} "
                        f"'{effect.scaling.input}' value (the buff scales "
                        f"with the owner's stat)"
                    )
                value = effect.scaling.resolve(float(x))
            else:
                value = effect.value
            amount = value * stacks
            if effect.kind in _LIST_KINDS:
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

    def apply_squad_buff(engine: Engine, rank: int, name: str, stacks: int,
                         owner_label: str) -> None:
        """Route one engine squad buff into the totals.

        Squad buffs are squad-WIDE, so this is used both for off-field
        supports' engines AND the on-field agent's OWN engine (the wearer
        is a squad member too — e.g. Velina + Joyau Doré's +60 AP).
        """
        squad_by_name = {b.name: b for b in engine.squad_buffs}
        buff = squad_by_name.get(name)
        if buff is None:
            raise CalcError(
                f"Engine '{engine.name}' has no squad buff named '{name}'; "
                f"options: {sorted(squad_by_name)}"
            )
        check_stacks(stacks, buff.max_stacks, f"{engine.name}: {buff.name}")
        if not stacks:
            return
        amount = buff.value(rank) * stacks
        if buff.kind in _LIST_KINDS:
            totals[buff.kind].append(amount)
        elif buff.kind in totals and buff.kind != "items":
            totals[buff.kind] += amount
        else:
            raise CalcError(
                f"Engine '{engine.name}': squad buff kind '{buff.kind}' "
                f"is not supported"
            )
        record(f"{owner_label}: {buff.name} (R{rank})", buff.kind, amount)

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

    # --- On-field agent's equipped-set AUTO 4pc DMG% (effectively-always-on
    # 4pc parts that ramp up in a rotation, e.g. Wuthering Salon's +18% on
    # Windswept trigger). Auto-applied (no toggle) via the kit dmg_bonus
    # path so the anomaly range's unbuffed floor covers the ramp / uptime.
    auto_pieces: dict[str, int] = {}
    for disc in config.discs:
        if disc.disc_set:
            auto_pieces[disc.disc_set] = auto_pieces.get(disc.disc_set, 0) + 1
    for set_key, count in auto_pieces.items():
        entry = disc_sets.get(set_key)
        if entry is None or count < 4 or not entry.auto_4pc_dmg:
            continue
        # Some auto 4pc DMG parts need a TEAMMATE to trigger (Phaethon's
        # Melody: "when a teammate — not the equipper — uses an EX
        # Special"). Solo (no supports) it can't fire.
        if entry.auto_4pc_dmg_needs_teammate and not config.supports:
            continue
        totals["dmg_bonus"].append(entry.auto_4pc_dmg)
        record(f"{entry.name} 4pc (auto)", "dmg_bonus", entry.auto_4pc_dmg)

    # --- On-field agent's OWN engine squad buffs (the wearer is a squad
    # member too, so they receive their own engine's squad buffs — e.g.
    # Velina + Joyau Doré's +60 AP at 2 stacks). Off-field supports' squad
    # buffs are handled below.
    if config.engine_squad_buffs:
        own_engine_key = (config.engine_key if config.engine_key is not None
                          else agent.default_engine)
        if own_engine_key not in engines:
            raise CalcError(
                f"Unknown engine '{own_engine_key}'; options: {sorted(engines)}"
            )
        own_engine = engines[own_engine_key]
        own_rank = _resolve_engine_rank(own_engine, config.engine_rank)
        for name, stacks in config.engine_squad_buffs.items():
            apply_squad_buff(own_engine, own_rank, name, stacks,
                             f"{agent.name} (self)")

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
                if entry.squad_4pc.kind in _LIST_KINDS:
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
            for name, stacks in support.engine_buffs.items():
                apply_squad_buff(support_engine, rank, name, stacks,
                                 member.name)
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
    kit = _kit_contributions(
        agent, config, agents, disc_sets, engines,
        mode="direct", dealt_element=agent.attribute,
    )
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
    engine_crit = engine.passive_crit(rank)
    if engine_crit:
        crit_rate += engine_crit
        buff_items.append({"source": f"{engine.name} passive CRIT (R{rank})",
                           "kind": "crit_rate", "value": engine_crit})
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
    for part in _engine_buff_bonuses(
        engine, config.engine_buff_stacks, agent.attribute, rank,
        mode="direct", skill_tag=config.skill_tag,
    ):
        bonus_entries.append(part["value"])
        buff_items.append({
            "source": f"{part['name']} (R{rank})",
            "kind": "dmg_bonus", "value": part["value"],
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

    eff_def = formulas.effective_def(
        boss.base_def, build.pen_ratio + kit["pen_ratio"], build.pen_flat,
        def_red=kit["def_red"])
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
    """Compute anomaly-proc, Disorder or Vortex damage for one configuration.

    Plain anomaly (neither ``disorder_replaced`` nor ``vortex_infused``):
    the agent's own attribute anomaly — per-proc and full-duration damage,
    scaled by ``anomaly_mv_mult`` (Velina's Ablooms / M6 overlap).

    Disorder (``config.disorder_replaced`` set): the burst dealt when the
    agent's anomaly replaces the given element's active anomaly. Damage is
    dealt as the REPLACED element. Closed form: ``base + additive buffs +
    time_mult × max(0, window − elapsed)``.

    Vortex (``config.vortex_infused`` set): the burst dealt when a second
    (non-wind) anomaly meets an active Windswept — replaces Disorder, no
    daze. Damage is dealt as the INFUSED element, same closed form with the
    infused element's Vortex rule.

    In every mode the build's Attribute DMG% only applies when the dealt
    element matches the agent's own attribute, and
    ``config.skill_multiplier`` is ignored.

    Kit conditionals (core passive / mindscapes / additional ability) and
    the supports' team buffs ARE applied here (Phase 5e), under the adopted
    concession that buff state is held constant over the whole buildup —
    describe varying engine-buff/external states with
    ``buildup_segments``. Effects gated by ``modes``/``element`` apply per
    the calculation mode (e.g. Velina's Vortex-only RES ignore).

    ⚠️ Several underlying values are provisional pending in-game
    calibration — see DOCS/sources.md Phase 5.

    Raises:
        CalcError: unknown agent/engine/boss, unsupported anomaly, both
            disorder and vortex requested, or invalid mode inputs.
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
    if config.disorder_replaced is not None and config.vortex_infused is not None:
        raise CalcError(
            "'disorder_replaced' and 'vortex_infused' are mutually "
            "exclusive - Vortex replaces Disorder while Windswept is active"
        )
    if config.abloom:
        if config.disorder_replaced is not None or config.vortex_infused is not None:
            raise CalcError(
                "'abloom' is a plain-anomaly instance - not compatible with "
                "disorder/vortex mode"
            )
        if config.anomaly_mult_override is not None:
            raise CalcError(
                "'abloom' auto-computes the multiplier - don't also set "
                "anomaly_mult_override (that's Velina's fixed-MV Ablooms)"
            )
    # Additive burst-multiplier buffs (kit sources join below, once the
    # calculation mode is known and the kit is aggregated).
    extra_burst = config.external_disorder_mult_add
    if extra_burst < 0:
        raise CalcError("external_disorder_mult_add must be >= 0")

    if config.vortex_infused is not None:
        infused_key = config.vortex_infused.lower()
        rule = anomaly_data.vortex.get(infused_key)
        if rule is None:
            raise CalcError(
                f"No Vortex rule for infused element "
                f"'{config.vortex_infused}'; options: "
                f"{sorted(anomaly_data.vortex)} (wind itself is never the "
                f"infusion)"
            )
        infused = anomaly_data.anomalies[infused_key]
        kind = "vortex"
        name = f"Vortex ({infused.name})"
        dealt_element = infused_key
        hits = 1
    elif config.disorder_replaced is not None:
        replaced_key = config.disorder_replaced.lower()
        if replaced_key not in anomaly_data.anomalies:
            raise CalcError(
                f"Unknown replaced anomaly element '{config.disorder_replaced}'"
            )
        # Normal Disorder needs a DIFFERENT element (you can't Disorder an
        # anomaly with the same element). POLARITY DISORDER (Yanagi) is the
        # exception: it fires on the enemy's EXISTING anomaly — including
        # her own Shock (same element) — without replacing it, so it may be
        # same-element.
        if replaced_key == agent.attribute and not config.polarity_disorder:
            raise CalcError(
                "Disorder needs a different element: the replaced anomaly "
                "matches the agent's own attribute (use polarity_disorder "
                "for Yanagi's same-element Polarity Disorder)"
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
        name = (f"Polarity Disorder ({replaced.name})"
                if config.polarity_disorder else f"Disorder ({replaced.name})")
        dealt_element = replaced_key
        hits = 1
    elif config.abloom:
        # Vivian's Abloom: AP-scaled instance of the anomaly on the target.
        # The multiplier depends on AP, so it is finalised after the
        # buildup-weighted AP is known (below); here we set up the element.
        abloom_key = (config.abloom_element or agent.attribute).lower()
        if abloom_key not in VIVIAN_ABLOOM_MV:
            raise CalcError(
                f"No Abloom MV for element '{abloom_key}'; options: "
                f"{sorted(VIVIAN_ABLOOM_MV)}"
            )
        anomaly = anomaly_data.anomalies[abloom_key]
        try:
            anomaly.require_supported()
        except Exception as exc:
            raise CalcError(str(exc)) from None
        kind = "anomaly"
        name = f"Abloom ({anomaly.name})"
        dealt_element = abloom_key
        per_proc_mult = None            # set after weighted_ap
        hits = 1                        # one Abloom instance
    else:
        anomaly = anomaly_data.anomalies[agent.attribute]
        try:
            anomaly.require_supported()
        except Exception as exc:
            raise CalcError(str(exc)) from None
        override = config.anomaly_mult_override
        if override is not None and (
                isinstance(override, bool)
                or not isinstance(override, (int, float)) or override <= 0):
            raise CalcError(
                "anomaly_mult_override must be a positive multiplier "
                "(Velina Ablooms: Condensed 1.45 / Sweeping 2.55 / "
                "Ultimate 6.80; None = the element's normal proc)"
            )
        kind = "anomaly"
        name = anomaly.name
        if override is not None:
            name += f" (Abloom / mult x{override:g})"
        dealt_element = agent.attribute
        per_proc_mult = override if override is not None else anomaly.mult
        hits = anomaly.hits

    # --- Kit conditionals + supports (constant over the buildup) ----------
    kit = _kit_contributions(
        agent, config, agents, disc_sets, engines,
        mode=kind, dealt_element=dealt_element,
    )
    if kind == "vortex":
        per_proc_mult = formulas.burst_conversion_mult(
            rule.mult, rule.time_mult, rule.window,
            config.disorder_elapsed_seconds,
            extra_mult=extra_burst + kit["disorder_mult_add"],
        )
    elif kind == "disorder":
        disorder_additive = extra_burst + kit["disorder_mult_add"]
        if config.polarity_disorder:
            # Polarity Disorder (Yanagi): reduce ONLY the base Disorder
            # (base + time decay) to 15%; the Disorder-mult additives
            # (Yanagi's core +250%, etc.) apply at FULL, NOT reduced. Plus
            # the AP term (added after weighted_ap). CALIBRATED in-game
            # 2026-07-05 (controlled solo, elapsed 0-3s all within 0.3% —
            # see DOCS/sources.md): 0.15 x (4.5 + 1.25 x (10 - t)) + 2.5
            # + coeff x AP/100 reproduced the whole decay cycle exactly.
            base_disorder = formulas.burst_conversion_mult(
                rule.base, rule.time_mult, rule.window,
                config.disorder_elapsed_seconds, extra_mult=0.0,
            )
            per_proc_mult = (POLARITY_DISORDER_FRACTION * base_disorder
                             + disorder_additive)
        else:
            per_proc_mult = formulas.burst_conversion_mult(
                rule.base, rule.time_mult, rule.window,
                config.disorder_elapsed_seconds, extra_mult=disorder_additive,
            )

    # --- Attacker-side snapshot: buildup-weighted over segments ------------
    # (mechanic discovery #2: procs average attacker state over buildup)
    if isinstance(config.unbuffed_share, bool) or not isinstance(
            config.unbuffed_share, (int, float)) or not 0 <= config.unbuffed_share < 1:
        raise CalcError(
            f"unbuffed_share must be a fraction in [0, 1), "
            f"got {config.unbuffed_share!r}"
        )
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

    # Separate Anomaly/Disorder Buff Multiplier bracket (PROVISIONAL):
    # kit sources (Yuzuha's ability, Velina's kit), the engine's
    # anomaly-bracket conditional (Joyau Doré — uses the TOP-LEVEL stack
    # count, treated as constant over the buffed buildup), and manual
    # entries. Snapshots with the attacker state, so it joins the
    # per-segment product (the unbuffed slice runs without it).
    abuff_entries = list(config.external_anomaly_buff) + kit["anomaly_buff"]
    for part in _engine_buff_bonuses(
        engine, config.engine_buff_stacks, dealt_element, rank,
        mode=kind, bracket="anomaly_buff",
    ):
        abuff_entries.append(part["value"])
        kit["items"].append({
            "source": f"{part['name']} (R{rank})",
            "kind": "anomaly_buff", "value": part["value"],
        })
    abuff = formulas.anomaly_buff_mult(abuff_entries)

    # ``unbuffed_share`` of the buildup runs at the plain panel (teammate
    # dilution / buff downtime); the described segments share the rest.
    buffed_scale = 1.0 - config.unbuffed_share

    weighted_product = 0.0
    weighted_atk = 0.0
    weighted_ap = 0.0
    weighted_bonus = 0.0
    for seg in segments:
        if seg.atk_override is not None:
            # Teammate segment: their panel ATK / AP replace the build's.
            seg_atk = seg.atk_override
            seg_ap_base = 0.0
        else:
            seg_build = aggregate_build(
                agent, engine, config.discs, disc_data, consts,
                disc_sets=disc_sets, set_stacks=seg.set_stacks,
            )
            # Kit/team ATK buffs join the panel bracket / flat bucket the
            # same way as in the direct-hit path (validation #4).
            seg_atk = (
                seg_build.atk_pre_combat
                * (1.0 + seg_build.combat_atk_pct + kit["atk_pct"])
                + kit["flat_atk"]
            )
            seg_ap_base = (
                seg_build.anomaly_proficiency
                + engine.passive_ap(rank)          # e.g. Joyau Doré +70..110
                + kit["anomaly_proficiency"]       # squad AP buffs
            )
        seg_ap_total = (
            seg.anomaly_proficiency_override
            if seg.anomaly_proficiency_override is not None
            else seg_ap_base + seg.external_anomaly_proficiency
        )
        bonus_entries = list(seg.external_dmg_bonuses) + kit["dmg_bonus"]
        if dealt_element == agent.attribute:
            bonus_entries = build.dmg_bonuses + bonus_entries
        bonus_entries.extend(
            _engine_passive_bonuses(engine, dealt_element, rank)
        )
        for part in _engine_buff_bonuses(
            engine, seg.engine_buff_stacks, dealt_element, rank, mode=kind
        ):
            bonus_entries.append(part["value"])
        seg_bonus = formulas.dmg_bonus_mult(bonus_entries)

        w = seg.share * buffed_scale
        weighted_product += w * (
            seg_atk * formulas.ap_mult(seg_ap_total) * seg_bonus * abuff
        )
        weighted_atk += w * seg_atk
        weighted_ap += w * seg_ap_total
        weighted_bonus += w * seg_bonus

    if config.unbuffed_share > 0:
        # The diluted slice: plain panel — no kit/team buffs, no engine
        # conditional, no set 4pc stacks, no externals, buff bracket ×1.
        # Unconditional parts (disc stats, 2pc bonuses, always-on engine
        # passives incl. Joyau Doré's AP) remain.
        bare_build = aggregate_build(
            agent, engine, config.discs, disc_data, consts,
            disc_sets=disc_sets, set_stacks={},
        )
        bare_atk = bare_build.atk
        bare_ap = bare_build.anomaly_proficiency + engine.passive_ap(rank)
        bare_entries = (
            list(bare_build.dmg_bonuses)
            if dealt_element == agent.attribute else []
        )
        bare_entries.extend(_engine_passive_bonuses(engine, dealt_element, rank))
        bare_bonus = formulas.dmg_bonus_mult(bare_entries)
        w = config.unbuffed_share
        weighted_product += w * (
            bare_atk * formulas.ap_mult(bare_ap) * bare_bonus
        )
        weighted_atk += w * bare_atk
        weighted_ap += w * bare_ap
        weighted_bonus += w * bare_bonus

    ap = formulas.ap_mult(weighted_ap)
    bonus = weighted_bonus

    if config.abloom:
        # Vivian's Abloom multiplier = base_mult × (MV/100) × (AP/10),
        # ×1.30 with her M2 (mindscapes[2]). AP-scaled, so finalised here
        # with the buildup-weighted AP. ⚠️ PROVISIONAL (uncalibrated).
        abloom_key = (config.abloom_element or agent.attribute).lower()
        mv = VIVIAN_ABLOOM_MV[abloom_key]
        m2 = VIVIAN_ABLOOM_M2_FACTOR if config.mindscapes.get(2) else 1.0
        per_proc_mult = (anomaly_data.anomalies[abloom_key].mult
                         * (mv / 100.0) * (weighted_ap / 10.0) * m2)

    if config.polarity_disorder:
        # Polarity Disorder adds a second, AP-scaled term on top of the
        # 15% base (GO: anom_flat_dmg = (5% + special-skill-Lv × 2.25%) ×
        # AnomProf). Modeled as an addition to the burst multiplier of
        # coeff × (AP/100) — combined with the standard AP term this gives
        # the AP² scaling that makes AP doubly valuable for Yanagi.
        # ⚠️ PROVISIONAL: matches the user's in-game LOWER bound at Lv12
        # (2026-07-05); the median/peak imply a higher effective
        # coefficient — needs a controlled measurement to pin exactly.
        coeff = 0.05 + config.polarity_special_level * 0.0225
        per_proc_mult = per_proc_mult + coeff * (weighted_ap / 100.0)

    eff_def = formulas.effective_def(
        boss.base_def, build.pen_ratio + kit["pen_ratio"], build.pen_flat,
        def_red=kit["def_red"])
    defense = formulas.def_mult(consts.level_coefficient(agent.level), eff_def)
    res = formulas.res_mult(
        boss.res_for(dealt_element),
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

    def dmg(mult: float, stun: float) -> float:
        # weighted_product already contains ATK x AP x DMG% x buff-bracket
        # per segment — the exact snapshot math; the breakdown fields
        # report weighted averages / full-state values for display only.
        return (mult * weighted_product * level_mult
                * defense * res * taken * stun)

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
        anomaly_buff_mult=abuff,
        def_mult=defense,
        res_mult=res,
        dmg_taken_mult=taken,
        stun_mult=stun_on,
        buff_breakdown=tuple(kit["items"]),
    )


def calculate_sheer(
    config: CalcConfig,
    *,
    consts: Constants | None = None,
    disc_data: DiscData | None = None,
    bosses: dict[str, Boss] | None = None,
    agents: dict[str, Agent] | None = None,
    engines: dict[str, Engine] | None = None,
    disc_sets: dict[str, DiscSet] | None = None,
) -> SheerResults:
    """Run the Sheer (Rupture) damage calculation for one configuration.

    Sheer damage is a direct-hit-shaped hit with three differences
    (zenless-optimizer datamine 2026-07-10 + Yixuan's core text — see
    DOCS/rupture_plan.md):

    1. ``BaseDMG = SkillMultiplier × Sheer Force`` where
       ``SF = ATK_final × 0.30 + HP_final × hp_conversion + flat SF``
       (both post-buff panels — team ATK%/HP% feed the conversions).
    2. **No DEF zone**: Sheer damage ignores enemy DEF entirely, so
       PEN Ratio / flat PEN / DEF reduction do nothing.
    3. A dedicated **Sheer DMG bracket** (``1 + Σ Sheer-DMG%``) multiplies
       on top of the ordinary DMG% bracket.

    Everything else matches :func:`calculate`: Sheer damage crits, is dealt
    as the agent's attribute (RES/Attribute-DMG% apply), reacts to Stun,
    and takes the same kit/support/engine/set conditionals under
    ``mode="sheer"`` gates.

    ⚠️ All Sheer coefficients are PROVISIONAL until an in-game popup
    calibration (DOCS/rupture_plan.md §5).

    Raises:
        CalcError: unknown agent/engine/boss, a non-Rupture agent, or
            invalid inputs.
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
    if agent.specialty != "rupture":
        raise CalcError(
            f"Agent '{agent.name}' is not a Rupture agent — Sheer damage "
            f"needs the rupture specialty (its skills scale off Sheer "
            f"Force). Use the direct mode instead."
        )

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
    if config.sheer_force_flat < 0:
        raise CalcError("sheer_force_flat must be >= 0")

    build = aggregate_build(
        agent, engine, config.discs, disc_data, consts,
        disc_sets=disc_sets, set_stacks=config.set_stacks,
    )

    rank = _resolve_engine_rank(engine, config.engine_rank)
    kit = _kit_contributions(
        agent, config, agents, disc_sets, engines,
        mode="sheer", dealt_element=agent.attribute,
    )
    buff_items = kit["items"]

    # ATK: same bracket structure as the direct-hit path (validation #4).
    atk_total = (
        build.atk_pre_combat
        * (1.0 + build.combat_atk_pct + kit["atk_pct"])
        + kit["flat_atk"]
    )
    # HP: team HP% buffs (Lucia, Dreamlit Hearth) multiply the finished
    # panel — the combat HP bracket (datamine: final hp = initial ×
    # (1 + combat hp_)).
    hp_total = build.hp * (1.0 + kit["hp_pct"])

    # --- Sheer Force (the Rupture base stat) ------------------------------
    flat_sf = config.sheer_force_flat + kit["sheer_force"]
    sf_atk_part = atk_total * consts.sheer_force_atk_conversion
    sf_hp_part = hp_total * agent.sheer_force_hp_conversion
    sf = sf_atk_part + sf_hp_part + flat_sf

    base = formulas.base_dmg(config.skill_multiplier, sf)

    crit_rate = build.crit_rate + config.external_crit_rate + kit["crit_rate"]
    crit_dmg = build.crit_dmg + config.external_crit_dmg + kit["crit_dmg"]
    engine_crit = engine.passive_crit(rank)
    if engine_crit:
        crit_rate += engine_crit
        buff_items.append({"source": f"{engine.name} passive CRIT (R{rank})",
                           "kind": "crit_rate", "value": engine_crit})
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
    for part in _engine_buff_bonuses(
        engine, config.engine_buff_stacks, agent.attribute, rank,
        mode="sheer", skill_tag=config.skill_tag,
    ):
        bonus_entries.append(part["value"])
        buff_items.append({
            "source": f"{part['name']} (R{rank})",
            "kind": "dmg_bonus", "value": part["value"],
        })
    bonus_entries.extend(kit["dmg_bonus"])
    tagged = set_tagged_dmg_bonuses(
        config.discs, disc_sets, config.skill_tag,
        valid_tags=frozenset(consts.skill_tags),
    )
    bonus_entries.extend(tagged.values())
    bonus = formulas.dmg_bonus_mult(bonus_entries)

    # Dedicated Sheer DMG bracket (build sets + kit + engine parts +
    # manual entries). CALIBRATED multiplicatively separate from DMG%
    # in-game 2026-07-10 (Rupture calibration #2: Yunkui Tales +10%).
    sheer_entries = (build.sheer_dmg_bonuses
                     + list(config.external_sheer_dmg) + kit["sheer_dmg"])
    for part in _engine_buff_bonuses(
        engine, config.engine_buff_stacks, agent.attribute, rank,
        mode="sheer", bracket="sheer_dmg", skill_tag=config.skill_tag,
    ):
        sheer_entries.append(part["value"])
        buff_items.append({
            "source": f"{part['name']} (R{rank})",
            "kind": "sheer_dmg", "value": part["value"],
        })
    sheer_mult = formulas.sheer_dmg_bonus_mult(sheer_entries)

    # NO DEF zone: Sheer damage ignores enemy DEF entirely (datamine:
    # the sheerDmg formula never multiplies def_mult; Yixuan's core text:
    # "ignoring enemy DEF"). PEN Ratio / flat PEN / DEF shred do nothing.
    defense = 1.0

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
        return (formulas.total_dmg(base, crit_mult, bonus, defense, res,
                                   taken, stun)
                * sheer_mult)

    return SheerResults(
        non_crit=dmg(crit_none, stun_off),
        crit=dmg(crit_full, stun_off),
        average=dmg(crit_avg, stun_off),
        non_crit_stunned=dmg(crit_none, stun_on),
        crit_stunned=dmg(crit_full, stun_on),
        average_stunned=dmg(crit_avg, stun_on),
        sheer_force=sf,
        sheer_force_atk_part=sf_atk_part,
        sheer_force_hp_part=sf_hp_part,
        sheer_force_flat_part=flat_sf,
        atk_final=atk_total,
        hp_final=hp_total,
        crit_rate=crit_rate,
        crit_dmg=crit_dmg,
        base_dmg=base,
        dmg_bonus_mult=bonus,
        sheer_dmg_mult=sheer_mult,
        def_mult=defense,
        res_mult=res,
        dmg_taken_mult=taken,
        stun_mult=stun_on,
        element=agent.attribute,
        buff_breakdown=tuple(buff_items),
    )
