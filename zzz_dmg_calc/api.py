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
from .agent import (
    LUMIFLUX, PERCENT_SCALING_INPUT_MAX, PERCENT_SCALING_INPUTS,
    TOTAL_CRIT_DMG_INPUT, Agent, aggregate_build, load_agents,
)
from .anomalies import AnomalyData, load_anomalies
from .constants import Constants, load_constants
from .discs import (
    Disc, DiscData, DiscSet, load_disc_data, load_disc_sets,
    set_tagged_dmg_bonuses,
)
from .enemies import ELEMENTS, Boss, load_bosses
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
class LuminizeConfig:
    """LUMINIZE MODE (Lumiflux, 3.1) — Remielle's own damage.

    A Luminize is an Attribute Anomaly instance fired from a stored
    Voidflare. It scales off the **stored anomaly's** Effect Strength —
    the ATK, DMG% and PEN of the TEAMMATE who triggered it — while its
    multiplier and its Anomaly DMG Bonus come from Remielle. So the
    calculation runs with that teammate as ``CalcConfig.agent_key``, and
    this block supplies Remielle's side.

    Set it on :attr:`CalcConfig.luminize` and the anomaly pipeline:

    1. replaces the anomaly multiplier with
       ``skill MV × (1 + 0.002 × anomaly_proficiency)``, ×1.12 at ``m4``;
    2. **hands the Anomaly-DMG bracket to Remielle** — every attacker-side
       ``anomaly_buff`` contribution (their kit, their engine, manual
       entries) is dropped and replaced by :attr:`anomaly_dmg_bonuses`,
       because only *general* Anomaly DMG bonuses reach a Luminize and
       they come from her (Lin-0: Vivian's Corruption bonus would not
       apply, Yuzuha's general one would);
    3. reports ``hits = voidflares`` so ``full`` is the whole volley.

    Refringe still applies once — the stored Voidflare holds the already
    Refringed Effect Strength — so keep Remielle in ``supports`` with her
    Refringe buffs ticked.

    Attributes:
        skill: Key into the agent's ``luminize.skills`` table
            (``ultimate`` / ``assist`` / ``fleeting_grace`` /
            ``rainbows_end``).
        anomaly_proficiency: REMIELLE's panel AP — drives the multiplier.
        provider_key: Roster key of the Lumiflux agent (default
            ``"remielle"``); its ``luminize`` block supplies the table.
        m4: Her M4 is unlocked (×1.12 on the Luminize multiplier).
        voidflares: Stored Voidflares consumed/reused by the skill (1-3);
            ``full`` = one instance × this. M6 doubles the count on her
            enhanced Basics — pass 6 for that. **Assumes every one was
            stored by ``CalcConfig.agent_key``**: a Voidflare holds the
            Effect Strength *and element* of the anomaly that made it, so
            a volley whose Voidflares came from different teammates is
            several different instances. Model that as one run per source
            agent with ``voidflares=1``, summed.
        anomaly_dmg_bonuses: REMIELLE's own general Anomaly DMG bonuses as
            fractions (Ode of Resurrected Wings +20%, Feathered Fate 4pc
            +15%). These are the ONLY entries in the Anomaly-DMG bracket
            for this instance.
        res_ignore: Her M1 (Naive Oath) — Luminize ignores 50% of the
            target's All-Attribute RES. Added on top of the config's
            ``external_res_shred``.
    """

    skill: str
    anomaly_proficiency: float
    provider_key: str = "remielle"
    m4: bool = False
    voidflares: int = 1
    anomaly_dmg_bonuses: list[float] = field(default_factory=list)
    res_ignore: bool = False


@dataclass
class CalcConfig:
    """Everything the calculation needs, gathered by the front end.

    Attributes:
        agent_key: Key into the preset agent list (e.g. ``"ellen"``).
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
        counts_as_aftershock: The hit ALSO counts as Aftershock DMG on
            top of its own ``skill_tag`` (dual damage type — datamine
            ``damageType2``): Anby S's Chain Attack / Ultimate (her
            ability treats them as Aftershocks) and Trigger's Harmonizing
            Shots (Basic + Aftershock). Both the hit's own tag AND
            ``"aftershock"``-gated bonuses then apply — e.g. Anby S's
            Ultimate takes Ultimate DMG% and Aftershock DMG% together.
            Redundant (harmless) when ``skill_tag`` is ``"aftershock"``.
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
        potentials: Potential Awakening level -> active stacks, same shape
            as ``mindscapes``. A potential's levels REPLACE each other
            (the user owns exactly one level), so AT MOST ONE level may be
            active — more than one is rejected. Per-stack effects use the
            same stack count as the kit buff they augment (e.g. Ellen's
            potential rides her Rising Storm stacks); ``at_max_stacks``
            effects fire only at full stacks.
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
        own_team_buffs: The ON-FIELD agent's OWN ``team_buffs`` stacks
            (buff name -> stacks). Squad-wide buffs and enemy debuffs
            cover their owner too (Norma's En-Nah +20% DMG / M1 RES
            shred, Zhao's Ether Veil flat ATK), and a main agent cannot
            sit in ``supports`` — this field applies the self share.
            Whether a specific buff includes its owner is the user's
            responsibility (each buff's note says; e.g. Zhao's M2 ATK%
            is "OTHER squad members" — don't tick it here). Scaled
            effects read ``scaling_inputs``.
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
        luminize: LUMINIZE MODE (:class:`LuminizeConfig`) — Remielle's
            own damage, computed with the TEAMMATE who triggered the
            stored anomaly as ``agent_key``. Anomaly modes only;
            mutually exclusive with ``anomaly_mult_override`` and
            ``abloom`` (it sets the multiplier itself).
        mutated_element: LUMIFLUX ATTRIBUTE MUTATION (Remielle): the
            element her Lumiflux damage is TREATED as — official rule:
            the attribute of the next agent in the squad. REQUIRED when
            the agent's attribute is ``lumiflux`` (every mode: it drives
            enemy RES, element-gated buffs, the slot-5 Attribute-DMG%
            match, and which anomaly her hits build); rejected for any
            other agent. One of the six real elements.
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
    counts_as_aftershock: bool = False
    engine_key: str | None = None
    engine_rank: int | None = None
    core_passive_active: bool = False
    mindscapes: dict[int, int] = field(default_factory=dict)
    potentials: dict[int, int] = field(default_factory=dict)
    additional_ability_stacks: int = 0
    scaling_inputs: dict[str, float] = field(default_factory=dict)
    supports: list[SupportConfig] = field(default_factory=list)
    discs: list[Disc] = field(default_factory=list)
    set_stacks: dict[str, int] = field(default_factory=dict)
    engine_squad_buffs: dict[str, int] = field(default_factory=dict)
    own_team_buffs: dict[str, int] = field(default_factory=dict)
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
    mutated_element: str | None = None
    luminize: "LuminizeConfig | None" = None


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
    refringe_mult: float      # Lumiflux Refringe zone (1.0 when unrefringed)
    def_mult: float
    res_mult: float
    dmg_taken_mult: float
    stun_mult: float
    # Every active conditional buff as {source, kind, value} — the front
    # ends render this as an itemized "Active buffs" summary.
    buff_breakdown: tuple = ()
    # Luminize mode: the display name of the skill that fired the
    # instance ("" in every other mode).
    luminize_skill: str = ""


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


def attack_element(agent: Agent, config: CalcConfig) -> str:
    """The element the agent's damage is dealt as.

    Every agent but a Lumiflux one deals their own attribute. A Lumiflux
    agent (Remielle — Attribute Mutation) deals the attribute of the next
    agent in the squad, a runtime choice: ``config.mutated_element``.

    Raises:
        CalcError: a Lumiflux agent without a ``mutated_element``, an
            unknown element, or a ``mutated_element`` set for a
            non-Lumiflux agent (it would be silently ignored otherwise).
    """
    if agent.attribute == LUMIFLUX:
        if config.mutated_element is None:
            raise CalcError(
                f"Agent '{agent.name}' has the Lumiflux attribute — her "
                f"damage counts as the NEXT squad agent's attribute "
                f"(Attribute Mutation). Set 'mutated_element' to one of "
                f"{list(ELEMENTS)}"
            )
        key = str(config.mutated_element).lower()
        if key not in ELEMENTS:
            raise CalcError(
                f"'mutated_element' must be one of {list(ELEMENTS)}, "
                f"got {config.mutated_element!r}"
            )
        return key
    if config.mutated_element is not None:
        raise CalcError(
            f"'mutated_element' only applies to Lumiflux agents — "
            f"{agent.name} always deals {agent.attribute} damage"
        )
    return agent.attribute


def _tag_matches(gate_tag: str, hit_tag: str | None,
                 counts_as_aftershock: bool = False) -> bool:
    """True when a skill-tag gate matches the calculated hit.

    A hit carries its selected ``hit_tag`` plus ``"aftershock"`` when it
    also counts as one (dual damage type, datamine ``damageType2`` —
    Anby S's Chain/Ultimate, Trigger's Harmonizing Shots), so gates on
    EITHER type apply to the same hit.
    """
    return gate_tag == hit_tag or (
        counts_as_aftershock and gate_tag == "aftershock"
    )


def _engine_buff_bonuses(
    engine: Engine, stacks: float, dealt_element: str, rank: int,
    mode: str = "direct", bracket: str = "dmg_bonus",
    skill_tag: str | None = None, counts_as_aftershock: bool = False,
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
    than the entered count clamps to its own cap, and an ``at_max_only``
    part contributes all-or-nothing at its cap (see ``EngineBuff``).

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
        if buff.skill_tags is not None and not any(
                _tag_matches(t, skill_tag, counts_as_aftershock)
                for t in buff.skill_tags):
            continue
        element_gate_waived = (
            buff.modes is not None and mode in ("disorder", "vortex")
        )
        if (buff.element is not None and not element_gate_waived
                and buff.element != dealt_element):
            continue
        value = buff.value_at(rank, stacks)
        if value:
            entries.append({"name": buff.name, "value": value})
    return entries


def _engine_buff_bonus(
    engine: Engine, stacks: float, dealt_element: str, rank: int,
    mode: str = "direct", bracket: str = "dmg_bonus",
    skill_tag: str | None = None, counts_as_aftershock: bool = False,
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
            counts_as_aftershock=counts_as_aftershock,
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


def _resolve_deferred_crit_dmg(
    kit: dict, crit_dmg_total: float
) -> tuple[float, list[dict]]:
    """Resolve kit effects that scale with the owner's TOTAL CRIT DMG.

    Called once ``crit_dmg_total`` holds every other CRIT DMG source
    (build panel + external + kit + engine passive + engine conditional).
    Anby S's Silver Star is the pilot: Aftershocks gain 35% of her total
    CRIT DMG, so the input is everything *except* what this resolves.

    Returns:
        ``(added, items)`` — the total CRIT DMG to add, and one
        buff-breakdown entry per contribution.
    """
    added = 0.0
    items: list[dict] = []
    for pending in kit["deferred_crit_dmg"]:
        value = pending["scaling"].resolve(crit_dmg_total) * pending["stacks"]
        if not value:
            continue
        added += value
        items.append({"source": pending["source"],
                      "kind": "crit_dmg", "value": value})
    return added, items


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
    ``disorder_mult_add``, ``refringe`` (floats) and ``dmg_bonus``,
    ``dmg_taken``, ``daze_bonus``, ``anomaly_buff`` (lists of bracket
    entries).

    Raises:
        CalcError: unmodeled/unknown buffs or mindscapes, stacks out of
            range, unknown support agents or squad sets, or a missing
            scaling input (e.g. Zhao's initial Max HP).
    """
    totals = {"crit_rate": 0.0, "crit_dmg": 0.0, "res_shred": 0.0,
              "flat_atk": 0.0, "atk_pct": 0.0, "pen_ratio": 0.0,
              "anomaly_proficiency": 0.0, "disorder_mult_add": 0.0,
              "sheer_force": 0.0, "hp_pct": 0.0, "def_red": 0.0,
              "refringe": 0.0,
              "dmg_bonus": [], "dmg_taken": [], "daze_bonus": [],
              "anomaly_buff": [], "sheer_dmg": [],
              "deferred_crit_dmg": [],
              "items": []}

    def record(source: str, kind: str, value: float) -> None:
        totals["items"].append(
            {"source": source, "kind": kind, "value": value}
        )

    def apply(buff, stacks: float, scaling_inputs: dict | None = None,
              owner: str = "") -> None:
        for effect in buff.effects:
            # Skill-tag-gated effects apply only to matching hits (e.g.
            # Catwalk buffs EX Specials only); a hit that also counts as
            # Aftershock matches aftershock gates on top of its own tag.
            if effect.skill_tag is not None and not _tag_matches(
                    effect.skill_tag, config.skill_tag,
                    config.counts_as_aftershock):
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
            # Effects scaling on the owner's TOTAL combat CRIT DMG cannot
            # be resolved yet: that total is only known once every other
            # crit_dmg source (build, external, kit, engine) is summed.
            # Park them; calculate()/calculate_sheer() resolve them last.
            # Self-referential by construction, so the effect's own output
            # is excluded from its input — Anby S's Silver Star takes 35%
            # of everything *else* (in-game 2026-08-02).
            if (effect.scaling is not None
                    and effect.scaling.input == TOTAL_CRIT_DMG_INPUT):
                if effect.at_max_stacks and stacks < buff.max_stacks:
                    continue
                totals["deferred_crit_dmg"].append({
                    "source": f"{owner}{buff.name}",
                    "scaling": effect.scaling,
                    "stacks": 1.0 if effect.at_max_stacks else stacks,
                })
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
                # Percent-stat inputs are FRACTIONS here. A percentage
                # typed into one (100 for "100% CRIT Rate") survives the
                # damage number — the scaling caps swallow it — but turns
                # into a panel minimum the optimizer can never satisfy,
                # so reject it where the unit is still obvious.
                if (effect.scaling.input in PERCENT_SCALING_INPUTS
                        and x > PERCENT_SCALING_INPUT_MAX):
                    raise CalcError(
                        f"{owner}{buff.name}: '{effect.scaling.input}' is "
                        f"{float(x):g}, which reads as {float(x) * 100:,.0f}%"
                        f" — this input is a FRACTION (0.78 = 78%). The web "
                        f"page takes the percentage and converts it for you."
                    )
                value = effect.scaling.resolve(float(x))
            else:
                value = effect.value
            if effect.at_max_stacks:
                # Fires only at FULL stacks, at its plain value (the kit
                # analog of the disc sets' at_max_extra — e.g. Ellen's
                # Potential Ice RES ignore at 10 Rising Storm stacks).
                if stacks < buff.max_stacks:
                    continue
                amount = value
            else:
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

    # Potential Awakening: the levels of one track replace each other, so
    # at most ONE may be active (the user's unlocked level).
    active_potentials = [
        (int(level_raw), stacks)
        for level_raw, stacks in config.potentials.items() if stacks
    ]
    if len(active_potentials) > 1:
        raise CalcError(
            f"Agent '{agent.name}': at most one Potential level may be "
            f"active (its levels replace each other), got levels "
            f"{sorted(level for level, _ in active_potentials)}"
        )
    for level_raw, stacks in config.potentials.items():
        level = int(level_raw)
        potential = agent.potentials.get(level)
        if potential is None:
            raise CalcError(
                f"Agent '{agent.name}' has no potential level {level} in "
                f"the data"
            )
        if potential.buff is None:
            raise CalcError(
                f"Potential level {level} ({potential.name}) has no "
                f"modeled damage effect: {potential.note}"
            )
        check_stacks(stacks, potential.buff.max_stacks,
                     f"Potential level {level} ({potential.name})")
        if stacks:
            apply(potential.buff, stacks, config.scaling_inputs,
                  owner=f"{agent.potential_name} P{level}: ")

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
        if entry is None or count < 4:
            continue
        if entry.auto_4pc_dmg:
            # Some auto 4pc DMG parts need a TEAMMATE to trigger (Phaethon's
            # Melody: "when a teammate — not the equipper — uses an EX
            # Special"). Solo (no supports) it can't fire.
            if not (entry.auto_4pc_dmg_needs_teammate and not config.supports):
                totals["dmg_bonus"].append(entry.auto_4pc_dmg)
                record(f"{entry.name} 4pc (auto)", "dmg_bonus",
                       entry.auto_4pc_dmg)
        # Anomaly-bracket half of a 4pc, optionally gated to a wearer
        # attribute (Feathered Fate: Attribute Anomaly DMG +15%, Lumiflux
        # equippers only). Anomaly modes only — the bracket doesn't exist
        # in direct/sheer.
        if (entry.auto_4pc_anomaly_dmg
                and mode in ("anomaly", "disorder", "vortex")
                and (entry.auto_4pc_attribute is None
                     or entry.auto_4pc_attribute == agent.attribute)):
            totals["anomaly_buff"].append(entry.auto_4pc_anomaly_dmg)
            record(f"{entry.name} 4pc (auto)", "anomaly_buff",
                   entry.auto_4pc_anomaly_dmg)

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

    # --- On-field agent's OWN team buffs: squad-wide buffs and enemy
    # debuffs cover their owner too (Norma's En-Nah +20% / Tech Divide /
    # M1 RES shred, Zhao's Ether Veil flat ATK), and the main agent can
    # never sit in ``supports`` — the self share is applied from here.
    # Scaled effects read the main config's scaling_inputs.
    if config.own_team_buffs:
        own_by_name = {b.name: b for b in agent.team_buffs}
        # Attacker-role slots currently switched ON, for the structural
        # duplicate check below. Mirrors the slots ``apply``-ed above.
        active_slots: list[tuple[str, object]] = []
        if config.core_passive_active and agent.core_passive is not None:
            active_slots.append(("core passive", agent.core_passive))
        if (config.additional_ability_stacks
                and agent.additional_ability is not None):
            active_slots.append(
                ("additional ability", agent.additional_ability))
        for level_raw, slot_stacks in config.mindscapes.items():
            entry = agent.mindscapes.get(int(level_raw))
            if slot_stacks and entry is not None and entry.buff is not None:
                active_slots.append((f"M{int(level_raw)}", entry.buff))
        for level_raw, slot_stacks in config.potentials.items():
            entry = agent.potentials.get(int(level_raw))
            if slot_stacks and entry is not None and entry.buff is not None:
                active_slots.append(
                    (f"Potential P{int(level_raw)}", entry.buff))

        for name, stacks in config.own_team_buffs.items():
            buff = own_by_name.get(name)
            if buff is None:
                raise CalcError(
                    f"Agent '{agent.name}' has no team buff named "
                    f"'{name}'; options: {sorted(own_by_name)}"
                )
            check_stacks(stacks, buff.max_stacks,
                         f"{agent.name} (self): {buff.name}")
            # A self_duplicate_of team buff is the SUPPORT-role copy of an
            # effect this agent already has as an attacker-role slot — the
            # same in-game buff, entered from the other side. On-field, the
            # attacker-role slot is always the right one, so this copy is
            # refused outright rather than only when the twin is switched
            # on: the two copies can differ in what they ask for (Anby S's
            # support copy takes a typed CRIT DMG, which the optimizer
            # would then turn into an unreachable panel minimum).
            if stacks and buff.self_duplicate_of is not None:
                slot = buff.self_duplicate_of.replace("_", " ")
                raise CalcError(
                    f"Agent '{agent.name}': team buff '{buff.name}' is the "
                    f"support-role copy of their {slot} — the same in-game "
                    f"effect. As the ON-FIELD agent use the {slot} instead; "
                    f"applying this copy as well would double-count it."
                )
            # Backstop for kits whose duplication is NOT declared in the
            # data: an own team buff carrying exactly the same effects as
            # an attacker-role slot that is also switched on is the same
            # in-game effect entered twice. Refuse rather than skip — a
            # silently halved buff is the failure mode this whole check
            # exists to prevent. Only same-character pairs reach here, so
            # kits whose two copies target DIFFERENT characters (Seed's
            # M2: hers vs her Vanguard's) are unaffected: as a support
            # her buffs arrive through ``supports``, never here.
            if stacks:
                for slot_label, slot_buff in active_slots:
                    if slot_buff.effects == buff.effects:
                        raise CalcError(
                            f"Agent '{agent.name}': team buff "
                            f"'{buff.name}' has the same effects as their "
                            f"{slot_label} ('{slot_buff.name}'), which is "
                            f"also active — the same in-game effect "
                            f"entered twice. Keep whichever one matches "
                            f"your rotation and clear the other."
                        )
            if stacks:
                apply(buff, stacks, config.scaling_inputs,
                      owner=f"{agent.name} (self): ")

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


def panel_stats(
    config: CalcConfig,
    *,
    consts: Constants | None = None,
    disc_data: DiscData | None = None,
    agents: dict[str, Agent] | None = None,
    engines: dict[str, Engine] | None = None,
    disc_sets: dict[str, DiscSet] | None = None,
) -> dict[str, float]:
    """Approximate in-game character-panel stats for the equipped build.

    What the out-of-combat panel shows: agent base + max-core stat
    boosts + engine (base ATK + advanced stat) + discs + always-on
    2-piece set bonuses. Conditional 4-piece stacks, kit/team buffs,
    engine conditionals and external buffs are combat-time state the
    in-game screen does not display, so they are deliberately excluded —
    these are the numbers the user can check against the character
    screen (hence "approximate": panel-only, not the combat totals the
    damage calculation uses).

    Returns:
        stat name -> value (fractions for percent stats). ``Sheer
        Force`` appears only for rupture-specialty agents (only their
        panel shows it): sheet ATK/HP conversions, no flat combat SF.

    Raises:
        CalcError: unknown agent or engine key.
        DiscError / AgentError: invalid discs.
    """
    consts = consts if consts is not None else load_constants()
    disc_data = disc_data if disc_data is not None else load_disc_data()
    agents = agents if agents is not None else load_agents()
    engines = engines if engines is not None else load_engines()
    disc_sets = disc_sets if disc_sets is not None else load_disc_sets()

    if config.agent_key not in agents:
        raise CalcError(
            f"Unknown agent '{config.agent_key}'; options: {sorted(agents)}"
        )
    agent = agents[config.agent_key]
    engine_key = (config.engine_key if config.engine_key is not None
                  else agent.default_engine)
    if engine_key not in engines:
        raise CalcError(
            f"Unknown engine '{engine_key}'; options: {sorted(engines)}"
        )
    engine = engines[engine_key]

    # The dealt element matters even out of combat: without it a Lumiflux
    # agent's would be "lumiflux", and every element-tagged disc (and
    # element-gated 2pc) would be dropped from the panel. This is a
    # display helper, so an un-chosen Lumiflux mutation falls back to the
    # pre-fix behavior instead of failing — calculate() raises the real
    # error, and the UI only reaches here after a successful calculation.
    try:
        attr = attack_element(agent, config)
    except CalcError:
        attr = None
    # set_stacks=None: 2pc bonuses only — 4pc stacks are combat state.
    build = aggregate_build(
        agent, engine, config.discs, disc_data, consts,
        disc_sets=disc_sets, set_stacks=None, attack_element=attr,
    )
    other = build.other
    stats = {
        "HP": build.hp,
        "ATK": build.atk_pre_combat,
        "DEF": (agent.base_def * (1.0 + other.get("DEF%", 0.0))
                + other.get("DEF", 0.0)),
        "Impact": ((agent.base_impact + agent.core_bonus_impact)
                   * (1.0 + other.get("Impact%", 0.0))),
        "CRIT Rate": build.crit_rate,
        "CRIT DMG": build.crit_dmg,
        "PEN Ratio": build.pen_ratio,
        "PEN": build.pen_flat,
        "Energy Regen": (
            (agent.base_energy_regen + agent.core_bonus_energy_regen)
            * (1.0 + other.get("Energy Regen%", 0.0))
        ),
        "Anomaly Mastery": build.anomaly_mastery,
        "Anomaly Proficiency": build.anomaly_proficiency,
    }
    if agent.specialty == "rupture":
        stats["Sheer Force"] = (
            build.atk_pre_combat * consts.sheer_force_atk_conversion
            + build.hp * agent.sheer_force_hp_conversion
        )
    return stats


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

    # Lumiflux (Attribute Mutation): resolve the dealt element once; it
    # drives the disc element match, every element gate and the boss RES.
    attr = attack_element(agent, config)

    build = aggregate_build(
        agent, engine, config.discs, disc_data, consts,
        disc_sets=disc_sets, set_stacks=config.set_stacks,
        attack_element=attr,
    )

    rank = _resolve_engine_rank(engine, config.engine_rank)
    kit = _kit_contributions(
        agent, config, agents, disc_sets, engines,
        mode="direct", dealt_element=attr,
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
    engine_crit_dmg = engine.passive_crit_dmg(rank)
    if engine_crit_dmg:
        crit_dmg += engine_crit_dmg
        buff_items.append({
            "source": f"{engine.name} passive CRIT DMG (R{rank})",
            "kind": "crit_dmg", "value": engine_crit_dmg,
        })
    # Engine conditional parts routed to the crit totals (Heartstring
    # Nocturne's unconditional CRIT DMG, Deep Sea Visitor's CRIT Rate
    # stacks) — added before the crit zones.
    for bracket, total_kind in (("crit_dmg", "crit_dmg"),
                                ("crit_rate", "crit_rate")):
        for part in _engine_buff_bonuses(
            engine, config.engine_buff_stacks, attr, rank,
            mode="direct", bracket=bracket, skill_tag=config.skill_tag,
            counts_as_aftershock=config.counts_as_aftershock,
        ):
            if total_kind == "crit_dmg":
                crit_dmg += part["value"]
            else:
                crit_rate += part["value"]
            buff_items.append({
                "source": f"{part['name']} (R{rank})",
                "kind": total_kind, "value": part["value"],
            })
    # LAST: effects taking a share of the owner's total CRIT DMG.
    scaled_cd, scaled_items = _resolve_deferred_crit_dmg(kit, crit_dmg)
    crit_dmg += scaled_cd
    buff_items.extend(scaled_items)
    crit_none = formulas.crit_mult_non_crit()
    crit_full = formulas.crit_mult_crit(crit_dmg)
    crit_avg = formulas.crit_mult_average(
        crit_rate, crit_dmg, consts.crit_rate_cap
    )

    bonus_entries = build.dmg_bonuses + list(config.external_dmg_bonuses)
    for passive_value in _engine_passive_bonuses(engine, attr, rank):
        bonus_entries.append(passive_value)
        buff_items.append({"source": f"{engine.name} passive (R{rank})",
                           "kind": "dmg_bonus", "value": passive_value})
    for part in _engine_buff_bonuses(
        engine, config.engine_buff_stacks, attr, rank,
        mode="direct", skill_tag=config.skill_tag,
        counts_as_aftershock=config.counts_as_aftershock,
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
        counts_as_aftershock=config.counts_as_aftershock,
    )
    bonus_entries.extend(tagged.values())
    bonus = formulas.dmg_bonus_mult(bonus_entries)

    eff_def = formulas.effective_def(
        boss.base_def, build.pen_ratio + kit["pen_ratio"], build.pen_flat,
        def_red=kit["def_red"])
    defense = formulas.def_mult(consts.level_coefficient(agent.level), eff_def)

    res = formulas.res_mult(
        boss.res_for(attr),
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

    # Lumiflux (Attribute Mutation): the agent's own hits — and therefore
    # the anomaly gauge they build — count as the mutated element.
    attr = attack_element(agent, config)

    build = aggregate_build(
        agent, engine, config.discs, disc_data, consts,
        disc_sets=disc_sets, set_stacks=config.set_stacks,
        attack_element=attr,
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

    # --- LUMINIZE MODE: resolve Remielle's multiplier from her data -----
    luminize_mult = None
    luminize_name = ""
    if config.luminize is not None:
        lum = config.luminize
        if config.disorder_replaced is not None or config.vortex_infused is not None:
            raise CalcError(
                "Luminize is a plain Attribute Anomaly instance - not "
                "compatible with disorder/vortex mode"
            )
        if config.abloom or config.anomaly_mult_override is not None:
            raise CalcError(
                "Luminize sets its own multiplier - don't also set "
                "'abloom' or 'anomaly_mult_override'"
            )
        provider = agents.get(lum.provider_key)
        if provider is None:
            raise CalcError(
                f"Unknown Luminize provider '{lum.provider_key}'; "
                f"options: {sorted(agents)}"
            )
        if provider.luminize is None:
            raise CalcError(
                f"Agent '{provider.name}' has no Luminize table - the mode "
                f"is for Lumiflux agents (Remielle)"
            )
        if provider.key == config.agent_key:
            raise CalcError(
                f"Luminize mode runs with the TEAMMATE whose anomaly is "
                f"stored in the Voidflare as the attacker, not with "
                f"{provider.name} herself - her instance scales off that "
                f"teammate's ATK / DMG% / PEN"
            )
        max_vf = provider.luminize.max_voidflares
        if (isinstance(lum.voidflares, bool)
                or not isinstance(lum.voidflares, int)
                or not 1 <= lum.voidflares <= max_vf * 2):
            raise CalcError(
                f"Luminize 'voidflares' must be an integer 1..{max_vf} "
                f"(up to {max_vf * 2} with her M6, which fires each "
                f"enhanced Basic's Luminizes twice), got {lum.voidflares!r}"
            )
        if any(v < 0 for v in lum.anomaly_dmg_bonuses):
            raise CalcError(
                "Luminize 'anomaly_dmg_bonuses' entries must be >= 0"
            )
        try:
            luminize_mult = provider.luminize.multiplier(
                lum.skill, lum.anomaly_proficiency, m4=lum.m4)
        except Exception as exc:
            raise CalcError(str(exc)) from None
        luminize_name = provider.luminize.skills[lum.skill].name
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
        if replaced_key == attr and not config.polarity_disorder:
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
        abloom_key = (config.abloom_element or attr).lower()
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
        anomaly = anomaly_data.anomalies[attr]
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
        dealt_element = attr
        per_proc_mult = override if override is not None else anomaly.mult
        hits = anomaly.hits
        if luminize_mult is not None:
            # The instance keeps the stored anomaly's ELEMENT but not its
            # type: it is Luminize DMG, so the element's own proc count
            # and multiplier are both replaced.
            name = f"Luminize - {luminize_name} [{dealt_element}]"
            per_proc_mult = luminize_mult
            hits = config.luminize.voidflares

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
    if config.luminize is not None:
        # A Luminize takes its Anomaly DMG Bonus from REMIELLE, and only
        # GENERAL ones reach it (Lin-0: Vivian's Corruption-specific bonus
        # would not apply, Yuzuha's general one would). So the attacker's
        # whole anomaly-buff bracket — their kit, their engine, manual
        # entries — is dropped and replaced by hers. Dropping it silently
        # would be a trap, so each discarded entry is recorded.
        for item in kit["items"]:
            if item["kind"] == "anomaly_buff":
                item["kind"] = "anomaly_buff (excluded: attacker's)"
        abuff_entries = list(config.luminize.anomaly_dmg_bonuses)
        for value in abuff_entries:
            kit["items"].append({
                "source": f"{config.luminize.provider_key} (Luminize)",
                "kind": "anomaly_buff", "value": value,
            })
    else:
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

    # REFRINGE (Lumiflux): a Lumiflux teammate enhances this anomaly's
    # Effect Strength by their coefficient. Its own multiplicative zone,
    # so it compounds with the anomaly-buff bracket instead of diluting
    # into it. It scales the finished anomaly, so it multiplies outside
    # the per-segment snapshot product.
    refringe = formulas.refringe_mult([kit["refringe"]])

    # Engine conditional flat AP (Weeping Gemini): applied per segment
    # below; recorded here (top-level stacks) for the buff breakdown.
    for part in _engine_buff_bonuses(
        engine, config.engine_buff_stacks, dealt_element, rank,
        mode=kind, bracket="anomaly_proficiency",
    ):
        kit["items"].append({
            "source": f"{part['name']} (R{rank})",
            "kind": "anomaly_proficiency", "value": part["value"],
        })

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
                attack_element=attr,
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
                # Engine conditional AP (Weeping Gemini's stacking AP) —
                # snapshot-weighted like every attacker-side stat.
                + _engine_buff_bonus(
                    engine, seg.engine_buff_stacks, dealt_element, rank,
                    mode=kind, bracket="anomaly_proficiency")
            )
        seg_ap_total = (
            seg.anomaly_proficiency_override
            if seg.anomaly_proficiency_override is not None
            else seg_ap_base + seg.external_anomaly_proficiency
        )
        bonus_entries = list(seg.external_dmg_bonuses) + kit["dmg_bonus"]
        if dealt_element == attr:
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
            attack_element=attr,
        )
        bare_atk = bare_build.atk
        bare_ap = bare_build.anomaly_proficiency + engine.passive_ap(rank)
        bare_entries = (
            list(bare_build.dmg_bonuses)
            if dealt_element == attr else []
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
        abloom_key = (config.abloom_element or attr).lower()
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
    # Her M1 (Naive Oath) ignores 50% All-Attribute RES on Luminize only,
    # and she is the SUPPORT here, so it cannot come from the attacker's
    # mindscape toggles.
    luminize_res = 0.50 if (config.luminize is not None
                            and config.luminize.res_ignore) else 0.0
    if luminize_res:
        kit["items"].append({
            "source": f"{config.luminize.provider_key} M1 (Luminize)",
            "kind": "res_shred", "value": luminize_res,
        })
    res = formulas.res_mult(
        boss.res_for(dealt_element),
        res_ignore=(config.external_res_shred + kit["res_shred"]
                    + luminize_res),
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
        return (mult * weighted_product * level_mult * refringe
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
        refringe_mult=refringe,
        def_mult=defense,
        res_mult=res,
        dmg_taken_mult=taken,
        stun_mult=stun_on,
        buff_breakdown=tuple(kit["items"]),
        luminize_skill=luminize_name,
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

    attr = attack_element(agent, config)

    build = aggregate_build(
        agent, engine, config.discs, disc_data, consts,
        disc_sets=disc_sets, set_stacks=config.set_stacks,
        attack_element=attr,
    )

    rank = _resolve_engine_rank(engine, config.engine_rank)
    kit = _kit_contributions(
        agent, config, agents, disc_sets, engines,
        mode="sheer", dealt_element=attr,
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
    engine_crit_dmg = engine.passive_crit_dmg(rank)
    if engine_crit_dmg:
        crit_dmg += engine_crit_dmg
        buff_items.append({
            "source": f"{engine.name} passive CRIT DMG (R{rank})",
            "kind": "crit_dmg", "value": engine_crit_dmg,
        })
    # Engine conditional parts routed to the crit totals (Sheer hits crit
    # normally, so both brackets apply here too).
    for bracket, total_kind in (("crit_dmg", "crit_dmg"),
                                ("crit_rate", "crit_rate")):
        for part in _engine_buff_bonuses(
            engine, config.engine_buff_stacks, attr, rank,
            mode="sheer", bracket=bracket, skill_tag=config.skill_tag,
            counts_as_aftershock=config.counts_as_aftershock,
        ):
            if total_kind == "crit_dmg":
                crit_dmg += part["value"]
            else:
                crit_rate += part["value"]
            buff_items.append({
                "source": f"{part['name']} (R{rank})",
                "kind": total_kind, "value": part["value"],
            })
    # LAST: effects taking a share of the owner's total CRIT DMG.
    scaled_cd, scaled_items = _resolve_deferred_crit_dmg(kit, crit_dmg)
    crit_dmg += scaled_cd
    buff_items.extend(scaled_items)
    crit_none = formulas.crit_mult_non_crit()
    crit_full = formulas.crit_mult_crit(crit_dmg)
    crit_avg = formulas.crit_mult_average(
        crit_rate, crit_dmg, consts.crit_rate_cap
    )

    bonus_entries = build.dmg_bonuses + list(config.external_dmg_bonuses)
    for passive_value in _engine_passive_bonuses(engine, attr, rank):
        bonus_entries.append(passive_value)
        buff_items.append({"source": f"{engine.name} passive (R{rank})",
                           "kind": "dmg_bonus", "value": passive_value})
    for part in _engine_buff_bonuses(
        engine, config.engine_buff_stacks, attr, rank,
        mode="sheer", skill_tag=config.skill_tag,
        counts_as_aftershock=config.counts_as_aftershock,
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
        counts_as_aftershock=config.counts_as_aftershock,
    )
    bonus_entries.extend(tagged.values())
    bonus = formulas.dmg_bonus_mult(bonus_entries)

    # Dedicated Sheer DMG bracket (build sets + kit + engine parts +
    # manual entries). CALIBRATED multiplicatively separate from DMG%
    # in-game 2026-07-10 (Rupture calibration #2: Yunkui Tales +10%).
    sheer_entries = (build.sheer_dmg_bonuses
                     + list(config.external_sheer_dmg) + kit["sheer_dmg"])
    for part in _engine_buff_bonuses(
        engine, config.engine_buff_stacks, attr, rank,
        mode="sheer", bracket="sheer_dmg", skill_tag=config.skill_tag,
        counts_as_aftershock=config.counts_as_aftershock,
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
        boss.res_for(attr),
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
        element=attr,
        buff_breakdown=tuple(buff_items),
    )
