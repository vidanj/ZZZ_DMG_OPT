"""Best-build search over the user's saved disc inventory.

Design: DOCS/optimizer_plan.md (§11 for the E1-E3 extensions). Given a
:class:`~.api.CalcConfig` the user just calculated, find the combination
of their SAVED discs (``data/user_discs.json``, plus the currently
equipped discs as unsaved virtual candidates) that maximizes one field of
the direct-hit results — or confirm the current discs are already
optimal. Everything except ``config.discs`` / ``config.set_stacks`` is
held constant.

Entry points::

    optimize(config, options) -> OptimizeResult          # direct hits
    optimize_anomaly(config, options) -> OptimizeResult  # anomaly family
    optimize_sheer(config, options) -> OptimizeResult    # Sheer (Rupture)

Layering: this module is front-end-agnostic (the UI's ``/optimize``
endpoint is a thin consumer, the CLI can adopt it later).

Search (plan §6 + §11 E2 + §17): per-slot dominance pruning, then an
exact **branch-and-bound** depth-first search. A partial build whose
upper bound cannot beat the current top-N threshold is skipped whole.
The bound (§17, :mod:`.linear_bound`) is a linear upper bound on
log(damage) valid over every reachable stat vector, so each disc gets
ONE scalar score and a node's bound is its score so far plus each
remaining slot's best score plus a count-aware set score — O(1) per
node, with no phantom disc carrying every main stat at once (the older
per-stat bound, still the fallback, was ~2x loose). Set-pattern seeds
give the search a near-optimal incumbent from the start. The winner and
every reported runner-up are re-run through :func:`~.api.calculate` and
must match to 1e-9 (the exactness guarantee, plan §4), so the reported
numbers are exactly what the user gets after equipping the build.

The three entry points share one implementation (:func:`_optimize`);
each mode contributes only its constant folding and fast path
(:class:`_ModeModel`).

Set-bonus policy per candidate combination (plan §5): 2-piece bonuses
always apply; a modeled 4-piece effect uses the user's entered stacks
when the set is also 4-piece in the baseline build, and otherwise —
depending on ``OptimizeOptions.set_assumption`` — ``max_stacks``
(``"max"``, default) or 0 (``"current"``). Team-element gate (§18): a
4-piece whose trigger needs an anomaly element no squad member has
(``needs_squad_element`` — Thunder Metal's Shock needs an Electric main
or support) is credited at 0 stacks in every build, the baseline
included; ``OptimizeResult.gated_sets`` lists them.

Required sets (§11b, §18): ``OptimizeOptions.required_4pc`` /
``required_2pc`` force the build to wear >= 4 / >= 2 pieces of a set —
the 2pc one exists for utility bonuses the damage math can't value
(King of the Summit's Daze).

Constraints (§11 E3): ``OptimizeOptions.min_stats`` filters the search
to builds meeting minimum final-build totals; constrained stats join the
dominance vectors and the bound, so pruning stays exact under them.

Maximize stat (§15 E5): ``OptimizeOptions.maximize_stat`` makes a stat
the PRIMARY ranking key and damage a bounded constraint — two passes,
"best damage" then "most of that stat among builds within
``maximize_tolerance`` of it" (see :func:`_search_maximizing`).
"""

from __future__ import annotations

import heapq
import math
import threading
from dataclasses import dataclass, field, replace
from itertools import combinations, product

from . import formulas
from .agent import Agent, load_agents
from .linear_bound import (
    LinearBoundError, LogLinear, affine, affine_combine, crit_zone_log_upper,
    golden_min, mccormick_upper, step_scaling_upper,
)
from .anomalies import AnomalyData, load_anomalies
from .api import (
    AnomalyResults, CalcConfig, CalcResults, attack_element, calculate,
    calculate_anomaly, calculate_sheer, _engine_buff_bonus,
    _engine_passive_bonuses, _kit_contributions, _resolve_engine_rank,
)
from .constants import Constants, load_constants
from .discs import (
    Disc, DiscData, DiscSet, disc_stats, load_disc_data, load_disc_sets,
    load_user_discs, set_4pc_max_stacks, set_4pc_stats,
)
from .enemies import ELEMENTS, Boss, load_bosses
from .engines import Engine, load_engines

#: Result fields a search may maximize (crit outcome × boss state).
OBJECTIVES = ("non_crit", "crit", "average",
              "non_crit_stunned", "crit_stunned", "average_stunned")

#: Anomaly-mode result fields :func:`optimize_anomaly` may maximize
#: (proc/tick vs full duration × boss state). ``full`` == ``per_proc`` for
#: one-shot bursts (Disorder/Vortex/Abloom), so those two rank identically
#: there — the split matters only for multi-tick anomalies (Burn, Shock…).
ANOMALY_OBJECTIVES = ("per_proc", "full",
                      "per_proc_stunned", "full_stunned")

#: 4-piece assumption modes for sets a combination newly completes (§5).
SET_ASSUMPTIONS = ("max", "current")

#: Stats ``OptimizeOptions.min_stats`` may constrain (§11 E3). Every mode
#: accepts the full list — a constrained stat need not scale that mode's
#: damage (kit requirements: an agent may need e.g. Energy Regen to use
#: its rotation at all). Semantics are the final panel totals: ATK/HP
#: are the finished panels (combat + kit brackets included); DEF is
#: agent base DEF × (1 + DEF%) + flat DEF; Impact / Energy Regen are
#: agent base × (1 + % from discs and the engine advanced stat); CRIT
#: Rate/DMG include external and kit conditionals; AP/AM are the build
#: aggregation's totals.
CONSTRAINT_STATS = ("ATK", "HP", "DEF", "CRIT Rate", "CRIT DMG",
                    "PEN Ratio", "PEN", "Impact", "Energy Regen",
                    "Anomaly Proficiency", "Anomaly Mastery")

#: Stats a sheer (Rupture) search may constrain: the shared list plus the
#: finished Sheer Force panel SheerResults reports.
SHEER_CONSTRAINT_STATS = CONSTRAINT_STATS + ("Sheer Force",)

#: Default cap on builds *evaluated* by the branch-and-bound search —
#: keeps the local server responsive on adversarial inventories (§11 E2;
#: the raw cartesian product no longer matters).
DEFAULT_COMBO_BUDGET = 5_000_000

#: Relative tolerance of the fast-path vs calculate() re-check (plan §4).
_RECHECK_RTOL = 1e-9

#: Default damage the maximize-stat search (§15 E5) may give up vs the
#: best build found — 5%.
DEFAULT_MAXIMIZE_TOLERANCE = 0.05

#: Weight of the damage term in the maximize-stat ranking key (§15 E5).
#: Both terms are normalized to ~1, so damage only ever decides between
#: builds whose maximized stat differs by less than this relative amount
#: — i.e. exact ties, which are COMMON (stat totals are discrete: many
#: builds reach the same 350 AP with wildly different damage).
_MAXIMIZE_TIE_WEIGHT = 1e-6

#: Stat buckets tracked per candidate. Indices 0-7 are the direct-hit
#: damage-relevant stats (the fast path and the dominance pruning share
#: these vectors); 8-9 exist only for min-stat constraints and join the
#: dominance comparison only when constrained (§11 E3); 10-12 are the
#: sheer-mode (Rupture) stats — HP feeds Sheer Force, Sheer DMG% is its
#: dedicated bracket — read only by :func:`optimize_sheer`'s fast path
#: (inert in direct/anomaly evaluations); 13-16 (DEF / Impact% / Energy
#: Regen%) exist only for min-stat constraints, inert for every mode's
#: damage. Routing mirrors ``agent._fold_stat``; a stat outside the
#: vector affects neither the damage nor a supported constraint and is
#: ignored, which is also what makes pruning effective (plan §6).
_B_ATK_PCT, _B_COMBAT_ATK_PCT, _B_ATK_FLAT, _B_CRIT_RATE, _B_CRIT_DMG, \
    _B_PEN_RATIO, _B_PEN_FLAT, _B_ATTR_DMG, _B_AP, _B_AM, \
    _B_HP_PCT, _B_HP_FLAT, _B_SHEER_DMG, \
    _B_DEF_PCT, _B_DEF_FLAT, _B_IMPACT, _B_ER, _B_AM_PCT, \
    _B_GEN_DMG, _B_GATED_CD = range(20)

_N_BUCKETS = 20
#: A stat-GATED 4pc CRIT DMG (Branch & Blade Song: +30% while Anomaly
#: Mastery >= 115) — bucket 19, filled only by set vectors. The fast paths
#: credit it only when the combination's panel AM clears the gate
#: (``_StatModel.gated_crit_dmg``); bounds assume it is met (it only ever
#: adds), so they stay valid.
#: Generic (element-agnostic) DMG% — Fanged Metal 4pc, tag-gated 2pc DMG%
#: — has its own bucket (18): it joins the same additive bracket as
#: Attribute DMG% in every mode, but unlike it still applies to a
#: Disorder / Vortex dealt as another element.
_DAMAGE_INDICES = tuple(range(8)) + (_B_GEN_DMG,)

#: The linear bound (§17) works on the bucket vector plus the two set
#: tag channels ``evaluate`` takes separately: the ordinary DMG% bracket
#: part and the Anomaly Buff bracket part of 4pc bonuses.
_T_DMG = _N_BUCKETS
_T_ANOM = _N_BUCKETS + 1
_N_LINEAR = _N_BUCKETS + 2

#: Safety margin (in log space, ~1e-9 relative) before the linear bound
#: may prune — float rounding in the score sums must never cut a build
#: that ties or beats the threshold.
_LOG_MARGIN = 1e-9

#: Anomaly damage-relevant buckets: the ATK / PEN / Attribute-DMG% / AP
#: stats. A build's CRIT (indices 3-4) never affects an anomaly — anomaly
#: crit (Phoenix's Sear) comes from its own kit stats, never disc CRIT;
#: AP (index 8) DOES, unlike direct hits. ``_B_ATTR_DMG`` is included here
#: but dropped by the caller when the burst deals an element other than the
#: agent's own attribute (cross-element Disorder/Vortex): the fast path
#: mirrors ``calculate_anomaly``, which credits the build's Attribute DMG%
#: only when ``dealt_element == agent.attribute``.
_ANOMALY_DAMAGE_INDICES = (_B_ATK_PCT, _B_COMBAT_ATK_PCT, _B_ATK_FLAT,
                           _B_PEN_RATIO, _B_PEN_FLAT, _B_ATTR_DMG, _B_AP,
                           _B_GEN_DMG)
_ANOMALY_DAMAGE_INDICES_NO_ATTR = tuple(
    i for i in _ANOMALY_DAMAGE_INDICES if i != _B_ATTR_DMG
)

#: Sheer (Rupture) damage-relevant buckets: ATK (feeds Sheer Force), CRIT
#: (Sheer crits like a direct hit), Attribute/generic DMG%, and the
#: dedicated Sheer DMG% bracket. PEN buckets (5-6) are deliberately absent
#: — Sheer damage has no DEF zone (defense = 1.0), so a PEN-heavy disc is
#: correctly dominated/ignored. The HP buckets join only when the agent
#: converts HP to Sheer Force (``sheer_force_hp_conversion > 0``) or an
#: HP constraint watches them — see :func:`optimize_sheer`.
_SHEER_DAMAGE_INDICES = (_B_ATK_PCT, _B_COMBAT_ATK_PCT, _B_ATK_FLAT,
                         _B_CRIT_RATE, _B_CRIT_DMG, _B_ATTR_DMG,
                         _B_SHEER_DMG, _B_GEN_DMG)
_SHEER_HP_INDICES = (_B_HP_PCT, _B_HP_FLAT)

#: Buckets a min-stat constraint watches (composed totals like ATK or
#: Sheer Force watch every bucket they are built from). Constrained
#: stats' buckets join the dominance comparison so a disc kept alive
#: only by a constraint is never discarded (§11 E3).
_CONSTRAINT_BUCKETS = {
    "ATK": (_B_ATK_PCT, _B_COMBAT_ATK_PCT, _B_ATK_FLAT),
    "HP": (_B_HP_PCT, _B_HP_FLAT),
    "DEF": (_B_DEF_PCT, _B_DEF_FLAT),
    "CRIT Rate": (_B_CRIT_RATE,),
    "CRIT DMG": (_B_CRIT_DMG,),
    "PEN Ratio": (_B_PEN_RATIO,),
    "PEN": (_B_PEN_FLAT,),
    "Impact": (_B_IMPACT,),
    "Energy Regen": (_B_ER,),
    "Anomaly Proficiency": (_B_AP,),
    "Anomaly Mastery": (_B_AM, _B_AM_PCT),
    "Sheer Force": (_B_ATK_PCT, _B_COMBAT_ATK_PCT, _B_ATK_FLAT,
                    _B_HP_PCT, _B_HP_FLAT),
}


def _dominance_indices(
    damage_indices: tuple[int, ...], min_stats: dict[str, float]
) -> tuple[int, ...]:
    """The bucket indices dominance pruning must compare: the mode's
    damage-relevant stats plus every constrained stat's buckets."""
    extra: list[int] = []
    for stat in min_stats:
        for index in _CONSTRAINT_BUCKETS.get(stat, ()):
            if index not in damage_indices and index not in extra:
                extra.append(index)
    return damage_indices + tuple(extra)


def _watched_stats(options: OptimizeOptions) -> dict[str, float]:
    """Stats dominance pruning must keep visible: every constrained stat
    plus the maximize-stat (§15 E5) — a disc that only carries the
    maximized stat is worthless for damage and would otherwise be pruned
    as dominated before it could win."""
    watched = {**options.min_stats, **options.min_panel_stats}
    if options.maximize_stat is not None:
        watched.setdefault(options.maximize_stat, 0.0)
    return watched


_BUCKET_BY_STAT = {
    "ATK%": _B_ATK_PCT,
    "Combat ATK%": _B_COMBAT_ATK_PCT,
    "ATK": _B_ATK_FLAT,
    "CRIT Rate": _B_CRIT_RATE,
    "CRIT DMG": _B_CRIT_DMG,
    "PEN Ratio": _B_PEN_RATIO,
    "PEN": _B_PEN_FLAT,
    "Attribute DMG%": _B_ATTR_DMG,
    # Generic (element-agnostic) DMG% — e.g. Fanged Metal 4pc's +35% — joins
    # the SAME additive DMG% bracket as Attribute DMG% (mirrors
    # agent._fold_stat) but in its own bucket: it still applies to a
    # Disorder / Vortex dealt as another element.
    "DMG%": _B_GEN_DMG,
    "Anomaly Proficiency": _B_AP,
    "Anomaly Mastery": _B_AM,
    # Sheer-mode stats (mirroring agent._fold_stat): sheet HP% scales the
    # base-HP bucket only; flat HP is added after it; Sheer DMG% is the
    # dedicated multiplicative bracket. Only optimize_sheer reads these.
    "HP%": _B_HP_PCT,
    "HP": _B_HP_FLAT,
    "Sheer DMG%": _B_SHEER_DMG,
    # Constraint-only stats (inert for damage in every mode): disc mains
    # and engine advanced stats use exactly these names.
    "DEF%": _B_DEF_PCT,
    "DEF": _B_DEF_FLAT,
    "Impact%": _B_IMPACT,
    "Energy Regen%": _B_ER,
    # Percent AM (slot-6 main, Phaethon's Melody 2pc): multiplies the
    # flat AM total — the AM constraint composes it like aggregate_build.
    "Anomaly Mastery%": _B_AM_PCT,
}


class OptimizeError(ValueError):
    """Raised for invalid optimizer options, an over-budget search, or
    constraints no combination can meet."""


class OptimizeCancelled(Exception):
    """Raised when :attr:`OptimizeOptions.cancel` is set mid-search (the
    page's Stop button). Not a ValueError: a stopped search is not an
    input error, and it carries no partial result — the user stops
    because the inputs were wrong."""


class _BudgetExhausted(Exception):
    """Internal: the DFS hit the evaluation budget — the heap holds the
    best builds found so far (anytime behavior, not an error)."""


#: Pay for the tighter count-aware set bound only when the quick static
#: bound lands within this factor of the incumbent threshold — nodes far
#: above it descend anyway, so the second bound would be wasted work.
_TIGHT_BOUND_MARGIN = 1.5

#: Wide searches WITHOUT a linear bound (§17 fallback) first run an
#: INTERNAL fast pass capped at this many candidates per slot
#: (budget-sliced): it finds set-coherent optima the greedy ascent cannot
#: reach (completing a 4pc takes coordinated swaps), giving the full DFS a
#: near-optimal incumbent before the main budget. With a linear bound the
#: pattern seeds (:data:`_PATTERN_SEEDS`) do that job in milliseconds.
_SEED_STAGE_CAP = 15

#: Set-pattern seeds (§17) evaluated exactly, best linear score first.
_PATTERN_SEEDS = 24


@dataclass(frozen=True)
class OptimizeOptions:
    """Search options (plan §3, §11).

    Attributes:
        objective: The :class:`~.api.CalcResults` field to maximize
            (one of :data:`OBJECTIVES`).
        set_assumption: 4-piece stacks assumed for sets a combination
            newly completes: ``"max"`` (full stacks, itemized in the
            result) or ``"current"`` (never credit undeclared
            conditionals). Sets already 4-piece in the baseline always
            use the user's entered stacks.
        locked_slots: Slots that must keep their current disc (or stay
            empty if the baseline leaves them empty).
        top_n: How many builds to report (the best + runners-up).
        combo_budget: Cap on total search WORK (leaf evaluations + the
            equally-priced interior bound probes). Reaching it is not an
            error: the best builds found so far are returned with
            ``budget_exhausted`` set (anytime behavior), so wall time
            stays bounded on any inventory.
        min_stats: stat name -> minimum final-build total the build must
            reach (keys from :data:`CONSTRAINT_STATS`). If the baseline
            itself misses a minimum, the best *feasible* build may deal
            less damage than the baseline (negative delta).
        required_4pc: Set key the build MUST wear >= 4 pieces of
            (§11b E4) — the user decides which 4pc passive to build
            around (synergy the model can't value); ``None`` = no
            requirement. Any registry set is allowed, modeled 4pc or
            not. Feasibility semantics mirror ``min_stats``.
        required_2pc: Set key the build MUST wear >= 2 pieces of (§18) —
            for a 2-piece the damage math can't value (King of the
            Summit's Daze +6% for a stunner), so the search would never
            pick it on its own. Must differ from ``required_4pc`` (a
            4-piece already includes its 2-piece); with both set the build
            is exactly 4pc + 2pc. Feasibility semantics mirror
            ``min_stats``.
        slot_main_stats: slot (4/5/6) -> main stat the slot's disc MUST
            have — for kit-required stats the damage math can't value
            (e.g. an Energy Regen% slot 6). Only discs with that main
            are candidates; an "Attribute DMG%" requirement also demands
            a matching (or legacy element-less) element. Feasibility
            semantics mirror ``min_stats``.
        sets_only: 2pc priority — every disc must belong to a set with
            >= 2 pieces in the build (no set-less discs, no singleton
            pieces). With ``required_4pc`` that means exactly 4pc + 2pc;
            alone it also allows 2+2+2 / 3+3 / 6. Feasibility semantics
            mirror ``min_stats``.
        candidate_cap: fast search — keep only the best N candidates per
            slot (by solo value; the equipped disc and the required
            4pc / 2pc sets survive the cut). APPROXIMATE by design: the true
            optimum may use a disc outside the cap. 0 = exact search.
        min_panel_stats: like ``min_stats``, but checked against the
            PANEL totals (agent base + core + engine + discs + set
            stats) — no kit/team/combat brackets. The user's BASE-STAT
            constraints ("the build itself must reach X, no combat buffs
            counted" — UI 'Base (panel)' rows) land here, and so do the
            self-scaling kit inputs, automatically: Astra claiming
            ``initial_atk`` 3,430 must reach panel ATK 3,430, and the
            combat total (which contains the very flat ATK that claim
            grants — circular) must not satisfy it. An explicit and an
            automatic minimum on the same stat merge by max.
        maximize_stat: Stat to maximize as the PRIMARY objective (§15
            E5), with damage demoted to a floor + tie-break: the search
            reports the build with the most of this stat among those
            within ``maximize_tolerance`` of the best damage found.
            ``None`` = pure damage (today's behavior). Same stat names
            and final-total definitions as ``min_stats`` — the stat need
            not scale the mode's damage at all, which is the point (an
            agent's OTHER hit reads it: Remielle's Luminize scales with
            her Anomaly Proficiency, which her direct hits ignore).
        maximize_tolerance: Fraction of damage the maximize-stat search
            may give up vs the best build found, in ``[0, 1)`` (0.05 =
            accept up to 5% less damage; 0 = only exact-damage ties).
            Ignored without ``maximize_stat``.
        cancel: Stop token — once set (from another thread), the search
            raises :class:`OptimizeCancelled` at its next checkpoint
            (every 256 interior nodes). ``None`` = not stoppable.
    """

    objective: str = "average"
    set_assumption: str = "max"
    locked_slots: frozenset[int] = frozenset()
    top_n: int = 5
    combo_budget: int = DEFAULT_COMBO_BUDGET
    min_stats: dict[str, float] = field(default_factory=dict)
    required_4pc: str | None = None
    required_2pc: str | None = None
    slot_main_stats: dict[int, str] = field(default_factory=dict)
    sets_only: bool = False
    candidate_cap: int = 0
    min_panel_stats: dict[str, float] = field(default_factory=dict)
    maximize_stat: str | None = None
    maximize_tolerance: float = DEFAULT_MAXIMIZE_TOLERANCE
    cancel: threading.Event | None = field(
        default=None, compare=False, repr=False)


@dataclass(frozen=True)
class BuildOption:
    """One searched build, re-verified through :func:`~.api.calculate`.

    Attributes:
        value: The objective field's value for this build.
        delta: Fractional improvement vs the baseline (0.124 = +12.4%);
            may be negative when constraints exclude the baseline.
        discs: The build's discs, ordered by slot.
        disc_ids: Parallel inventory ids; ``None`` marks a currently
            equipped disc that is not saved in the inventory.
        set_stacks: 4-piece stacks the evaluation assumed (§5) — feed
            these back into the config when equipping the build.
        changed_slots: Slots whose disc differs from the baseline build.
        results: Full verified results table for display.
        stat_value: This build's total of ``maximize_stat`` (``None``
            when the search maximized damage only). ``value`` stays the
            damage either way, so the comparison table is unchanged.
    """

    value: float
    delta: float
    discs: tuple[Disc, ...]
    disc_ids: tuple[str | None, ...]
    set_stacks: dict[str, int]
    changed_slots: tuple[int, ...]
    results: CalcResults
    stat_value: float | None = None


@dataclass(frozen=True)
class OptimizeResult:
    """Outcome of :func:`optimize` (plan §3).

    ``best`` equals the baseline build (``changed_slots == ()``) when
    ``already_optimal`` — ties break in favor of the current discs, so
    an equal-value alternative never reports a false improvement.
    ``already_optimal`` requires a *feasible* baseline: when the current
    build misses a ``min_stats`` minimum or doesn't wear the
    ``required_4pc`` set (``baseline_feasible`` False), the best
    feasible build is reported even at a negative delta.
    """

    objective: str
    set_assumption: str
    baseline_value: float
    baseline_feasible: bool
    already_optimal: bool
    best: BuildOption
    alternatives: tuple[BuildOption, ...]
    min_stats: dict[str, float]
    required_4pc: str | None
    combos_evaluated: int
    candidates_per_slot: dict[int, int]
    discs_pruned: int
    slot_main_stats: dict[int, str] = field(default_factory=dict)
    sets_only: bool = False
    # The evaluation budget ran out: ``best``/``alternatives`` are the
    # best builds FOUND, with no optimality claim (anytime behavior).
    budget_exhausted: bool = False
    # Panel-stat minimums enforced (auto-derived from the main agent's
    # self-scaling kit inputs; see OptimizeOptions.min_panel_stats).
    min_panel_stats: dict[str, float] = field(default_factory=dict)
    # Maximize-stat search (§15 E5): the stat that outranked damage, the
    # tolerance it was allowed, the best damage the first pass found
    # (``damage_ceiling``) and the floor that implied. All None/0 for an
    # ordinary damage search.
    maximize_stat: str | None = None
    maximize_tolerance: float = 0.0
    damage_ceiling: float | None = None
    damage_floor: float | None = None
    # Required 2-piece set (§18), or None.
    required_2pc: str | None = None
    # Team-element gate (§18): set key -> the element no squad member
    # has, for every relevant set whose 4pc was credited at 0 stacks.
    gated_sets: dict[str, str] = field(default_factory=dict)


def _bucket_vector(stats: dict[str, float]) -> tuple[float, ...]:
    """Fold named stat totals into the tracked bucket vector."""
    buckets = [0.0] * _N_BUCKETS
    for stat, value in stats.items():
        index = _BUCKET_BY_STAT.get(stat)
        if index is not None:
            buckets[index] += value
    return tuple(buckets)


def _disc_buckets(
    disc: Disc, disc_data: DiscData, attribute: str
) -> tuple[float, ...]:
    """One disc's bucket vector for an agent of ``attribute``.

    An Attribute DMG% main only counts when the disc's element matches
    the agent (§11 E1); a legacy element-less disc keeps the original
    assume-it-matches behavior — mirroring ``aggregate_build``.
    """
    stats = dict(disc_stats(disc, disc_data))
    if disc.element is not None and disc.element != attribute:
        stats.pop("Attribute DMG%", None)
    return _bucket_vector(stats)


def _set_2pc_buckets(
    entry: DiscSet, attribute: str,
    skill_tag: str | None = None, counts_as_aftershock: bool = False,
) -> tuple[float, ...]:
    """One set's 2-piece bucket vector for an agent attacking as
    ``attribute``.

    Mirrors ``set_bonus_stats``' element gate: an element-specific 2pc
    (Fanged Metal = Physical DMG +10%) is worth nothing when the damage
    is dealt as another element, so the optimizer must not value it —
    otherwise it recommends an off-element set for a phantom +10% (and
    the §4 re-check would fail, since ``calculate`` drops it).

    Skill-tag-gated 2pc DMG% (Shadow Harmony's Aftershock & Dash +15%)
    joins the same additive bracket as ``Attribute DMG%``, so it rides
    the same bucket — which also gives it the right piece threshold for
    free, since ``_fold_sets`` applies this vector from 2 pieces up. The
    tag arguments default to "no tagged bonus" so the ANOMALY fast path,
    where ``set_tagged_dmg_bonuses`` never runs, stays correct by
    omission.
    """
    stats = dict(entry.bonus_2pc)
    if (entry.bonus_2pc_attribute is not None
            and entry.bonus_2pc_attribute != attribute):
        stats.pop("Attribute DMG%", None)
    tagged = sum(
        b.value for b in entry.bonus_2pc_dmg
        if (skill_tag is not None and b.skill_tag == skill_tag)
        or (counts_as_aftershock and b.skill_tag == "aftershock")
    )
    if tagged:
        stats["DMG%"] = stats.get("DMG%", 0.0) + tagged
    return _bucket_vector(stats)


@dataclass(frozen=True)
class _Candidate:
    """One disc a slot may equip: inventory id (or None), disc, buckets."""

    disc_id: str | None
    disc: Disc
    buckets: tuple[float, ...]


def _prune_dominated(
    candidates: list[_Candidate],
    indices: tuple[int, ...] = _DAMAGE_INDICES,
) -> list[_Candidate]:
    """Drop candidates that can never appear in an optimum (plan §6).

    A is dominated by B (same slot) when A carries no set — or the same
    set as B — and B's bucket vector is >= A's in every compared
    component (``indices``: the damage-relevant stats, plus any stats a
    ``min_stats`` constraint watches, so a disc kept alive only by a
    constraint is never discarded). Set bonuses only ever add value, so
    replacing A with B can't lower any combination. On identical vectors
    the *earlier* candidate survives (the equipped disc is listed first,
    keeping ties resolved toward the current build).
    """
    kept: list[_Candidate] = []
    for i, a in enumerate(candidates):
        dominated = False
        for j, b in enumerate(candidates):
            if i == j:
                continue
            if a.disc.disc_set is not None and a.disc.disc_set != b.disc.disc_set:
                continue
            if any(b.buckets[k] < a.buckets[k] for k in indices):
                continue
            # b >= a everywhere compared; strict dominance, or an
            # identical vector where b's set coverage is at least a's
            # (ties within the same coverage keep the earlier candidate).
            a_vec = tuple(a.buckets[k] for k in indices)
            b_vec = tuple(b.buckets[k] for k in indices)
            if b_vec != a_vec:
                dominated = True
            elif a.disc.disc_set is None and b.disc.disc_set is not None:
                dominated = True
            elif a.disc.disc_set == b.disc.disc_set and j < i:
                dominated = True
            if dominated:
                break
        if not dominated:
            kept.append(a)
    return kept


def _validate_options(
    options: OptimizeOptions, objectives: tuple,
    constraint_stats: tuple = CONSTRAINT_STATS,
) -> dict[str, int]:
    """Validate the pre-load parts of ``options`` (shared by the search
    entry points); return the set requirements as ``{set key: minimum
    pieces}`` — 4 for ``required_4pc``, 2 for ``required_2pc`` (empty
    when neither is set).

    Raises:
        OptimizeError: unknown objective/assumption, bad top_n/budget,
            invalid locked slots or min-stat constraints, a malformed
            ``required_4pc`` / ``required_2pc``, or both naming the same
            set.
    """
    if options.objective not in objectives:
        raise OptimizeError(
            f"Unknown objective '{options.objective}'; "
            f"options: {list(objectives)}"
        )
    if options.set_assumption not in SET_ASSUMPTIONS:
        raise OptimizeError(
            f"Unknown set assumption '{options.set_assumption}'; "
            f"options: {list(SET_ASSUMPTIONS)}"
        )
    if isinstance(options.top_n, bool) or not isinstance(options.top_n, int) \
            or options.top_n < 1:
        raise OptimizeError("'top_n' must be an integer >= 1")
    if isinstance(options.combo_budget, bool) \
            or not isinstance(options.combo_budget, int) \
            or options.combo_budget < 1:
        raise OptimizeError("'combo_budget' must be an integer >= 1")
    if isinstance(options.candidate_cap, bool) \
            or not isinstance(options.candidate_cap, int) \
            or options.candidate_cap < 0:
        raise OptimizeError("'candidate_cap' must be an integer >= 0")
    bad_slots = [s for s in options.locked_slots if s not in (1, 2, 3, 4, 5, 6)]
    if bad_slots:
        raise OptimizeError(f"Invalid locked slots: {sorted(bad_slots)}")
    for source in (options.min_stats, options.min_panel_stats):
        for stat, minimum in source.items():
            if stat not in constraint_stats:
                raise OptimizeError(
                    f"Unknown constraint stat '{stat}'; "
                    f"options: {list(constraint_stats)}"
                )
            if isinstance(minimum, bool) \
                    or not isinstance(minimum, (int, float)) or minimum < 0:
                raise OptimizeError(
                    f"Minimum for '{stat}' must be a number >= 0, "
                    f"got {minimum!r}"
                )
    maximize = options.maximize_stat
    if maximize is not None:
        if maximize not in constraint_stats:
            raise OptimizeError(
                f"Unknown maximize stat '{maximize}'; "
                f"options: {list(constraint_stats)}"
            )
        tolerance = options.maximize_tolerance
        if isinstance(tolerance, bool) \
                or not isinstance(tolerance, (int, float)) \
                or not 0.0 <= tolerance < 1.0:
            raise OptimizeError(
                f"'maximize_tolerance' must be a number in [0, 1) "
                f"(fraction of damage the search may give up), "
                f"got {tolerance!r}"
            )
    requirements: dict[str, int] = {}
    for name, key, pieces in (("required_4pc", options.required_4pc, 4),
                              ("required_2pc", options.required_2pc, 2)):
        if key is None:
            continue
        if not isinstance(key, str) or not key.strip():
            raise OptimizeError(f"'{name}' must be a set key or None")
        if key in requirements:
            raise OptimizeError(
                f"'required_2pc' and 'required_4pc' are both '{key}': a "
                f"4-piece already includes its 2-piece bonus — pick a "
                f"different 2pc set or clear it"
            )
        requirements[key] = pieces
    for slot, main in options.slot_main_stats.items():
        if slot not in (4, 5, 6):
            raise OptimizeError(
                f"Main-stat requirements only apply to slots 4/5/6, "
                f"got slot {slot!r} (slots 1-3 have fixed mains)"
            )
        if not isinstance(main, str) or not main.strip():
            raise OptimizeError(
                f"Main-stat requirement for slot {slot} must be a "
                f"non-empty stat name"
            )
    return requirements


def _validate_slot_mains(
    options: OptimizeOptions, disc_data: DiscData
) -> None:
    """Check each required main stat exists in its slot's main table.

    Raises:
        OptimizeError: a stat the slot cannot roll as its main.
    """
    for slot, main in options.slot_main_stats.items():
        if main not in disc_data.main_stats[slot]:
            raise OptimizeError(
                f"Slot {slot} cannot have main stat '{main}'; options: "
                f"{sorted(disc_data.main_stats[slot])}"
            )


def _main_matches(disc: Disc, main: str, attribute: str) -> bool:
    """Whether ``disc`` satisfies a required main stat.

    An "Attribute DMG%" requirement also demands the disc's element
    matches the agent (legacy element-less discs pass) — an off-element
    Attribute DMG% main contributes nothing, so it doesn't satisfy the
    requirement's intent.
    """
    if disc.main_stat != main:
        return False
    if main == "Attribute DMG%":
        return disc.element is None or disc.element == attribute
    return True


def _gather_candidates(
    equipped: dict[int, Disc],
    user_discs: dict[str, Disc],
    disc_data: DiscData,
    attribute: str,
    options: OptimizeOptions,
    dominance_indices: tuple[int, ...],
    prune: bool,
) -> tuple[dict[int, list[_Candidate]], int]:
    """Per-slot candidate lists (plan §2): the equipped disc as a virtual
    candidate first, then the inventory — filtered by locks, by the
    ``slot_main_stats`` requirements and by ``sets_only`` (set-less discs
    can never satisfy 2pc priority) — dominance-pruned per slot.

    Shared by all three optimizers (they differ only in
    ``dominance_indices``). Returns ``(candidates, pruned_count)``.

    Raises:
        OptimizeError: a slot has discs but none satisfies the required
            main stat / set requirement.
    """
    candidates: dict[int, list[_Candidate]] = {}
    pruned_count = 0
    for slot in (1, 2, 3, 4, 5, 6):
        required_main = options.slot_main_stats.get(slot)

        def admissible(disc: Disc) -> bool:
            if options.sets_only and disc.disc_set is None:
                return False
            return required_main is None or _main_matches(
                disc, required_main, attribute)

        current = equipped.get(slot)
        slot_candidates: list[_Candidate] = []
        if current is not None and admissible(current):
            current_id = next(
                (i for i, d in user_discs.items() if d == current), None
            )
            slot_candidates.append(_Candidate(
                current_id, current,
                _disc_buckets(current, disc_data, attribute),
            ))
        if slot not in options.locked_slots:
            for disc_id, disc in user_discs.items():
                if disc.slot != slot or disc == current:
                    continue
                if not admissible(disc):
                    continue
                slot_candidates.append(_Candidate(
                    disc_id, disc,
                    _disc_buckets(disc, disc_data, attribute),
                ))
        if not slot_candidates:
            requirements = []
            if required_main is not None:
                requirements.append(f"main stat '{required_main}'")
            if options.sets_only:
                requirements.append("a drive set (2pc priority)")
            if requirements and (current is not None or any(
                    d.slot == slot for d in user_discs.values())):
                raise OptimizeError(
                    f"No equippable disc in slot {slot} has "
                    f"{' and '.join(requirements)} (check the 🔒 lock "
                    f"and your saved inventory)"
                )
        if prune and len(slot_candidates) > 1:
            before = len(slot_candidates)
            slot_candidates = _prune_dominated(slot_candidates,
                                               dominance_indices)
            pruned_count += before - len(slot_candidates)
        if slot_candidates:
            candidates[slot] = slot_candidates
    return candidates, pruned_count


def _sets_coverage_ok(discs) -> bool:
    """Whether every disc belongs to a set with >= 2 pieces (2pc
    priority): no set-less discs, no singleton pieces."""
    counts: dict[str, int] = {}
    for disc in discs:
        if disc.disc_set is None:
            return False
        counts[disc.disc_set] = counts.get(disc.disc_set, 0) + 1
    return all(count >= 2 for count in counts.values())


def _squad_elements(
    main_element: str, config: CalcConfig, agents: dict[str, Agent]
) -> frozenset[str]:
    """The anomaly elements the squad can apply (§18): the element the
    main agent's damage is dealt as (a Lumiflux main's mutated element)
    plus every entered support's attribute. A Lumiflux support adds
    nothing — her mutated element copies another squad member's."""
    elements = {main_element}
    elements.update(agents[support.agent_key].attribute
                    for support in config.supports)
    return frozenset(e for e in elements if e in ELEMENTS)


def _gated_sets(
    disc_sets: dict[str, DiscSet], squad: frozenset[str]
) -> dict[str, str]:
    """Sets whose 4pc needs an anomaly element no squad member has
    (``needs_squad_element``), mapped to that element (§18)."""
    return {
        key: entry.bonus_4pc.needs_squad_element
        for key, entry in disc_sets.items()
        if entry.bonus_4pc is not None
        and entry.bonus_4pc.needs_squad_element is not None
        and entry.bonus_4pc.needs_squad_element not in squad
    }


@dataclass(frozen=True)
class _StatModel:
    """The disc-independent pieces of the constraint / maximize-stat
    totals — the numbers the results panel reports — shared by every
    mode (each optimizer fills in its own constants).

    :meth:`total` is a stat's exact final total for a bucket vector;
    :meth:`upper` an affine upper bound of it on a box of bucket vectors
    (§17), for the maximize-stat ranking bound. Every total is
    non-decreasing in every bucket.
    """

    atk_base: float           # agent + engine base ATK
    kit_atk_pct: float
    kit_flat_atk: float
    hp_base: float
    core_hp_pct: float
    kit_hp_mult: float
    def_base: float
    crit_rate_base: float     # base + core CRIT Rate
    crit_dmg_base: float      # base + core CRIT DMG
    deferred_cd: tuple        # total-CRIT-DMG-scaled kit effects
    pen_ratio_base: float
    impact_base: float
    er_base: float
    ap_base: float
    am_base: float
    crit_rate_extra: float = 0.0  # kit / external / engine CRIT Rate
    crit_dmg_extra: float = 0.0   # kit / external / engine CRIT DMG
    sf_atk_conv: float = 0.0  # Sheer Force (rupture mode only)
    sf_hp_conv: float = 0.0
    sf_flat: float = 0.0
    deferred_atk: tuple = ()  # total-CRIT-Rate-scaled flat ATK (Severian)
    am_gate: float | None = None  # stat-gated 4pc AM threshold (B&B: 115)

    def gated_crit_dmg(self, buckets) -> float:
        """The stat-gated set CRIT DMG this combination earns: all of it
        when its PANEL Anomaly Mastery (composed like aggregate_build)
        clears the gate, else none."""
        gated = buckets[_B_GATED_CD]
        if not gated or self.am_gate is None:
            return 0.0
        am = (self.am_base + buckets[_B_AM]) * (1.0 + buckets[_B_AM_PCT])
        return gated if am >= self.am_gate - 1e-9 else 0.0

    def crit_dmg_total(self, total: float) -> float:
        """CRIT DMG after effects that take a share of the total."""
        if not self.deferred_cd:
            return total
        return total + sum(p["scaling"].resolve(total) * p["stacks"]
                           for p in self.deferred_cd)

    def atk(self, buckets) -> float:
        atk_pre_combat = (self.atk_base * (1.0 + buckets[_B_ATK_PCT])
                          + buckets[_B_ATK_FLAT])
        return (atk_pre_combat
                * (1.0 + buckets[_B_COMBAT_ATK_PCT] + self.kit_atk_pct)
                + self.kit_flat_atk
                + _deferred_atk_value(
                    self.deferred_atk, self.crit_rate_base
                    + buckets[_B_CRIT_RATE] + self.crit_rate_extra))

    def hp(self, buckets) -> float:
        return (self.hp_base * (1.0 + self.core_hp_pct + buckets[_B_HP_PCT])
                + buckets[_B_HP_FLAT]) * self.kit_hp_mult

    def total(self, stat: str, buckets) -> float:
        """``stat``'s final total (keys validated by the caller)."""
        if stat == "ATK":
            return self.atk(buckets)
        if stat == "HP":
            return self.hp(buckets)
        if stat == "Sheer Force":
            return (self.atk(buckets) * self.sf_atk_conv
                    + self.hp(buckets) * self.sf_hp_conv + self.sf_flat)
        if stat == "DEF":
            return (self.def_base * (1.0 + buckets[_B_DEF_PCT])
                    + buckets[_B_DEF_FLAT])
        if stat == "CRIT Rate":
            return (self.crit_rate_base + buckets[_B_CRIT_RATE]
                    + self.crit_rate_extra)
        if stat == "CRIT DMG":
            return self.crit_dmg_total(self.crit_dmg_base
                                       + buckets[_B_CRIT_DMG]
                                       + self.gated_crit_dmg(buckets)
                                       + self.crit_dmg_extra)
        if stat == "PEN Ratio":
            return buckets[_B_PEN_RATIO] + self.pen_ratio_base
        if stat == "PEN":
            return buckets[_B_PEN_FLAT]
        if stat == "Impact":
            return self.impact_base * (1.0 + buckets[_B_IMPACT])
        if stat == "Energy Regen":
            return self.er_base * (1.0 + buckets[_B_ER])
        if stat == "Anomaly Proficiency":
            return self.ap_base + buckets[_B_AP]
        # "Anomaly Mastery"
        return (self.am_base + buckets[_B_AM]) * (1.0 + buckets[_B_AM_PCT])

    def upper(self, stat: str, x0, lo, hi):
        """Affine upper bound of :meth:`total` on the box (exact for the
        affine stats; a McCormick plane for the ATK and AM products; the
        scaled CRIT DMG effects by their chord)."""
        hp = affine(self.hp_base * (1.0 + self.core_hp_pct) * self.kit_hp_mult,
                    (_B_HP_PCT, self.hp_base * self.kit_hp_mult),
                    (_B_HP_FLAT, self.kit_hp_mult))
        if stat in ("ATK", "Sheer Force"):
            u, v = _atk_factors(self.atk_base, self.kit_atk_pct)
            atk = affine_combine(
                (1.0, mccormick_upper(u, v, x0, lo, hi)),
                (1.0, _deferred_atk_upper(
                    self.deferred_atk,
                    self.crit_rate_base + self.crit_rate_extra, x0, lo, hi)),
                const=self.kit_flat_atk)
            if stat == "ATK":
                return atk
            return affine_combine((self.sf_atk_conv, atk),
                                  (self.sf_hp_conv, hp), const=self.sf_flat)
        if stat == "HP":
            return hp
        if stat == "DEF":
            return affine(self.def_base, (_B_DEF_PCT, self.def_base),
                          (_B_DEF_FLAT, 1.0))
        if stat == "CRIT Rate":
            return affine(self.crit_rate_base + self.crit_rate_extra,
                          (_B_CRIT_RATE, 1.0))
        if stat == "CRIT DMG":
            base = self.crit_dmg_base + self.crit_dmg_extra
            # The gated set bucket is counted as if met: a valid bound.
            dmg = affine(base, (_B_CRIT_DMG, 1.0), (_B_GATED_CD, 1.0))
            d_lo = base + lo[_B_CRIT_DMG] + lo[_B_GATED_CD]
            d_hi = base + hi[_B_CRIT_DMG] + hi[_B_GATED_CD]
            d0 = base + x0[_B_CRIT_DMG] + x0[_B_GATED_CD]
            slope, const = 1.0, 0.0
            for stacks, *shape in _deferred_params(list(self.deferred_cd)):
                k0, k1 = step_scaling_upper(*shape, d_lo, d_hi, d0)
                const += stacks * k0
                slope += stacks * k1
            return affine_combine((slope, dmg), const=const)
        if stat == "PEN Ratio":
            return affine(self.pen_ratio_base, (_B_PEN_RATIO, 1.0))
        if stat == "PEN":
            return affine(0.0, (_B_PEN_FLAT, 1.0))
        if stat == "Impact":
            return affine(self.impact_base, (_B_IMPACT, self.impact_base))
        if stat == "Energy Regen":
            return affine(self.er_base, (_B_ER, self.er_base))
        if stat == "Anomaly Proficiency":
            return affine(self.ap_base, (_B_AP, 1.0))
        return mccormick_upper(affine(self.am_base, (_B_AM, 1.0)),
                               affine(1.0, (_B_AM_PCT, 1.0)), x0, lo, hi)


def _panel_constants(agent: Agent, engine: Engine, consts: Constants) -> dict:
    """Constants for the PANEL-stat compositions (mode-independent):
    agent base + core boosts + engine base — the parts of the character
    screen no disc changes. Kit/team/combat brackets are deliberately
    absent (that is the point of panel constraints)."""
    return {
        "atk_base": agent.total_base_atk() + engine.base_atk,
        "hp_base": agent.total_base_hp(),
        "core_hp_pct": agent.core_bonus_hp_pct,
        "def_base": agent.base_def,
        "impact_base": agent.base_impact + agent.core_bonus_impact,
        "er_base": agent.base_energy_regen + agent.core_bonus_energy_regen,
        "crit_rate_base": consts.base_crit_rate + agent.core_bonus_crit_rate,
        "crit_dmg_base": consts.base_crit_dmg + agent.core_bonus_crit_dmg,
        "pen_ratio_core": agent.core_bonus_pen_ratio,
        "ap_base": (agent.base_anomaly_proficiency
                    + agent.core_bonus_anomaly_proficiency),
        "am_base": (agent.base_anomaly_mastery
                    + agent.core_bonus_anomaly_mastery),
        "sf_atk_conv": consts.sheer_force_atk_conversion,
        "sf_hp_conv": agent.sheer_force_hp_conversion,
    }


def _panel_feasible(
    min_panel_stats: dict[str, float], panel: dict, buckets: list[float],
    report: list | None = None,
) -> bool:
    """Whether a bucket total meets every PANEL minimum (see
    ``OptimizeOptions.min_panel_stats``). Compositions mirror
    :func:`~.api.panel_stats`; assumed 4pc stacks that grant sheet stats
    are the only (over-crediting) approximation. Monotone in every
    bucket, so the search bounds stay exact.

    ``report`` (diagnosis only, never the hot path) collects every
    shortfall as ``("panel", stat, total, minimum)`` and keeps checking
    instead of returning at the first miss."""
    for stat, minimum in min_panel_stats.items():
        if stat == "ATK":
            total = (panel["atk_base"] * (1.0 + buckets[_B_ATK_PCT])
                     + buckets[_B_ATK_FLAT])
        elif stat == "HP":
            total = (panel["hp_base"]
                     * (1.0 + panel["core_hp_pct"] + buckets[_B_HP_PCT])
                     + buckets[_B_HP_FLAT])
        elif stat == "DEF":
            total = (panel["def_base"] * (1.0 + buckets[_B_DEF_PCT])
                     + buckets[_B_DEF_FLAT])
        elif stat == "CRIT Rate":
            total = panel["crit_rate_base"] + buckets[_B_CRIT_RATE]
        elif stat == "CRIT DMG":
            total = panel["crit_dmg_base"] + buckets[_B_CRIT_DMG]
        elif stat == "PEN Ratio":
            total = panel["pen_ratio_core"] + buckets[_B_PEN_RATIO]
        elif stat == "PEN":
            total = buckets[_B_PEN_FLAT]
        elif stat == "Impact":
            total = panel["impact_base"] * (1.0 + buckets[_B_IMPACT])
        elif stat == "Energy Regen":
            total = panel["er_base"] * (1.0 + buckets[_B_ER])
        elif stat == "Anomaly Proficiency":
            total = panel["ap_base"] + buckets[_B_AP]
        elif stat == "Anomaly Mastery":
            total = ((panel["am_base"] + buckets[_B_AM])
                     * (1.0 + buckets[_B_AM_PCT]))
        else:   # "Sheer Force" (keys validated against the mode's list)
            atk = (panel["atk_base"] * (1.0 + buckets[_B_ATK_PCT])
                   + buckets[_B_ATK_FLAT])
            hp = (panel["hp_base"]
                  * (1.0 + panel["core_hp_pct"] + buckets[_B_HP_PCT])
                  + buckets[_B_HP_FLAT])
            total = atk * panel["sf_atk_conv"] + hp * panel["sf_hp_conv"]
        if total < minimum - 1e-12:
            if report is None:
                return False
            report.append(("panel", stat, total, minimum))
    return not report


def _make_feasible(
    options: OptimizeOptions, panel_consts: dict, stat_total,
):
    """Build a mode's ``feasible(buckets, aux, report=None)`` predicate.

    Shared by all three optimizers: the panel minimums compose
    identically everywhere (:func:`_panel_feasible`), and the combat
    minimums differ only through the mode's ``stat_total(stat, buckets,
    aux)`` closure — the same closure the maximize-stat search reads
    (§15 E5), so a stat can never be constrainable but not maximizable.

    ``report`` (diagnosis only, never the hot path) collects every
    shortfall as ``(kind, stat, total, minimum)`` instead of returning at
    the first miss.
    """
    def feasible(buckets: list[float], aux, report: list | None = None) -> bool:
        if options.min_panel_stats and not _panel_feasible(
                options.min_panel_stats, panel_consts, buckets, report):
            if report is None:
                return False
        for stat, minimum in options.min_stats.items():
            total = stat_total(stat, buckets, aux)
            if total < minimum - 1e-12:
                if report is None:
                    return False
                report.append(("combat", stat, total, minimum))
        return not report
    return feasible


#: Kit scaling inputs that ARE panel stats of their owner (agents.json
#: scaling.input names -> the min-stat constraint that panel maps to).
#: When the MAIN agent's active kit reads one, the user-entered value is
#: auto-added as a minimum-stat constraint: the search varies the panel,
#: but the buff amount stays at the entered constant — the constraint
#: guarantees no build is credited a bigger buff than its panel supports
#: (builds above the entered value under-credit, the conservative
#: direction). Non-stat inputs (potential_level, power_at_smash) and
#: stats the mode can't constrain (Norma's sheer_force outside rupture)
#: are skipped. Supports' inputs are THEIR panels — never constrained.
_SCALING_INPUT_STAT = {
    "initial_atk": "ATK",
    "initial_max_hp": "HP",
    "initial_crit_rate": "CRIT Rate",
    "initial_crit_dmg": "CRIT DMG",
    "initial_energy_regen": "Energy Regen",
    "impact": "Impact",
    "anomaly_mastery": "Anomaly Mastery",
    "anomaly_proficiency": "Anomaly Proficiency",
    "pen_ratio": "PEN Ratio",
    "sheer_force": "Sheer Force",
}


def _apply_scaling_input_minimums(
    agent: Agent, config: CalcConfig, options: OptimizeOptions,
    constraint_stats: tuple,
) -> OptimizeOptions:
    """Fold the MAIN agent's self-scaling kit inputs into
    ``min_panel_stats``.

    Collects the scaling inputs read by the agent's ACTIVE kit buffs
    (core passive / additional ability / mindscapes / potentials / own
    team buffs / riders — mirroring ``_kit_contributions``) and, for each that
    maps to a constrainable stat, raises the PANEL minimum to the
    entered value. Panel, not combat: an ``initial_*`` input is the
    character-screen value BEFORE kit/team buffs — comparing against the
    combat total would be circular (Astra's claimed panel ATK grants the
    flat ATK that inflates the combat total).
    """
    active_buffs = []
    if config.core_passive_active and agent.core_passive is not None:
        active_buffs.append(agent.core_passive)
    if config.additional_ability_stacks and agent.additional_ability is not None:
        active_buffs.append(agent.additional_ability)
    for level, stacks in config.mindscapes.items():
        entry = agent.mindscapes.get(int(level))
        if stacks and entry is not None and entry.buff is not None:
            active_buffs.append(entry.buff)
    for level, stacks in config.potentials.items():
        entry = agent.potentials.get(int(level))
        if stacks and entry is not None and entry.buff is not None:
            active_buffs.append(entry.buff)
    for pool, picks in ((agent.team_buffs, config.own_team_buffs),
                        (agent.riders, config.riders)):
        by_name = {b.name: b for b in pool}
        for name, stacks in picks.items():
            buff = by_name.get(name)
            if stacks and buff is not None:
                active_buffs.append(buff)

    minimums = dict(options.min_panel_stats)
    changed = False
    for buff in active_buffs:
        for effect in buff.effects:
            if effect.scaling is None:
                continue
            stat = _SCALING_INPUT_STAT.get(effect.scaling.input)
            value = config.scaling_inputs.get(effect.scaling.input)
            if (stat is None or stat not in constraint_stats
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float)) or value <= 0):
                continue
            if float(value) > minimums.get(stat, 0.0):
                minimums[stat] = float(value)
                changed = True
    return replace(options, min_panel_stats=minimums) if changed else options


def _baseline_mains_ok(
    equipped: dict[int, Disc],
    slot_main_stats: dict[int, str],
    attribute: str,
) -> bool:
    """Whether the baseline build satisfies every required main stat."""
    return all(
        equipped.get(slot) is not None
        and _main_matches(equipped[slot], main, attribute)
        for slot, main in slot_main_stats.items()
    )


#: Constraint stats the front ends show as percentages (fractions here).
_PERCENT_STATS = frozenset({"CRIT Rate", "CRIT DMG", "PEN Ratio"})

#: Constraint stat -> the kit scaling input(s) that can auto-derive a
#: panel minimum for it (reverse of ``_SCALING_INPUT_STAT``), so the
#: diagnosis can name the box the claimed value came from.
_STAT_SCALING_INPUTS: dict[str, tuple[str, ...]] = {
    stat: tuple(name for name, s in _SCALING_INPUT_STAT.items() if s == stat)
    for stat in set(_SCALING_INPUT_STAT.values())
}


def _fmt_stat(stat: str, value: float) -> str:
    """One constraint stat in the units the front ends display."""
    if stat in _PERCENT_STATS:
        return f"{value * 100:,.1f}%"
    return f"{value:,.1f}" if value % 1 else f"{value:,.0f}"


def _baseline_reasons(
    shortfalls: list,
    *,
    discs,
    equipped: dict[int, Disc],
    requirements: dict[str, int],
    baseline_counts: dict[str, int],
    disc_sets: dict[str, DiscSet],
    slot_main_stats: dict[int, str],
    attribute: str,
    sets_only: bool,
) -> list[str]:
    """Why the CURRENT build fails the requirements, one phrase each.

    Every phrase carries the build's actual value next to the required
    one: a bare list of the constraints in force (the old message)
    cannot tell the user WHICH one to relax, and an auto-derived panel
    minimum isn't even a constraint they typed.
    """
    reasons = []
    for key, pieces in requirements.items():
        held = baseline_counts.get(key, 0)
        if held < pieces:
            name = disc_sets[key].name if key in disc_sets else key
            reasons.append(
                f"it wears {held} piece(s) of {name}, not the required "
                f"{pieces}"
            )
    for kind, stat, total, minimum in shortfalls:
        if kind == "panel":
            inputs = " / ".join(_STAT_SCALING_INPUTS.get(stat, ()))
            source = (f" (claimed by the '{inputs}' kit input)" if inputs
                      else "")
            reasons.append(
                f"its panel {stat} is {_fmt_stat(stat, total)}, below the "
                f"{_fmt_stat(stat, minimum)} it claims{source}"
            )
        elif kind == "damage":
            # The maximize-stat floor (§15 E5): ``stat`` names the
            # objective, ``minimum`` is the floor the tolerance allows.
            reasons.append(
                f"its {stat} is {total:,.1f}, below the {minimum:,.1f} "
                f"floor the maximize-stat tolerance allows"
            )
        else:
            reasons.append(
                f"its {stat} is {_fmt_stat(stat, total)}, below the "
                f"required {_fmt_stat(stat, minimum)}"
            )
    for slot, main in slot_main_stats.items():
        disc = equipped.get(slot)
        if disc is None:
            reasons.append(f"slot {slot} is empty, so it cannot have the "
                           f"required {main} main stat")
        elif not _main_matches(disc, main, attribute):
            reasons.append(f"its slot {slot} main stat is {disc.main_stat}, "
                           f"not the required {main}")
    if sets_only and not _sets_coverage_ok(discs):
        counts: dict[str, int] = {}
        for disc in discs:
            if disc.disc_set is not None:
                counts[disc.disc_set] = counts.get(disc.disc_set, 0) + 1
        loose = sorted(
            disc.slot for disc in discs
            if disc.disc_set is None or counts[disc.disc_set] < 2
        )
        reasons.append(
            f"slot(s) {', '.join(map(str, loose))} complete no 2-piece set "
            f"bonus (2pc priority)"
        )
    return reasons


def _infeasible_error(
    reasons: list[str], budget_exhausted: bool, combo_budget: int,
) -> OptimizeError:
    """The 'nothing found' error, explaining what the current build
    misses so the user knows which requirement to relax."""
    if budget_exhausted:
        return OptimizeError(
            f"Search budget exhausted ({combo_budget:,} builds) before any "
            f"acceptable build was found; lock slots, add requirements, or "
            f"use the fast search"
        )
    if reasons:
        return OptimizeError(
            f"No combination of your saved discs meets the requirements, "
            f"and your current build does not either: "
            f"{'; '.join(reasons)}. Relax the requirement(s) above and "
            f"optimize again"
        )
    return OptimizeError(
        "No combination of your saved discs meets the requirements; "
        "relax them and optimize again"
    )


def _fold_sets(
    buckets: list[float], counts: dict[str, int],
    set_2pc: dict[str, tuple[float, ...]],
    set_4pc: dict[str, tuple[float, ...]],
    set_tag_bonus: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Add set bonuses for ``counts`` into ``buckets`` in place; return the
    4-piece bracket totals that are NOT build stats.

    Two channels, because they land in different multiplier zones:
    ``(dmg, anomaly)`` — the additive DMG% bracket (skill-tag 4pc DMG and,
    in anomaly mode, auto 4pc DMG additives) and the separate Anomaly
    Buff Multiplier bracket (Feathered Fate's Lumiflux-gated part).
    """
    tagged = 0.0
    tagged_anom = 0.0
    for key, count in counts.items():
        if count >= 2:
            for k, v in enumerate(set_2pc[key]):
                buckets[k] += v
        if count >= 4:
            for k, v in enumerate(set_4pc[key]):
                buckets[k] += v
            dmg, anom = set_tag_bonus[key]
            tagged += dmg
            tagged_anom += anom
    return tagged, tagged_anom


def _set_counts(combo) -> dict[str, int]:
    """Pieces per set in a build (set-less discs count nothing)."""
    counts: dict[str, int] = {}
    for cand in combo:
        key = cand.disc.disc_set
        if key is not None:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _pairings(items: list) -> list[list[tuple]]:
    """Every way to split ``items`` (even length) into unordered pairs —
    the 15 pairings of six slots, for 2+2+2 set patterns."""
    if not items:
        return [[]]
    first, rest = items[0], items[1:]
    result = []
    for j, partner in enumerate(rest):
        remaining = rest[:j] + rest[j + 1:]
        for tail in _pairings(remaining):
            result.append([(first, partner)] + tail)
    return result


def _set_scores(
    weights: list[float], keys, set_2pc_sparse: dict, set_4pc_sparse: dict,
    set_tag_bonus: dict[str, tuple[float, float]], n_buckets: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-set linear scores of a linear form (§17): the 2pc bonus, and
    the 4pc bonus with its tag channels — clamped at 0 (every set bonus
    only adds, so a non-positive score never helps a bound)."""
    two = {key: max(sum(weights[k] * v for k, v in set_2pc_sparse[key]), 0.0)
           for key in keys}
    four = {key: max(sum(weights[k] * v for k, v in set_4pc_sparse[key])
                     + weights[n_buckets] * set_tag_bonus[key][0]
                     + weights[n_buckets + 1] * set_tag_bonus[key][1], 0.0)
            for key in keys}
    return two, four


def _pattern_score(two: dict[str, float], four: dict[str, float]) -> float:
    """The best set score six slots can hold, count-free (§17): three
    2pc, or one set at 4+ pieces (its 2pc included) plus another 2pc —
    always a CONSISTENT pattern, unlike a per-stat maximum."""
    twos = sorted(two.values(), reverse=True)
    best = sum(twos[:3])
    for key in two:
        # The best OTHER set's 2pc: the top value, unless this set holds
        # it (then the runner-up — equal values tie fine).
        if twos[0] != two[key]:
            other = twos[0]
        else:
            other = twos[1] if len(twos) > 1 else 0.0
        best = max(best, two[key] + four[key] + other)
    return best


def _search(
    candidates: dict[int, list[_Candidate]],
    slots: list[int],
    equipped: dict[int, Disc],
    base_buckets: tuple[float, ...],
    evaluate,
    feasible,
    set_2pc: dict[str, tuple[float, ...]],
    set_4pc: dict[str, tuple[float, ...]],
    set_tag_bonus: dict[str, tuple[float, float]],
    candidate_sets: set[str],
    required: dict[str, int],
    pairs_only: bool,
    top_n: int,
    budget: int,
    bound: bool,
    slot_cap: int = 0,
    linearize=None,
    constrained: bool = True,
    floor: float | None = None,
    rank_bound=None,
    seeds: tuple = (),
    cancel: threading.Event | None = None,
) -> tuple[list, int, bool]:
    """Branch-and-bound DFS over the per-slot candidates (plan §11 E2, §17).

    ``required`` (§11b, §18): set key -> minimum pieces every accepted
    build must wear (the required 4pc set at 4, the required 2pc set at
    2); empty = no set requirement.

    ``pairs_only`` (2pc priority): only accept leaves where every set
    worn has >= 2 pieces (the callers already excluded set-less
    candidates, so singleton pieces are the only thing to reject here).

    ``slot_cap`` > 0 truncates each slot to its best ``slot_cap``
    candidates by solo value (equipped disc always kept; each required
    set keeps a piece per slot) — the user-facing "fast search",
    APPROXIMATE by design.

    ``linearize(x0, lo, hi) -> (c0, w) | None`` (§17) returns a linear
    upper bound ``log(value(x)) <= c0 + w · x`` valid on the box
    ``lo <= x <= hi`` of stat vectors (the 18 buckets plus the two set
    tag channels, :data:`_N_LINEAR` entries), linearized at ``x0``. With
    it, every disc gets one scalar score ``w · v`` and a partial build's
    bound is its score so far plus each remaining slot's best score plus
    a scalar set bound — O(1) per node, and no phantom disc carrying
    every main stat at once. ``None`` (or no ``linearize``) falls back to
    the per-stat bound below. ``constrained`` says whether ``feasible``
    can ever reject a build (min-stat constraints, the maximize-stat
    floor); without constraints the per-node feasibility probe is
    skipped.

    ``floor`` (maximize-stat pass 2, §15 E5): the objective is then a
    stat RANK, not damage, so ``linearize`` bounds the DAMAGE instead and
    prunes every branch that cannot reach ``floor``. ``rank_bound`` =
    ``(stat_upper, combine)`` bounds the rank itself the same separable
    way: ``stat_upper(x0, lo, hi)`` is an affine upper bound of the
    ranked stat on the box, and ``combine(stat_ub, log_damage_ub)`` turns
    the two bounds into a rank bound (non-decreasing in both). Without it
    the rank keeps the per-stat bound.

    ``seeds``: extra builds (``slots`` order) offered right after the
    baseline — the maximize-stat pass hands over the damage pass's
    winners, which are feasible by construction.

    ``cancel``: stop token (:attr:`OptimizeOptions.cancel`), polled every
    256 interior nodes; once set the search raises
    :class:`OptimizeCancelled` (not caught here, unlike the budget).

    Returns ``(heap, evals, exhausted)``; ``exhausted`` means the budget
    ran out and the heap holds the best builds found so far, with no
    completeness claim (anytime behavior — no exception).

    Mode-agnostic search core shared by the three optimizers.
    ``evaluate(buckets, tagged) -> (value, aux)`` and
    ``feasible(buckets, aux) -> bool`` carry all the mode-specific damage
    math; everything here — candidate ordering, the bounds, the top-N
    heap and the evaluation budget — is identical across modes because
    every mode's damage is monotone in every tracked bucket.

    The heap holds ``(value, -counter, combo)`` tuples with ``combo`` in
    ``slots`` order, ranked and de-duplicated by the caller (greedy seeds
    may also be found by the DFS).
    """
    if cancel is not None and cancel.is_set():
        raise OptimizeCancelled
    n_buckets = len(base_buckets)

    # Seed order: the equipped disc stays FIRST (the baseline is the first
    # build seeded, so exact ties resolve toward it); the rest are sorted
    # by their solo value so a strong incumbent forms early.
    def solo_value(cand: _Candidate) -> float:
        merged = list(base_buckets)
        for k, v in enumerate(cand.buckets):
            merged[k] += v
        return evaluate(merged, (0.0, 0.0))[0]

    for slot in slots:
        head = candidates[slot][:1] if equipped.get(slot) is not None else []
        tail = candidates[slot][len(head):]
        tail.sort(key=solo_value, reverse=True)
        candidates[slot] = head + tail

    # Fast search (approximate): keep the best ``slot_cap`` candidates per
    # slot. The tail is already solo-sorted; the equipped head survives at
    # index 0, and each required set keeps its best piece per slot so a
    # requirement can't become spuriously unreachable.
    if slot_cap:
        for slot in slots:
            full = candidates[slot]
            if len(full) <= slot_cap:
                continue
            kept = full[:slot_cap]
            for key in required:
                if not any(c.disc.disc_set == key for c in kept):
                    extra = next((c for c in full[slot_cap:]
                                  if c.disc.disc_set == key), None)
                    if extra is not None:
                        kept = kept + [extra]
            candidates[slot] = kept

    # Static cap: six slots fit at most three 2-piece sets and one 4-piece.
    static_set_ub = [0.0] * n_buckets
    for k in range(n_buckets):
        twos = sorted((set_2pc[key][k] for key in candidate_sets),
                      reverse=True)[:3]
        four = max((set_4pc[key][k] for key in candidate_sets), default=0.0)
        static_set_ub[k] = sum(twos) + four
    # Per-channel upper bound (value rises monotonically in both).
    static_tag_ub = (
        max((set_tag_bonus[key][0] for key in candidate_sets), default=0.0),
        max((set_tag_bonus[key][1] for key in candidate_sets), default=0.0),
    )
    set_2pc_sparse = {
        key: tuple((k, v) for k, v in enumerate(vec) if v)
        for key, vec in set_2pc.items()
    }
    set_4pc_sparse = {
        key: tuple((k, v) for k, v in enumerate(vec) if v)
        for key, vec in set_4pc.items()
    }

    heap: list[tuple[float, int, tuple[_Candidate, ...]]] = []
    heap_keys: set[tuple[int, ...]] = set()   # combos currently in the heap
    counter = 0
    evals = 0
    probes = 0     # interior node visits — same cost as a leaf

    def offer(value: float, combo: tuple[_Candidate, ...]) -> None:
        """Push one acceptable build into the top-N heap, deduplicated —
        a greedy seed the DFS finds again must not crowd out a genuine
        alternative."""
        nonlocal counter
        # A build that cannot enter a full heap is dropped before its
        # dedupe key is built — most offers end here.
        if len(heap) >= top_n and value <= heap[0][0]:
            return
        key = tuple(id(c) for c in combo)
        if key in heap_keys:
            return
        counter += 1
        entry = (value, -counter, combo)
        if len(heap) < top_n:
            heapq.heappush(heap, entry)
            heap_keys.add(key)
        elif value > heap[0][0]:
            removed = heapq.heapreplace(heap, entry)
            heap_keys.discard(tuple(id(c) for c in removed[2]))
            heap_keys.add(key)

    def sets_met(counts: dict[str, int]) -> bool:
        """Whether a build's set piece counts meet every requirement."""
        return all(counts.get(key, 0) >= pieces
                   for key, pieces in required.items())

    def combo_value(combo) -> tuple[float, bool]:
        """One full build's (value, acceptable) — the leaf math, reused
        by the greedy warm start below."""
        buckets, tagged, counts = _combo_buckets(
            combo, base_buckets, set_2pc, set_4pc, set_tag_bonus)
        value, aux = evaluate(buckets, tagged)
        acceptable = (
            feasible(buckets, aux)
            and sets_met(counts)
            and (not pairs_only
                 or all(count >= 2 for count in counts.values()))
        )
        return value, acceptable

    def push_seed(combo) -> None:
        value, acceptable = combo_value(combo)
        if acceptable:
            offer(value, tuple(combo))

    def ascend(start) -> None:
        """Coordinate ascent over the FULL candidate lists from ``start``;
        the (acceptable) local optimum joins the heap."""
        combo = list(start)
        for _ in range(3):                        # ascent passes
            improved = False
            for i, slot in enumerate(slots):
                best_cand = combo[i]
                best_value, best_ok = combo_value(combo)
                for cand in candidates[slot]:
                    if cand is combo[i]:
                        continue
                    combo[i] = cand
                    value, acceptable = combo_value(combo)
                    if acceptable and (not best_ok or value > best_value):
                        best_cand, best_value = cand, value
                        best_ok, improved = True, True
                combo[i] = best_cand
            if not improved:
                break
        push_seed(tuple(combo))

    def seed_incumbents() -> None:
        """Warm start: the DFS begins with a near-optimal top-N threshold
        so the bound prunes from node 1, and an exhausted budget still
        returns strong builds. Exactness is untouched — bounds only ever
        discard builds that cannot beat the heap, and seeds the DFS finds
        again are de-duplicated. Three stages, milliseconds total:
        the baseline (seeded FIRST so exact ties keep resolving toward
        the current discs), a mini exhaustive pass over the top 5 per
        slot, then coordinate ascent over the full lists from the best
        combos found so far."""
        push_seed(tuple(candidates[slot][0] for slot in slots))
        for combo in product(*(candidates[slot][:5] for slot in slots)):
            push_seed(combo)
        starts = [tuple(candidates[slot][0] for slot in slots),
                  tuple(max(candidates[slot], key=solo_value)
                        for slot in slots)]
        starts += [entry[2]
                   for entry in sorted(heap, key=lambda e: -e[0])[:3]]
        for start in starts:
            ascend(start)

    # --- Linear bound (§17): one scalar score per disc ---------------------
    lin = None
    x0 = None
    if bound and linearize is not None and slots:
        lo = list(base_buckets)
        hi = list(base_buckets)
        for slot in slots:
            for k in range(n_buckets):
                lo[k] += min(c.buckets[k] for c in candidates[slot])
                hi[k] += max(c.buckets[k] for c in candidates[slot])
        for k in range(n_buckets):
            hi[k] += static_set_ub[k]
        lo += [0.0, 0.0]
        hi += list(static_tag_ub)

    def linearize_at(combo):
        """The linear form tangent at one build (the bound is tightest
        near the builds it must separate from the threshold)."""
        buckets, tagged, _counts = _combo_buckets(
            combo, base_buckets, set_2pc, set_4pc, set_tag_bonus)
        point = buckets + list(tagged)
        return linearize(point, lo, hi), point

    def pattern_seeds(weights) -> None:
        """Set-aware warm start (§17): with one scalar score per disc,
        the best placement of every set PATTERN is cheap to assemble —
        each set at 4 pieces (every choice of its 4 slots) with the best
        2pc pair or best discs in the rest, and every 2+2+2 pairing of
        the six slots. The best few by score are evaluated exactly.
        Coordinate ascent alone cannot reach these: completing a 4pc
        takes coordinated swaps."""
        score = {id(c): sum(weights[k] * v
                            for k, v in enumerate(c.buckets) if v)
                 for slot in slots for c in candidates[slot]}
        two, four = _set_scores(weights, candidate_sets, set_2pc_sparse,
                                set_4pc_sparse, set_tag_bonus, n_buckets)

        def linear_total(combo) -> float:
            counts: dict[str, int] = {}
            total = 0.0
            for cand in combo:
                total += score[id(cand)]
                key = cand.disc.disc_set
                if key is not None:
                    counts[key] = counts.get(key, 0) + 1
            for key, count in counts.items():
                if count >= 2:
                    total += two[key]
                if count >= 4:
                    total += four[key]
            return total

        best_any = {slot: max(candidates[slot], key=lambda c: score[id(c)])
                    for slot in slots}
        best_of_set: dict[int, dict[str, _Candidate]] = {}
        for slot in slots:
            per_set: dict[str, _Candidate] = {}
            for cand in candidates[slot]:
                key = cand.disc.disc_set
                if key is not None and (key not in per_set or score[id(cand)]
                                        > score[id(per_set[key])]):
                    per_set[key] = cand
            best_of_set[slot] = per_set

        def best_pair(a: int, b: int) -> tuple:
            options = [(best_any[a], best_any[b])]
            for key, cand in best_of_set[a].items():
                other = best_of_set[b].get(key)
                if other is not None:
                    options.append((cand, other))
            return max(options, key=linear_total)

        # A required 2pc set (§18) usually scores ~0 here — that is why it
        # had to be required — so best_pair never picks it: each pattern
        # also gets variants that place it as one of its pairs.
        pair_keys = [key for key, pieces in required.items() if pieces == 2]

        def with_pair(pick: dict, a: int, b: int, key: str):
            """``pick`` with slots a/b holding ``key``'s best pieces, as a
            combo — or None when either slot has no piece of it."""
            first, second = best_of_set[a].get(key), best_of_set[b].get(key)
            if first is None or second is None:
                return None
            variant = dict(pick)
            variant[a], variant[b] = first, second
            return tuple(variant[slot] for slot in slots)

        combos = [tuple(best_any[slot] for slot in slots)]
        for key in candidate_sets:
            holders = [slot for slot in slots if key in best_of_set[slot]]
            for four_slots in combinations(holders, 4):
                pick = {slot: best_of_set[slot][key] for slot in four_slots}
                rest = [slot for slot in slots if slot not in pick]
                if len(rest) == 2:
                    pick[rest[0]], pick[rest[1]] = best_pair(*rest)
                    combos.extend(with_pair(pick, *rest, pair_key)
                                  for pair_key in pair_keys)
                else:
                    pick.update((slot, best_any[slot]) for slot in rest)
                combos.append(tuple(pick[slot] for slot in slots))
        if len(slots) == 6:
            for matching in _pairings(list(slots)):
                pick = {}
                for a, b in matching:
                    pick[a], pick[b] = best_pair(a, b)
                combos.append(tuple(pick[slot] for slot in slots))
                combos.extend(with_pair(pick, a, b, pair_key)
                              for pair_key in pair_keys
                              for a, b in matching)
        # Only builds meeting the set requirements can be accepted — don't
        # spend the exact evaluations on the rest.
        combos = [combo for combo in combos
                  if combo is not None and sets_met(_set_counts(combo))]
        combos.sort(key=linear_total, reverse=True)
        tried: set = set()
        for combo in combos:
            if len(tried) >= _PATTERN_SEEDS:
                break
            key = tuple(id(c) for c in combo)
            if key not in tried:
                tried.add(key)
                push_seed(combo)

    if slots:
        # The baseline is seeded FIRST so exact ties keep resolving toward
        # the current discs.
        push_seed(tuple(candidates[slot][0] for slot in slots))
        for combo in seeds:
            push_seed(combo)
        if bound and linearize is not None:
            first = (max(heap, key=lambda e: (e[0], e[1]))[2] if heap
                     else tuple(candidates[slot][0] for slot in slots))
            lin, x0 = linearize_at(first)
        if lin is not None:
            pattern_seeds(lin[1])
            for entry in sorted(heap, key=lambda e: -e[0])[:2]:
                ascend(entry[2])
            # Re-linearize at the (better) incumbent the seeds found.
            better, point = linearize_at(
                max(heap, key=lambda e: (e[0], e[1]))[2] if heap else first)
            if better is not None:
                lin, x0 = better, point
        else:
            seed_incumbents()
            # Two-stage warm start (wide searches without a linear bound
            # only): an internal fast pass over the top candidates per
            # slot on a slice of the budget; its winners join the heap.
            if (not slot_cap and bound and any(
                    len(candidates[slot]) > _SEED_STAGE_CAP
                    for slot in slots)):
                stage_heap, stage_evals, _ = _search(
                    {slot: list(candidates[slot]) for slot in slots},
                    list(slots), equipped, base_buckets, evaluate, feasible,
                    set_2pc, set_4pc, set_tag_bonus, candidate_sets,
                    required, pairs_only, top_n, min(budget // 2, 2_500_000),
                    True, slot_cap=_SEED_STAGE_CAP, cancel=cancel,
                )
                probes += stage_evals
                for value, _neg_counter, combo in sorted(stage_heap,
                                                         reverse=True):
                    offer(value, combo)

    def linear_scores(weights):
        """Per-candidate scores and per-set (2pc, 4pc) scores of one
        linear form."""
        per_disc = {id(c): sum(weights[k] * v
                               for k, v in enumerate(c.buckets) if v)
                    for slot in slots for c in candidates[slot]}
        two, four = _set_scores(weights, candidate_sets, set_2pc_sparse,
                                set_4pc_sparse, set_tag_bonus, n_buckets)
        return per_disc, two, four

    order = list(slots)
    scores: dict[int, float] = {}
    stat_scores: dict[int, float] = {}
    combine = None                       # the rank bound (pass 2 only)
    if lin is not None:
        c0, w = lin
        scores, sc2, sc4 = linear_scores(w)
        for slot in slots:
            # Stable: equal scores keep the seed order (equipped first).
            candidates[slot] = sorted(candidates[slot],
                                      key=lambda c: -scores[id(c)])
        # Slots whose discs differ most (variable mains: 4/5/6) first —
        # fixing them early tightens every deeper bound the most.
        order.sort(key=lambda slot: -(scores[id(candidates[slot][0])]
                                      - scores[id(candidates[slot][-1])]))
        lin_base = c0 + sum(w[k] * v for k, v in enumerate(base_buckets))
        static_set_score = _pattern_score(sc2, sc4)
    # Role of the linear form: bound the objective itself (damage), or —
    # with a ``floor`` — the damage that must stay above it.
    lin_objective = lin is not None and floor is None
    log_floor = (math.log(floor) - _LOG_MARGIN
                 if lin is not None and floor is not None and floor > 0.0
                 else None)
    lagrange: list = []                  # (λ, per-disc score, two, four)
    if lin is not None:
        if log_floor is not None and rank_bound is not None:
            try:
                stat_const, stat_terms = rank_bound[0](x0, lo, hi)
            except LinearBoundError:
                stat_terms = None
            if stat_terms is not None:
                sw = [0.0] * len(x0)
                for k, coef in stat_terms:
                    sw[k] += coef
                stat_scores, st2, st4 = linear_scores(sw)
                stat_base = stat_const + sum(
                    sw[k] * v for k, v in enumerate(base_buckets))
                combine = rank_bound[1]
                # Lagrangian relaxation of "damage >= floor": every
                # feasible build has damage_bound − log(floor) >= 0, so
                # stat + λ·(damage_bound − log floor) bounds its stat for
                # ANY λ >= 0 — and is still a sum of per-disc scores. The
                # root-optimal λ (convex in λ) and λ = 0 (the plain stat
                # bound) are kept; each node takes the smaller bound.
                # (More multipliers measured no fewer leaves on a
                # 653-disc inventory — the remaining leaves are builds
                # tied on the stat, which only their damage separates.)

                def lagrange_form(lam: float):
                    return (lam,
                            {cid: stat_scores[cid] + lam * scores[cid]
                             for cid in scores},
                            {k: st2[k] + lam * sc2[k] for k in candidate_sets},
                            {k: st4[k] + lam * sc4[k] for k in candidate_sets})

                def root_bound(lam: float) -> float:
                    _lam, per, two, four = lagrange_form(lam)
                    return (stat_base + lam * (lin_base - log_floor)
                            + sum(max(per[id(c)] for c in candidates[slot])
                                  for slot in slots)
                            + _pattern_score(two, four))

                stat_span = root_bound(0.0) - stat_base
                headroom = (lin_base + sum(
                    scores[id(candidates[slot][0])] for slot in slots)
                    + static_set_score - log_floor)
                lam_cap = 1e4 * max(stat_span, 1e-12) / max(headroom, 1e-12)
                lam_star = golden_min(root_bound, 0.0, lam_cap, 60)
                lambdas = sorted({0.0, lam_star})
                lagrange = [lagrange_form(lam) for lam in lambdas]

    # canonical position -> DFS position, to report combos in ``slots``
    # order (the seeds' order) whatever order the DFS walks.
    perm = (None if order == list(slots)
            else tuple(order.index(slot) for slot in slots))
    n_slots = len(order)
    last = n_slots - 1

    # suffix_max[i][k]: the most bucket k can still gain from order[i:].
    suffix_max = [[0.0] * n_buckets for _ in range(n_slots + 1)]
    for i in range(n_slots - 1, -1, -1):
        for k in range(n_buckets):
            suffix_max[i][k] = suffix_max[i + 1][k] + max(
                c.buckets[k] for c in candidates[order[i]]
            )
    # suffix_set_max[i][key]: how many MORE pieces of ``key`` order[i:] can
    # still equip (at most one per slot) — bounds reachable set bonuses.
    suffix_set_max: list[dict[str, int]] = [{} for _ in range(n_slots + 1)]
    for i in range(n_slots - 1, -1, -1):
        counts = dict(suffix_set_max[i + 1])
        for key in {c.disc.disc_set for c in candidates[order[i]]
                    if c.disc.disc_set is not None}:
            counts[key] = counts.get(key, 0) + 1
        suffix_set_max[i] = counts
    static_ub_vec = [
        [suffix_max[i][k] + static_set_ub[k] for k in range(n_buckets)]
        for i in range(n_slots + 1)
    ]
    # suffix_best[i]: the most linear score order[i:] can still add.
    suffix_best = [0.0] * (n_slots + 1)
    if lin is not None:
        for i in range(n_slots - 1, -1, -1):
            suffix_best[i] = (suffix_best[i + 1]
                              + scores[id(candidates[order[i]][0])])
    # Per Lagrangian form (pass 2): the most order[i:] can still add —
    # its lists are sorted by DAMAGE score, so take each slot's max.
    lagrange_suffix = []
    for _lam, per, _two, _four in lagrange:
        suffix = [0.0] * (n_slots + 1)
        for i in range(n_slots - 1, -1, -1):
            suffix[i] = suffix[i + 1] + max(per[id(c)]
                                            for c in candidates[order[i]])
        lagrange_suffix.append(suffix)

    work_buckets = list(base_buckets)
    work_counts: dict[str, int] = {}
    combo_stack: list[_Candidate] = []
    work_score = 0.0
    work_stat = 0.0
    slot_entries = [
        [(c, tuple((k, v) for k, v in enumerate(c.buckets) if v),
          c.disc.disc_set, scores.get(id(c), 0.0),
          stat_scores.get(id(c), 0.0))
         for c in candidates[slot]]
        for slot in order
    ]
    # Per slot, its candidates grouped by set (order kept) — when only one
    # set can keep a branch acceptable, only that group is walked.
    slot_groups: list[dict[str | None, list]] = []
    for entries in slot_entries:
        groups: dict[str | None, list] = {}
        for entry in entries:
            groups.setdefault(entry[2], []).append(entry)
        slot_groups.append(groups)
    check_reach = bool(required) or pairs_only
    # (slot depth, allowed set keys) -> that slot's candidates of those
    # sets, score order kept — for nodes where every remaining slot must
    # feed a set requirement.
    allowed_entries: dict[tuple, list] = {}

    def canonical(combo: tuple) -> tuple:
        return combo if perm is None else tuple(combo[p] for p in perm)

    def node_set_ub(i: int) -> tuple[list[float], float]:
        """Set-bonus upper bound (per bucket) for completions of the
        current partial build: fold every set at the piece count it can
        still reach, then cap componentwise by the static top-3+4pc
        bound. The fallback bound's tight half."""
        suffix_sets = suffix_set_max[i]
        ub = [0.0] * n_buckets
        tag_ub = [0.0, 0.0]
        for key, limit in suffix_sets.items():
            count = limit + work_counts.get(key, 0)
            if count >= 2:
                for k, v in set_2pc_sparse[key]:
                    ub[k] += v
            if count >= 4:
                for k, v in set_4pc_sparse[key]:
                    ub[k] += v
                tag_ub[0] += set_tag_bonus[key][0]
                tag_ub[1] += set_tag_bonus[key][1]
        for key, count in work_counts.items():
            if key in suffix_sets:
                continue        # already counted above
            if count >= 2:
                for k, v in set_2pc_sparse[key]:
                    ub[k] += v
            if count >= 4:
                for k, v in set_4pc_sparse[key]:
                    ub[k] += v
                tag_ub[0] += set_tag_bonus[key][0]
                tag_ub[1] += set_tag_bonus[key][1]
        for k in range(n_buckets):
            if ub[k] > static_set_ub[k]:
                ub[k] = static_set_ub[k]
        return ub, (min(tag_ub[0], static_tag_ub[0]),
                    min(tag_ub[1], static_tag_ub[1]))

    def set_bound(two: dict, four: dict, static: float):
        """``fn(i)``: a linear form's upper bound on every completion's
        set bonuses (count-aware, exact per set) — a small knapsack over
        the sets the held pieces plus ``order[i:]`` can still complete,
        with at most ``n_slots - i`` new pieces, capped by the static
        pattern score. Cached per (depth, held counts)."""
        cache: dict = {}

        def bound_at(i: int) -> float:
            state = (i, frozenset(work_counts.items()))
            hit = cache.get(state)
            if hit is not None:
                return hit
            avail = suffix_set_max[i]
            room = n_slots - i
            earned = 0.0
            best = [0.0] * (room + 1)   # best extra score with <= cap pieces
            for key in set(avail) | set(work_counts):
                held = work_counts.get(key, 0)
                reach = held + avail.get(key, 0)
                s2, s4 = two.get(key, 0.0), four.get(key, 0.0)
                if held >= 2:
                    earned += s2
                if held >= 4:
                    earned += s4
                options = []
                if held < 2 <= reach:
                    options.append((2 - held, s2))
                if held < 4 <= reach:
                    options.append((4 - held,
                                    (s2 if held < 2 else 0.0) + s4))
                if not options:
                    continue
                nxt = best[:]
                for cost, gain in options:
                    if cost > room or gain <= 0.0:
                        continue
                    for cap in range(cost, room + 1):
                        trial = best[cap - cost] + gain
                        if trial > nxt[cap]:
                            nxt[cap] = trial
                best = nxt
            value = min(earned + best[room], static)
            cache[state] = value
            return value

        return bound_at

    if lin is not None:
        set_score_ub = set_bound(sc2, sc4, static_set_score)
    lagrange_sets = [set_bound(two, four, _pattern_score(two, four))
                     for _lam, _per, two, four in lagrange]

    def lagrange_stat_ub(stat_sum: float, damage_sum: float, i: int,
                         set_parts: list) -> float:
        """The smallest Lagrangian bound (§17) on the ranked stat of any
        completion over ``order[i:]`` reaching the damage floor, given
        the stat / damage linear sums of the pieces placed so far."""
        best = math.inf
        for (lam, *_parts), suffix, set_part in zip(lagrange,
                                                    lagrange_suffix,
                                                    set_parts):
            value = (stat_sum + lam * (damage_sum - log_floor) + suffix[i]
                     + set_part)
            if value < best:
                best = value
        return best

    def rank_cannot_win(stat_ub: float, damage_log_ub: float) -> bool:
        """Pass 2: whether a rank bound cannot beat the top-N threshold."""
        if len(heap) < top_n:
            return False
        threshold = heap[0][0]
        return (combine(stat_ub, damage_log_ub)
                <= threshold - 1e-12 * abs(threshold))

    def reachable(i: int) -> bool:
        """Whether the partial build over ``order[:i]`` can still complete
        into an acceptable build (a necessary condition; exact at the
        leaves). Required sets: each one's pieces held plus what
        ``order[i:]`` can still equip must reach its minimum, and the
        pieces still missing must fit in the slots left (one per slot).
        2pc priority: no singleton set may be left that no later slot can
        complete."""
        suffix_sets = suffix_set_max[i]
        missing = 0
        for key, pieces in required.items():
            held = work_counts.get(key, 0)
            if held + suffix_sets.get(key, 0) < pieces:
                return False
            missing += max(pieces - held, 0)
        if missing > n_slots - i:
            return False
        if pairs_only and any(
                count == 1 and not suffix_sets.get(key)
                for key, count in work_counts.items()):
            return False
        return True

    def child_entries(i: int):
        """The ``order[i]`` candidates that keep the partial build
        completable into an acceptable one (exact, like
        :func:`reachable`, but settled once per node instead of per
        child): a required set or a 2pc-priority singleton that only this
        slot can still complete forces ONE set; when the pieces the
        required sets still miss fill every slot left, this slot must
        hold one of those sets; under 2pc priority a new set no later
        slot carries would be left a singleton.

        Returns ``(entries, pair_suffix)``: ``entries`` is ``None`` when
        no candidate qualifies; ``pair_suffix`` (2pc priority only, else
        ``None``) is what the later slots can still equip — a candidate
        opening a NEW set needs a later piece of it."""
        if not check_reach:
            return slot_entries[i], None
        suffix_sets = suffix_set_max[i + 1]
        pair_suffix = suffix_sets if pairs_only else None
        need_key = None
        missing = 0
        short: list[str] = []         # required sets still below minimum
        for key, pieces in required.items():
            held = work_counts.get(key, 0)
            if held >= pieces:
                continue
            have = held + suffix_sets.get(key, 0)
            if have < pieces - 1:
                return None, None
            if have == pieces - 1:     # only this slot can still close it
                if need_key is not None and need_key != key:
                    return None, None
                need_key = key
            missing += pieces - held
            short.append(key)
        room = n_slots - i             # this slot included
        if missing > room:
            return None, None
        if pairs_only:
            dying = [key for key, count in work_counts.items()
                     if count == 1 and not suffix_sets.get(key)]
            if len(dying) > 1:
                return None, None    # one piece completes only one pair
            if dying:
                if need_key is not None and need_key != dying[0]:
                    return None, None
                need_key = dying[0]
        if need_key is not None:
            if missing == room and need_key not in short:
                return None, None    # every slot left is spoken for
            return slot_groups[i].get(need_key), pair_suffix
        if short and missing == room:
            if len(short) == 1:
                return slot_groups[i].get(short[0]), pair_suffix
            state = (i, tuple(sorted(short)))
            entries = allowed_entries.get(state)
            if entries is None:
                allowed = set(short)
                entries = [entry for entry in slot_entries[i]
                           if entry[2] in allowed]
                allowed_entries[state] = entries
            return entries, pair_suffix
        return slot_entries[i], pair_suffix

    def score_cut(i: int, node_score: float, set_ub: float) -> float | None:
        """The disc score at or below which a child of node ``i`` cannot
        beat the current threshold (``None`` while the heap is not
        full). Children are score-sorted, so the loop stops there."""
        if len(heap) < top_n:
            return None
        threshold = heap[0][0]
        if threshold <= 0.0:
            return None
        return (math.log(threshold) - _LOG_MARGIN - node_score - set_ub
                - suffix_best[i + 1])

    def leaves(i: int, entries, pair_suffix, set_ub: float) -> None:
        """Evaluate every completion of the current partial build through
        the LAST slot inline (no call per leaf) — ``entries`` /
        ``pair_suffix`` from :func:`child_entries`: the set bonuses of the
        pieces already held are folded once, and each leaf adds only its
        own disc plus the set tier it completes."""
        nonlocal evals
        base_leaf = list(work_buckets)
        tag0, tag1 = _fold_sets(base_leaf, work_counts,
                                set_2pc, set_4pc, set_tag_bonus)
        base_tag = (tag0, tag1)
        node_score = lin_base + work_score if lin is not None else 0.0
        stat_node = stat_base + work_stat if combine is not None else 0.0
        cut = (score_cut(i, node_score, set_ub) if lin_objective
               else None)
        leaf_limit = (math.log(heap[0][0]) - _LOG_MARGIN
                      if lin_objective and len(heap) == top_n
                      and heap[0][0] > 0.0 else None)
        floor_cut = (log_floor - node_score - set_ub
                     if log_floor is not None else None)
        # A leaf's set state is exact (held pieces + its own), so its set
        # scores are too: the held sets' scores, plus the tier this piece
        # completes.
        if lin is not None:
            held_dmg = sum((sc2[key] if count >= 2 else 0.0)
                           + (sc4[key] if count >= 4 else 0.0)
                           for key, count in work_counts.items())
            held_stat = (sum((st2[key] if count >= 2 else 0.0)
                             + (st4[key] if count >= 4 else 0.0)
                             for key, count in work_counts.items())
                         if combine is not None else 0.0)
        for cand, sparse, key, score, stat_score in entries:
            if cut is not None and score <= cut:
                break            # score-sorted: no later leaf can win
            if floor_cut is not None and score < floor_cut:
                break            # no later leaf reaches the damage floor
            if (pair_suffix is not None and key is not None
                    and key not in work_counts):
                continue         # a new singleton set (2pc priority)
            if lin is not None:
                leaf_dmg, leaf_stat = held_dmg, held_stat
                if key is not None:
                    count = work_counts.get(key, 0) + 1
                    if count == 2:
                        leaf_dmg += sc2[key]
                        if combine is not None:
                            leaf_stat += st2[key]
                    elif count == 4:
                        leaf_dmg += sc4[key]
                        if combine is not None:
                            leaf_stat += st4[key]
                leaf_log = node_score + score + leaf_dmg
                if leaf_limit is not None and leaf_log <= leaf_limit:
                    continue     # its exact linear bound cannot win
                if log_floor is not None and leaf_log < log_floor:
                    continue     # cannot reach the damage floor
                if combine is not None and rank_cannot_win(
                        stat_node + stat_score + leaf_stat, leaf_log):
                    continue
            if evals + probes >= budget:
                # Anytime behavior: stop and report the best found so far.
                raise _BudgetExhausted
            evals += 1
            leaf = base_leaf[:]
            for k, v in sparse:
                leaf[k] += v
            tagged = base_tag
            if key is not None:
                count = work_counts.get(key, 0) + 1
                if count == 2:
                    for k, v in set_2pc_sparse[key]:
                        leaf[k] += v
                elif count == 4:
                    for k, v in set_4pc_sparse[key]:
                        leaf[k] += v
                    extra = set_tag_bonus[key]
                    tagged = (tag0 + extra[0], tag1 + extra[1])
            value, aux = evaluate(leaf, tagged)
            if len(heap) >= top_n and value <= heap[0][0]:
                continue         # cannot enter the top-N
            if feasible(leaf, aux):
                offer(value, canonical((*combo_stack, cand)))
                if lin_objective:
                    cut = score_cut(i, node_score, set_ub)
                    if len(heap) == top_n and heap[0][0] > 0.0:
                        leaf_limit = math.log(heap[0][0]) - _LOG_MARGIN

    def quick_infeasible(i: int) -> bool:
        """Whether even the optimistic per-bucket completion fails a
        constraint (constraints stay per-stat)."""
        static_vec = static_ub_vec[i]
        quick = [work_buckets[k] + static_vec[k] for k in range(n_buckets)]
        _quick_value, quick_aux = evaluate(quick, static_tag_ub)
        return not feasible(quick, quick_aux)

    def old_bound_prunes(i: int) -> bool:
        """The per-stat bound (the fallback without a linear form)."""
        threshold = heap[0][0] if len(heap) == top_n else None
        static_vec = static_ub_vec[i]
        quick = [work_buckets[k] + static_vec[k] for k in range(n_buckets)]
        quick_value, quick_aux = evaluate(quick, static_tag_ub)
        if not feasible(quick, quick_aux):
            return True          # not even the optimistic completion fits
        if threshold is not None and quick_value <= threshold:
            return True          # cannot beat the current top-N
        # The quick bound passed — pay for the tighter count-aware set
        # bound only when the quick bound is CLOSE to the threshold (a
        # node far above it descends anyway, so the second bound would be
        # wasted arithmetic).
        if (threshold is None
                or quick_value <= threshold * _TIGHT_BOUND_MARGIN):
            set_vec, tag_ub = node_set_ub(i)
            suffix = suffix_max[i]
            ub = [work_buckets[k] + suffix[k] + set_vec[k]
                  for k in range(n_buckets)]
            ub_value, ub_aux = evaluate(ub, tag_ub)
            if not feasible(ub, ub_aux):
                return True
            if threshold is not None and ub_value <= threshold:
                return True
        return False

    def dfs(i: int) -> None:
        nonlocal probes, work_score, work_stat
        # Every node visit counts toward the budget (bound probes cost
        # real time too) — so WALL TIME is capped even when the bounds
        # prune almost every leaf. Reachability is settled by the parent
        # BEFORE descending, so unreachable children cost no visit.
        probes += 1
        if evals + probes >= budget:
            raise _BudgetExhausted
        if cancel is not None and not (probes & 255) and cancel.is_set():
            raise OptimizeCancelled
        # Which candidates of this slot keep the build acceptable — before
        # the bound, so a node with none never pays for it.
        entries, pair_suffix = child_entries(i)
        if not entries:
            return
        set_ub = 0.0
        set_parts: list = []
        node_score = 0.0
        stat_node = 0.0
        if bound and lin_objective:
            if constrained and quick_infeasible(i):
                return
            node_score = lin_base + work_score
            if len(heap) == top_n and heap[0][0] > 0.0:
                limit = math.log(heap[0][0]) - _LOG_MARGIN
                if node_score + suffix_best[i] + static_set_score <= limit:
                    return
                set_ub = set_score_ub(i)
                if node_score + suffix_best[i] + set_ub <= limit:
                    return
            else:
                set_ub = set_score_ub(i)
        elif bound:
            if log_floor is not None:
                # Maximize-stat pass: no completion reaching the damage
                # floor -> nothing acceptable below this node.
                node_score = lin_base + work_score
                if node_score + suffix_best[i] + static_set_score < log_floor:
                    return
                set_ub = set_score_ub(i)
                if node_score + suffix_best[i] + set_ub < log_floor:
                    return
            if combine is not None:
                # ... and the rank (stat + tie-break damage) bound.
                if constrained and quick_infeasible(i):
                    return
                stat_node = stat_base + work_stat
                set_parts = [bound_at(i) for bound_at in lagrange_sets]
                if rank_cannot_win(
                        lagrange_stat_ub(stat_node, node_score, i, set_parts),
                        node_score + suffix_best[i] + set_ub):
                    return
            elif old_bound_prunes(i):
                return
        if i == last:
            leaves(i, entries, pair_suffix, set_ub)
            return
        use_cut = bound and lin_objective
        cut = score_cut(i, node_score, set_ub) if use_cut else None
        floor_cut = (log_floor - node_score - set_ub - suffix_best[i + 1]
                     if bound and log_floor is not None else None)
        rank_check = bound and combine is not None
        for cand, sparse, key, score, stat_score in entries:
            if cut is not None and score <= cut:
                break            # score-sorted: no later child can win
            if floor_cut is not None and score < floor_cut:
                break            # no later child reaches the damage floor
            if (pair_suffix is not None and key is not None
                    and key not in work_counts
                    and not pair_suffix.get(key)):
                continue         # a singleton no later slot can complete
            if rank_check and rank_cannot_win(
                    lagrange_stat_ub(stat_node + stat_score,
                                     node_score + score, i + 1, set_parts),
                    node_score + score + suffix_best[i + 1] + set_ub):
                continue
            for k, v in sparse:
                work_buckets[k] += v
            if key is not None:
                work_counts[key] = work_counts.get(key, 0) + 1
            work_score += score
            work_stat += stat_score
            combo_stack.append(cand)
            dfs(i + 1)
            combo_stack.pop()
            work_stat -= stat_score
            work_score -= score
            if key is not None:
                work_counts[key] -= 1
                if not work_counts[key]:
                    del work_counts[key]
            for k, v in sparse:
                work_buckets[k] -= v
            if use_cut:
                cut = score_cut(i, node_score, set_ub)

    def root() -> None:
        nonlocal evals, probes
        if check_reach and not reachable(0):
            return
        if n_slots:
            dfs(0)
            return
        # No slot to fill: the empty build is the only one.
        probes += 1
        if evals + probes >= budget:
            raise _BudgetExhausted
        evals += 1
        leaf = list(work_buckets)
        value, aux = evaluate(leaf, (0.0, 0.0))
        if feasible(leaf, aux):
            offer(value, ())

    exhausted = False
    try:
        root()
    except _BudgetExhausted:
        exhausted = True
    return heap, evals, exhausted


def _combo_buckets(
    combo, base_buckets: tuple[float, ...],
    set_2pc: dict[str, tuple[float, ...]],
    set_4pc: dict[str, tuple[float, ...]],
    set_tag_bonus: dict[str, tuple[float, float]],
) -> tuple[list[float], tuple[float, float], dict[str, int]]:
    """One full build's ``(buckets, tagged, set counts)`` — the leaf math
    of :func:`_search`, reused to re-evaluate a REPORTED build (the heap
    keeps only the ranking value, which in a maximize-stat search is not
    the damage the §4 re-check needs)."""
    buckets = list(base_buckets)
    counts: dict[str, int] = {}
    for cand in combo:
        for k, v in enumerate(cand.buckets):
            buckets[k] += v
        key = cand.disc.disc_set
        if key is not None:
            counts[key] = counts.get(key, 0) + 1
    tagged = _fold_sets(buckets, counts, set_2pc, set_4pc, set_tag_bonus)
    return buckets, tagged, counts


@dataclass(frozen=True)
class _SearchOutcome:
    """What :func:`_search_maximizing` hands back to an optimizer.

    ``check_baseline(buckets, aux, damage, report=None)`` is the predicate
    the BASELINE must be judged by — in a maximize-stat search it also
    enforces the damage floor, which is only known after the first pass
    (hence the baseline verdict is computed AFTER the search). ``rank`` is
    the ordering key the reported builds were chosen by (plain damage
    without a maximize stat).
    """

    heap: list
    evals: int
    exhausted: bool
    check_baseline: object
    rank: object
    ceiling: float | None = None
    floor: float | None = None


def _search_maximizing(
    candidates: dict[int, list[_Candidate]],
    slots: list[int],
    equipped: dict[int, Disc],
    base_buckets: tuple[float, ...],
    evaluate,
    feasible,
    stat_total,
    set_2pc: dict[str, tuple[float, ...]],
    set_4pc: dict[str, tuple[float, ...]],
    set_tag_bonus: dict[str, tuple[float, float]],
    candidate_sets: set[str],
    required: dict[str, int],
    options: OptimizeOptions,
    bound: bool,
    linearize=None,
    stat_model: _StatModel | None = None,
) -> _SearchOutcome:
    """Run the search, optionally under a maximize-stat objective (§15 E5).

    Without ``options.maximize_stat`` this is exactly one :func:`_search`
    call — the damage search, unchanged.

    With one, damage and the stat trade places in two passes:

    1. the damage search as usual → its best value is the **ceiling**;
    2. ``floor = ceiling × (1 - maximize_tolerance)``, and a second search
       ranks by the stat with ``damage >= floor`` as one more feasibility
       gate.

    Both passes share every other constraint, the §5 set policy and the
    same evaluation budget (so the search does up to twice the work — the
    floor usually prunes pass 2 hard).

    **Why the bounds stay valid:** the pass-2 ranking key is
    ``stat/stat_ref + w × damage/damage_ref`` with ``w`` tiny
    (:data:`_MAXIMIZE_TIE_WEIGHT`). Both terms are monotone in every stat
    bucket, so the branch-and-bound upper bound is still an over-estimate
    and the damage floor cuts a subtree only when even its optimistic
    completion falls short. The damage term exists to order the (common)
    exact ties in the stat; a stat difference below ``w`` relative counts
    as a tie, which is the documented approximation.
    """
    def run(cands, ev, fe, budget, lin, constrained, floor=None,
            rank_bound=None, seeds=()):
        return _search(
            cands, slots, equipped, base_buckets, ev, fe,
            set_2pc, set_4pc, set_tag_bonus, candidate_sets, required,
            options.sets_only, options.top_n, budget, bound,
            slot_cap=options.candidate_cap, linearize=lin,
            constrained=constrained, floor=floor, rank_bound=rank_bound,
            seeds=seeds, cancel=options.cancel,
        )

    # Pass 1 ranks by damage itself, so the mode's log-linear damage
    # bound (§17) applies; it is only ``constrained`` when a minimum can
    # reject a build.
    heap, evals, exhausted = run(
        candidates, evaluate, feasible, options.combo_budget, linearize,
        bool(options.min_stats or options.min_panel_stats))

    def damage_outcome() -> _SearchOutcome:
        """Pass 1 as the final answer (no maximize stat, or nothing
        feasible to measure a tolerance against)."""
        return _SearchOutcome(
            heap=heap, evals=evals, exhausted=exhausted,
            check_baseline=lambda buckets, aux, damage, report=None:
                feasible(buckets, aux, report),
            rank=lambda damage, stat_value: damage,
        )

    stat = options.maximize_stat
    # An empty heap means nothing satisfied the ordinary constraints:
    # there is no ceiling to measure a tolerance against, and the caller's
    # infeasibility diagnosis (which the empty heap triggers) is the right
    # answer.
    if stat is None or not heap:
        return damage_outcome()

    # --- Pass 1's winner sets the damage ceiling and the normalizers -----
    top = max(heap, key=lambda entry: (entry[0], entry[1]))
    ceiling = top[0]
    top_buckets, top_tagged, _counts = _combo_buckets(
        top[2], base_buckets, set_2pc, set_4pc, set_tag_bonus)
    _damage, top_aux = evaluate(top_buckets, top_tagged)
    floor = ceiling * (1.0 - options.maximize_tolerance)
    # Reference scales, so both terms of the ranking key are ~1 and the
    # tie weight keeps its intended meaning. The damage-optimal build's
    # own stat total is the natural scale; a zero one (maximizing a stat
    # that build has none of) falls back to 1.
    stat_ref = max(abs(stat_total(stat, top_buckets, top_aux)), 1.0)
    damage_ref = max(abs(ceiling), 1e-12)

    def rank(damage: float, stat_value: float | None) -> float:
        if stat_value is None:
            return damage
        return (stat_value / stat_ref
                + _MAXIMIZE_TIE_WEIGHT * damage / damage_ref)

    def evaluate_stat(buckets: list[float], tagged: tuple[float, float]):
        """Fast path of the second pass: rank by the stat, carry the
        damage along for the floor gate (and for the §4 re-check)."""
        damage, aux = evaluate(buckets, tagged)
        return rank(damage, stat_total(stat, buckets, aux)), (damage, aux)

    def feasible_stat(buckets: list[float], aux, report: list | None = None) -> bool:
        damage, inner_aux = aux
        ok = feasible(buckets, inner_aux, report)
        if damage < floor - abs(floor) * _RECHECK_RTOL:
            if report is None:
                return False
            report.append(("damage", f"{options.objective} damage",
                           damage, floor))
            ok = False
        return ok

    # The rank's own bound (§17): the stat's affine upper bound, plus the
    # tie-break damage term at its log-linear bound.
    rank_bound = None
    if stat_model is not None:
        def stat_upper(x0, lo, hi):
            return stat_model.upper(stat, x0, lo, hi)

        def combine(stat_ub: float, damage_log_ub: float) -> float:
            return (stat_ub / stat_ref + _MAXIMIZE_TIE_WEIGHT
                    * math.exp(damage_log_ub) / damage_ref)

        rank_bound = (stat_upper, combine)

    # A fresh candidate dict: _search reorders (and, under a candidate
    # cap, truncates) the lists it is given.
    pass2 = {slot: list(candidates[slot]) for slot in slots}
    # Pass 2 ranks by a stat + damage sum, not a log-linear product: the
    # damage bound (§17) cuts every branch that cannot reach the floor,
    # and the rank bound every branch that cannot beat the top-N. Only
    # the user's minimums still need the per-stat feasibility probe.
    heap2, evals2, exhausted2 = run(
        pass2, evaluate_stat, feasible_stat, options.combo_budget,
        linearize,
        bool(options.min_stats or options.min_panel_stats)
        if rank_bound is not None else True,
        floor=floor - abs(floor) * _RECHECK_RTOL, rank_bound=rank_bound,
        # The damage pass's winners are feasible (their damage is the
        # ceiling region): strong first incumbents for the stat ranking.
        seeds=tuple(entry[2] for entry in sorted(heap, reverse=True)))
    return _SearchOutcome(
        heap=heap2, evals=evals + evals2,
        exhausted=exhausted or exhausted2,
        # The baseline's damage is its VERIFIED value, so the floor is
        # judged against the same number the user was shown.
        check_baseline=lambda buckets, aux, damage, report=None:
            feasible_stat(buckets, (damage, aux), report),
        rank=rank, ceiling=ceiling, floor=floor,
    )


def _atk_factors(sheet_base: float, kit_atk_pct: float):
    """The ATK zone as ``u · v + kit flat ATK`` (§17): ``u`` the sheet
    ATK (base × (1 + ATK%) + flat), ``v`` the combat ATK% bracket — both
    affine in the bucket vector, mirroring every fast path's ATK."""
    u = affine(sheet_base, (_B_ATK_PCT, sheet_base), (_B_ATK_FLAT, 1.0))
    v = affine(1.0 + kit_atk_pct, (_B_COMBAT_ATK_PCT, 1.0))
    return u, v


def _am_gate(disc_sets: dict) -> float | None:
    """The Anomaly Mastery threshold of the registry's stat-gated 4pc
    bases (Branch & Blade Song: 115), or ``None`` without one. One
    ``_B_GATED_CD`` bucket carries them, so they must share it.

    Raises:
        OptimizeError: two gated sets with different thresholds.
    """
    thresholds = {
        entry.bonus_4pc.base_min_stat[1] for entry in disc_sets.values()
        if entry.bonus_4pc is not None
        and entry.bonus_4pc.base_min_stat is not None
    }
    if len(thresholds) > 1:
        raise OptimizeError(
            f"Stat-gated 4pc sets with different thresholds "
            f"{sorted(thresholds)} are not supported by the optimizer"
        )
    return next(iter(thresholds), None)


def _deferred_atk_value(deferred_atk, crit_rate_total: float) -> float:
    """Flat ATK of the total-CRIT-Rate-scaled kit effects (Severian's
    Optimal Setup) — mirrors :func:`~.api._resolve_deferred_flat_atk`."""
    return sum(p["scaling"].resolve(crit_rate_total) * p["stacks"]
               for p in deferred_atk)


def _deferred_atk_upper(deferred_atk, rate_const: float, x0, lo, hi):
    """Affine upper bound of :func:`_deferred_atk_value` on the box: the
    CRIT Rate total is ``rate_const`` + the CRIT Rate bucket, and each
    stepped scaling is bounded by its chord / cap
    (:func:`~.linear_bound.step_scaling_upper`)."""
    const, slope = 0.0, 0.0
    if deferred_atk:
        r_lo = rate_const + lo[_B_CRIT_RATE]
        r_hi = rate_const + hi[_B_CRIT_RATE]
        r0 = rate_const + x0[_B_CRIT_RATE]
        for p in deferred_atk:
            s = p["scaling"]
            k0, k1 = step_scaling_upper(s.base, s.threshold, s.per_step,
                                        s.per_value, s.cap, r_lo, r_hi, r0)
            const += p["stacks"] * (k0 + k1 * rate_const)
            slope += p["stacks"] * k1
    return affine(const, (_B_CRIT_RATE, slope))


def _deferred_params(deferred_cd: list) -> list[tuple]:
    """Total-CRIT-DMG-scaled kit effects (Anby S's Silver Star) as the
    ``(stacks, base, threshold, per_step, per_value, cap)`` tuples
    :func:`~.linear_bound.crit_zone_log_upper` takes."""
    return [(p["stacks"], p["scaling"].base, p["scaling"].threshold,
             p["scaling"].per_step, p["scaling"].per_value, p["scaling"].cap)
            for p in deferred_cd]


def _add_crit_zone(lb: LogLinear, variant: str, rate_base: float,
                   cap: float, dmg_base: float, deferred_cd: list,
                   x0, lo, hi) -> None:
    """Add the crit zone's log bound (§17) for ``variant`` (non_crit /
    crit / average): CRIT Rate = ``rate_base`` + bucket, CRIT DMG =
    ``dmg_base`` + bucket (+ the stat-gated set bucket, assumed met)
    before the total-scaled kit effects."""
    zone = crit_zone_log_upper(
        variant, affine(rate_base, (_B_CRIT_RATE, 1.0)), cap,
        affine(dmg_base, (_B_CRIT_DMG, 1.0), (_B_GATED_CD, 1.0)),
        _deferred_params(deferred_cd), x0, lo, hi,
    )
    if zone is not None:
        lb.add_linear(zone)


def _add_def_zone(lb: LogLinear, enemy_def: float, def_red: float,
                  pen_ratio_const: float, level_coef: float, lo, hi) -> None:
    """Add the DEF zone's log bound (§17), mirroring
    :func:`~.formulas.effective_def`: ``max(A·(1 − PEN Ratio) − PEN, 0)``
    with ``A`` the reduced enemy DEF is ``max(A − s, 0)`` for
    ``s = A·PEN Ratio + PEN``."""
    armor = enemy_def * (1.0 - def_red)
    s = affine(armor * pen_ratio_const, (_B_PEN_RATIO, armor),
               (_B_PEN_FLAT, 1.0))
    lb.add_log_def(s, armor, level_coef, lo, hi)


@dataclass(frozen=True)
class _ModeModel:
    """What one calc mode contributes to the shared optimizer (§17).

    Everything else — validation, candidates, the §5 set policy, the
    search, the §4 re-check and the result — is mode-independent
    (:func:`_optimize`).

    Attributes:
        damage_indices: Buckets the mode's damage reads (dominance).
        evaluate: Fast path ``(buckets, tagged) -> (value, None)``,
            mirroring the mode's ``calculate*`` exactly.
        linearize: ``(x0, lo, hi) -> (c0, w) | None`` — the log-linear
            damage bound (:func:`_search`).
        stats: The constraint / maximize-stat totals.
        anomaly_sets: Set bonuses follow the anomaly rules — no
            skill-tag-gated 2pc DMG% (no hit tag), and a 4pc's
            Anomaly-Buff part fills the second tag channel.
    """

    damage_indices: tuple[int, ...]
    evaluate: object
    linearize: object
    stats: _StatModel
    anomaly_sets: bool = False


def _crit_hit_model(config: CalcConfig, objective: str, agent: Agent,
                    attr: str, engine: Engine, boss: Boss, rank: int,
                    data: dict, sheer: bool) -> _ModeModel:
    """Direct hits (mirrors :func:`~.api.calculate`) or — ``sheer`` —
    Rupture's Sheer damage (mirrors :func:`~.api.calculate_sheer`): the
    same crit-able hit, with Sheer Force for ATK, the dedicated Sheer DMG
    bracket, and no DEF zone."""
    consts = data["consts"]
    mode = "sheer" if sheer else "direct"
    # Kit is computed with discs=[] so a set's DISC-dependent auto 4pc DMG%
    # (Wuthering Salon, Phaethon's Melody) is excluded here and folded per
    # combo via set_tag_bonus instead — otherwise a combination that newly
    # forms such a set would be mis-valued (its auto DMG% would be missing).
    # ``mode``/``dealt_element`` MUST mirror the calculation — element-gated
    # kit effects (e.g. Rina's +10% electric core passive, "squad Electric
    # DMG") are dropped without the dealt element, silently omitting them
    # from every fast-path build; they stay element-checked in sheer mode.
    kit = _kit_contributions(agent, replace(config, discs=[]),
                             data["agents"], data["disc_sets"],
                             data["engines"], mode=mode, dealt_element=attr)

    base_crit_rate = consts.base_crit_rate + agent.core_bonus_crit_rate
    base_crit_dmg = consts.base_crit_dmg + agent.core_bonus_crit_dmg
    agent_plus_engine_atk = agent.total_base_atk() + engine.base_atk
    kit_atk_pct = kit["atk_pct"]
    kit_flat_atk = kit["flat_atk"]

    def engine_part(bracket: str = "dmg_bonus") -> float:
        return _engine_buff_bonus(
            engine, config.engine_buff_stacks, attr, rank, mode=mode,
            bracket=bracket, skill_tag=config.skill_tag,
            counts_as_aftershock=config.counts_as_aftershock)

    # Engine always-on CRIT Rate (Qingming Birdcage) and CRIT DMG (Severed
    # Innocence) join the crit totals exactly as in the calculation; same
    # for the crit_dmg / crit_rate bracket conditional parts (Heartstring
    # Nocturne, Deep Sea Visitor).
    crit_rate_const = (config.external_crit_rate + kit["crit_rate"]
                       + engine.passive_crit(rank) + engine_part("crit_rate"))
    crit_dmg_const = (config.external_crit_dmg + kit["crit_dmg"]
                      + engine.passive_crit_dmg(rank)
                      + engine_part("crit_dmg"))
    # Anby S's Silver Star takes a share of the CRIT DMG TOTAL, so it
    # cannot be folded into the constant — it scales with the candidate
    # build's own CRIT DMG rolls (see api._resolve_deferred_crit_dmg).
    deferred_cd = kit["deferred_crit_dmg"]

    def with_deferred_cd(total: float) -> float:
        """CRIT DMG total after total-CRIT-DMG-scaled kit effects."""
        if not deferred_cd:
            return total
        return total + sum(p["scaling"].resolve(total) * p["stacks"]
                           for p in deferred_cd)

    # Severian's Optimal Setup: flat ATK from the CRIT Rate TOTAL — like
    # the CRIT DMG case it follows the candidate build's own CRIT Rate
    # rolls (see api._resolve_deferred_flat_atk), so ATK now rises with
    # the CRIT Rate bucket too (still monotone: dominance stays exact).
    deferred_atk = tuple(kit["deferred_flat_atk"])
    rate_const = base_crit_rate + crit_rate_const

    # Kit/support PEN Ratio (e.g. Rina's squad PEN Ratio) — a constant that
    # calculate() folds into the DEF zone (build.pen_ratio + kit pen_ratio);
    # the fast path must too, or a support granting PEN Ratio under-values
    # every build's DefMult (PEN interacts non-linearly, so it is not a flat
    # scalar — hence the §4 re-check catches the omission). Same for enemy
    # DEF reduction (Qingyi M1), a separate factor in the DEF zone. Sheer
    # has no DEF zone; there PEN Ratio is a constraint stat only.
    pen_ratio_const = kit["pen_ratio"] + agent.core_bonus_pen_ratio
    def_red_const = kit["def_red"]
    # HP side: sheet HP% scales the base bucket only, flat HP is added
    # after it, and kit/team HP% (Lucia, Dreamlit Hearth) multiplies the
    # FINISHED panel — the combat HP bracket, mirroring aggregate_build +
    # calculate_sheer. Inert for direct-hit damage (constraints only).
    hp_base = agent.total_base_hp()
    core_hp_pct = agent.core_bonus_hp_pct
    kit_hp_mult = 1.0 + kit["hp_pct"]
    sf_atk_conv = consts.sheer_force_atk_conversion if sheer else 0.0
    sf_hp_conv = agent.sheer_force_hp_conversion if sheer else 0.0
    flat_sf_const = (config.sheer_force_flat + kit["sheer_force"]
                     if sheer else 0.0)
    bonus_const = (
        sum(config.external_dmg_bonuses)
        + sum(_engine_passive_bonuses(engine, attr, rank))
        + engine_part()
        + sum(kit["dmg_bonus"])
    )
    # Dedicated Sheer DMG bracket constants (manual entries + kit + engine
    # buff parts — e.g. Qingming Birdcage's tag-gated Sheer DMG stacks);
    # the disc side (Yunkui Tales 4pc at max stacks) arrives per combo via
    # the _B_SHEER_DMG bucket.
    sheer_const = (sum(config.external_sheer_dmg) + sum(kit["sheer_dmg"])
                   + engine_part("sheer_dmg")) if sheer else 0.0
    level_coef = consts.level_coefficient(agent.level)
    crit_cap = consts.crit_rate_cap
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
    stun = formulas.stun_mult(
        objective.endswith("_stunned"), stun_base,
        list(config.external_daze_bonuses) + kit["daze_bonus"],
    )
    crit_variant = objective.replace("_stunned", "")
    const_mult = res * taken * stun * config.skill_multiplier

    def crit_mult_of(buckets: list[float]) -> float:
        if crit_variant == "non_crit":
            return 1.0
        crit_dmg = with_deferred_cd(
            base_crit_dmg + buckets[_B_CRIT_DMG]
            + stats.gated_crit_dmg(buckets) + crit_dmg_const)
        if crit_variant == "crit":
            return 1.0 + crit_dmg
        crit_rate = base_crit_rate + buckets[_B_CRIT_RATE] + crit_rate_const
        return 1.0 + min(max(crit_rate, 0.0), crit_cap) * crit_dmg

    def atk_of(buckets: list[float]) -> float:
        atk_pre_combat = (
            agent_plus_engine_atk * (1.0 + buckets[_B_ATK_PCT])
            + buckets[_B_ATK_FLAT]
        )
        return (atk_pre_combat
                * (1.0 + buckets[_B_COMBAT_ATK_PCT] + kit_atk_pct)
                + kit_flat_atk
                + _deferred_atk_value(
                    deferred_atk, rate_const + buckets[_B_CRIT_RATE]))

    if sheer:
        def evaluate(buckets: list[float], tagged: tuple[float, float]):
            """Fast path: Sheer damage of one bucket total (NO DEF zone:
            calculate_sheer uses defense = 1.0)."""
            hp_total = (
                hp_base * (1.0 + core_hp_pct + buckets[_B_HP_PCT])
                + buckets[_B_HP_FLAT]
            ) * kit_hp_mult
            sf = (atk_of(buckets) * sf_atk_conv + hp_total * sf_hp_conv
                  + flat_sf_const)
            # Only the DMG% channel matters here: the anomaly-bracket
            # channel exists in anomaly modes only.
            bonus = (1.0 + buckets[_B_ATTR_DMG] + buckets[_B_GEN_DMG]
                     + bonus_const + tagged[0])
            sheer_mult = 1.0 + sheer_const + buckets[_B_SHEER_DMG]
            return (sf * crit_mult_of(buckets) * bonus * sheer_mult
                    * const_mult), None
    else:
        def evaluate(buckets: list[float], tagged: tuple[float, float]):
            """Fast path: the objective value of one bucket total."""
            # Only the DMG% channel matters here: the anomaly-bracket
            # channel exists in anomaly modes only.
            bonus = (1.0 + buckets[_B_ATTR_DMG] + buckets[_B_GEN_DMG]
                     + bonus_const + tagged[0])
            eff_def = formulas.effective_def(
                boss.base_def, buckets[_B_PEN_RATIO] + pen_ratio_const,
                buckets[_B_PEN_FLAT], def_red=def_red_const
            )
            defense = level_coef / (level_coef + eff_def)
            return (atk_of(buckets) * crit_mult_of(buckets) * bonus
                    * defense * const_mult), None

    def linearize(x0, lo, hi):
        """``(c0, w)`` with ``log(evaluate) <= c0 + w · x`` on the box
        (§17), zone by zone; ``None`` when a precondition fails. Sheer
        Force is ``conv · (u·v + kit flat ATK) + HP part + flat SF``: with
        no HP conversion it is the ATK product shifted by a constant
        (concave in log); with one, a McCormick plane bounds ``u·v`` and
        the whole Sheer Force becomes affine. A CRIT-Rate-scaled flat ATK
        (Severian) makes the ATK zone ``u·v + flat + g(CRIT Rate)``: the
        McCormick plane plus the chord of ``g`` is affine, so its log
        tangent bounds the zone."""
        if const_mult <= 0:
            return None
        lb = LogLinear(_N_LINEAR)
        try:
            lb.add_const(math.log(const_mult))
            u, v = _atk_factors(agent_plus_engine_atk, kit_atk_pct)
            atk_plane = None
            if deferred_atk:
                atk_plane = affine_combine(
                    (1.0, mccormick_upper(u, v, x0, lo, hi)),
                    (1.0, _deferred_atk_upper(deferred_atk, rate_const,
                                              x0, lo, hi)),
                    const=kit_flat_atk)
            if not sheer:
                if atk_plane is None:
                    lb.add_log_product(u, v, kit_flat_atk, x0, lo, hi)
                else:
                    lb.add_log_affine(atk_plane, x0)
            elif sf_hp_conv == 0 and sf_atk_conv > 0 and atk_plane is None:
                lb.add_const(math.log(sf_atk_conv))
                lb.add_log_product(u, v,
                                   kit_flat_atk + flat_sf_const / sf_atk_conv,
                                   x0, lo, hi)
            else:
                hp_scale = hp_base * kit_hp_mult
                hp = affine(hp_scale * (1.0 + core_hp_pct),
                            (_B_HP_PCT, hp_scale), (_B_HP_FLAT, kit_hp_mult))
                if atk_plane is None:
                    atk_plane = affine_combine(
                        (1.0, mccormick_upper(u, v, x0, lo, hi)),
                        const=kit_flat_atk)
                sheer_force = affine_combine(
                    (sf_atk_conv, atk_plane), (sf_hp_conv, hp),
                    const=flat_sf_const,
                )
                lb.add_log_affine(sheer_force, x0)
            _add_crit_zone(lb, crit_variant, base_crit_rate + crit_rate_const,
                           crit_cap, base_crit_dmg + crit_dmg_const,
                           deferred_cd, x0, lo, hi)
            lb.add_log_affine(affine(1.0 + bonus_const, (_B_ATTR_DMG, 1.0),
                                     (_B_GEN_DMG, 1.0), (_T_DMG, 1.0)), x0)
            if sheer:
                lb.add_log_affine(affine(1.0 + sheer_const,
                                         (_B_SHEER_DMG, 1.0)), x0)
            else:
                _add_def_zone(lb, boss.base_def, def_red_const,
                              pen_ratio_const, level_coef, lo, hi)
        except LinearBoundError:
            return None
        return lb.c0, lb.w

    if sheer:
        # HP only matters when the agent converts it to Sheer Force; for a
        # pure-ATK rupture agent the HP buckets must NOT weaken the
        # dominance pruning (an HP-heavy disc adds nothing there). PEN is
        # absent: Sheer has no DEF zone, so a PEN-heavy disc is correctly
        # dominated.
        damage_indices = _SHEER_DAMAGE_INDICES + (
            _SHEER_HP_INDICES if agent.sheer_force_hp_conversion > 0 else ())
    else:
        damage_indices = _DAMAGE_INDICES
    # AP is inert for direct-hit / Sheer damage, but the constraint /
    # maximize-stat total must be the SAME number the other modes report
    # (engine passive and kit/external AP included) — an agent whose other
    # hit scales with AP is exactly why such a search would ask for it.
    stats = _StatModel(
        atk_base=agent_plus_engine_atk, kit_atk_pct=kit_atk_pct,
        kit_flat_atk=kit_flat_atk, hp_base=hp_base, core_hp_pct=core_hp_pct,
        kit_hp_mult=kit_hp_mult, def_base=agent.base_def,
        crit_rate_base=base_crit_rate, crit_rate_extra=crit_rate_const,
        crit_dmg_base=base_crit_dmg, crit_dmg_extra=crit_dmg_const,
        deferred_cd=tuple(deferred_cd), pen_ratio_base=pen_ratio_const,
        impact_base=agent.base_impact + agent.core_bonus_impact,
        er_base=agent.base_energy_regen + agent.core_bonus_energy_regen,
        ap_base=(agent.base_anomaly_proficiency
                 + agent.core_bonus_anomaly_proficiency
                 + engine.passive_ap(rank)
                 + kit["anomaly_proficiency"]
                 + config.external_anomaly_proficiency),
        am_base=agent.base_anomaly_mastery + agent.core_bonus_anomaly_mastery,
        sf_atk_conv=sf_atk_conv, sf_hp_conv=sf_hp_conv, sf_flat=flat_sf_const,
        deferred_atk=deferred_atk,
        am_gate=_am_gate(data["disc_sets"]),
    )
    return _ModeModel(damage_indices, evaluate, linearize, stats)


def _direct_model(config, objective, agent, attr, engine, boss, rank, data,
                  anomaly_data) -> _ModeModel:
    """Direct hits (mirrors :func:`~.api.calculate`)."""
    return _crit_hit_model(config, objective, agent, attr, engine, boss,
                           rank, data, sheer=False)


def _sheer_model(config, objective, agent, attr, engine, boss, rank, data,
                 anomaly_data) -> _ModeModel:
    """Sheer / Rupture damage (mirrors :func:`~.api.calculate_sheer`)."""
    return _crit_hit_model(config, objective, agent, attr, engine, boss,
                           rank, data, sheer=True)


def _anomaly_model(config, objective, agent, attr, engine, boss, rank, data,
                   anomaly_data) -> _ModeModel:
    """The anomaly family (mirrors :func:`~.api.calculate_anomaly` for the
    config's mode: plain anomaly / Abloom / Disorder / Polarity Disorder /
    Vortex / Luminize)."""
    consts = data["consts"]
    # --- Which anomaly / element is dealt (disc-independent; the baseline
    # already validated every mode input, so these lookups can't fail) -----
    if config.vortex_infused is not None:
        rule = anomaly_data.vortex[config.vortex_infused.lower()]
        kind, dealt_element, hits = "vortex", config.vortex_infused.lower(), 1
    elif config.disorder_replaced is not None:
        replaced_key = config.disorder_replaced.lower()
        rule = anomaly_data.disorder[replaced_key]
        kind, dealt_element, hits = "disorder", replaced_key, 1
    elif config.abloom:
        abloom_key = (config.abloom_element or attr).lower()
        anomaly = anomaly_data.anomalies[abloom_key]
        kind, dealt_element, hits = "anomaly", abloom_key, 1
    else:
        # An override Abloom may resolve another attribute's anomaly
        # (abloom_element) — mirrors calculate_anomaly.
        element_key = attr
        if (config.anomaly_mult_override is not None
                and config.abloom_element is not None):
            element_key = config.abloom_element.lower()
        anomaly = anomaly_data.anomalies[element_key]
        kind, dealt_element, hits = "anomaly", element_key, anomaly.hits
    if config.luminize is not None:
        # LUMINIZE: the instance keeps the stored anomaly's element (this
        # attacker's own attribute) but replaces its proc count — one per
        # stored Voidflare. The multiplier is substituted further down.
        hits = config.luminize.voidflares
    # Build Attribute DMG% (disc + set + engine advanced) only counts when
    # the burst deals the agent's OWN attribute — mirrors calculate_anomaly.
    # In Luminize mode it does apply: the DMG% comes from THIS agent, the
    # one whose Refringed anomaly the Voidflare stored.
    attr_dmg_applies = (dealt_element == attr)

    # Kit is computed with discs=[] so its DISC-dependent part (a set's auto
    # 4pc DMG%, e.g. Wuthering Salon) is excluded here and folded per combo
    # via set_tag_bonus instead — the rest of the kit is disc-independent.
    kit = _kit_contributions(
        agent, replace(config, discs=[]), data["agents"], data["disc_sets"],
        data["engines"], mode=kind, dealt_element=dealt_element,
    )
    # Duration extensions (Phoenix M2's +4 Burn ticks) — a plain proc only,
    # exactly as calculate_anomaly.
    if (kind == "anomaly" and not config.abloom
            and config.anomaly_mult_override is None
            and config.luminize is None and kit["anomaly_hits_add"]):
        hits += int(round(kit["anomaly_hits_add"]))

    agent_plus_engine_atk = agent.total_base_atk() + engine.base_atk
    kit_atk_pct = kit["atk_pct"]
    kit_flat_atk = kit["flat_atk"]
    pen_ratio_const = kit["pen_ratio"] + agent.core_bonus_pen_ratio
    def_red_const = kit["def_red"]
    ap_const = (agent.base_anomaly_proficiency
                + agent.core_bonus_anomaly_proficiency
                + engine.passive_ap(rank)
                + kit["anomaly_proficiency"]
                + config.external_anomaly_proficiency
                # Engine conditional flat AP (Weeping Gemini's stacks) —
                # constant here, exactly as in calculate_anomaly's segments.
                + _engine_buff_bonus(engine, config.engine_buff_stacks,
                                     dealt_element, rank, mode=kind,
                                     bracket="anomaly_proficiency"))
    bonus_const = (
        sum(config.external_dmg_bonuses)
        + sum(_engine_passive_bonuses(engine, dealt_element, rank))
        + _engine_buff_bonus(engine, config.engine_buff_stacks,
                             dealt_element, rank, mode=kind)
        + sum(kit["dmg_bonus"])
    )

    # Separate Anomaly/Disorder "Buff Multiplier" bracket (constant).
    # Luminize takes it from the PROVIDER instead — the attacker's own
    # anomaly-DMG sources never reach it (mirrors calculate_anomaly).
    if config.luminize is not None:
        abuff_entries = list(config.luminize.anomaly_dmg_bonuses)
    else:
        abuff_entries = list(config.external_anomaly_buff) + kit["anomaly_buff"]
        engine_abuff = _engine_buff_bonus(
            engine, config.engine_buff_stacks, dealt_element, rank,
            mode=kind, bracket="anomaly_buff",
        )
        if engine_abuff:
            abuff_entries.append(engine_abuff)
    abuff = formulas.anomaly_buff_mult(abuff_entries)

    level_mult = consts.anomaly_level_multiplier(agent.level)
    level_coef = consts.level_coefficient(agent.level)
    crit_cap = consts.crit_rate_cap
    luminize_res = (
        data["agents"][config.luminize.provider_key].luminize.m1_res_ignore
        if (config.luminize is not None and config.luminize.res_ignore)
        else 0.0)
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
    stun = formulas.stun_mult(
        objective.endswith("_stunned"), stun_base,
        list(config.external_daze_bonuses) + kit["daze_bonus"],
    )
    # "full" multiplies the per-proc value by the anomaly's proc count.
    hits_factor = hits if objective.startswith("full") else 1
    # Refringe (Lumiflux teammate) is a kit constant in its own zone —
    # disc-independent, so it folds straight into the outer product.
    refringe = formulas.refringe_mult([kit["refringe"]])
    # Anomaly crit (Phoenix's Sear): the objectives are the EXPECTED value,
    # and the rate/dmg are kit constants (her CRIT Rate reads the CLAIMED
    # initial AM, which _apply_scaling_input_minimums turned into a panel
    # minimum) — so the zone is one more disc-independent factor.
    anomaly_crit = formulas.crit_mult_average(
        kit["anomaly_crit_rate"], kit["anomaly_crit_dmg"], crit_cap)
    outer = (level_mult * res * taken * stun * hits_factor * refringe
             * anomaly_crit)

    # --- Per-proc / burst multiplier (constant, or AP-scaled per build) ----
    # ``mult_affine`` is the same multiplier as an affine function of the
    # buckets, for the linear bound (§17).
    disorder_additive = config.external_disorder_mult_add + kit["disorder_mult_add"]
    if config.luminize is not None:
        # Constant: it depends on the PROVIDER's AP, not on this build.
        provider = data["agents"][config.luminize.provider_key]
        luminize_mult = provider.luminize.multiplier(
            config.luminize.skill, config.luminize.anomaly_proficiency,
            m4=config.luminize.m4)
        def mult_of_ap(ap: float) -> float:
            return luminize_mult
        mult_affine = affine(luminize_mult)
    elif kind == "vortex":
        const_mult = formulas.burst_conversion_mult(
            rule.mult, rule.time_mult, rule.window,
            config.disorder_elapsed_seconds, extra_mult=disorder_additive,
        )
        def mult_of_ap(ap: float) -> float:
            return const_mult
        mult_affine = affine(const_mult)
    elif kind == "disorder" and config.polarity_disorder:
        base_disorder = formulas.burst_conversion_mult(
            rule.base, rule.time_mult, rule.window,
            config.disorder_elapsed_seconds, extra_mult=0.0,
        )
        polarity = anomaly_data.polarity
        polarity_coeff = polarity.coefficient(config.polarity_special_level)
        def mult_of_ap(ap: float) -> float:
            return (polarity.fraction * base_disorder
                    + disorder_additive + polarity_coeff * (ap / 100.0))
        mult_affine = affine(
            polarity.fraction * base_disorder + disorder_additive
            + polarity_coeff * ap_const / 100.0,
            (_B_AP, polarity_coeff / 100.0))
    elif kind == "disorder":
        const_mult = formulas.burst_conversion_mult(
            rule.base, rule.time_mult, rule.window,
            config.disorder_elapsed_seconds, extra_mult=disorder_additive,
        )
        def mult_of_ap(ap: float) -> float:
            return const_mult
        mult_affine = affine(const_mult)
    elif config.abloom:
        mv = anomaly_data.abloom.mv_per_10_ap[dealt_element]
        m2 = (anomaly_data.abloom.m2_factor if config.mindscapes.get(2)
              else 1.0)
        abloom_base = anomaly.mult * (mv / 100.0) * m2
        def mult_of_ap(ap: float) -> float:
            return abloom_base * (ap / 10.0)
        mult_affine = affine(abloom_base * ap_const / 10.0,
                             (_B_AP, abloom_base / 10.0))
    else:
        const_mult = (config.anomaly_mult_override
                      if config.anomaly_mult_override is not None
                      else anomaly.mult)
        def mult_of_ap(ap: float) -> float:
            return const_mult
        mult_affine = affine(const_mult)

    def evaluate(buckets: list[float], tagged: tuple[float, float]):
        """Fast path: the objective value of one bucket total."""
        atk_pre_combat = (
            agent_plus_engine_atk * (1.0 + buckets[_B_ATK_PCT])
            + buckets[_B_ATK_FLAT]
        )
        atk_total = (
            atk_pre_combat
            * (1.0 + buckets[_B_COMBAT_ATK_PCT] + kit_atk_pct)
            + kit_flat_atk
        )
        ap_total = ap_const + buckets[_B_AP]
        bonus = 1.0 + bonus_const + tagged[0] + buckets[_B_GEN_DMG]
        if attr_dmg_applies:
            bonus += buckets[_B_ATTR_DMG]
        # A 4pc part in the separate Anomaly Buff bracket (Feathered Fate
        # on a Lumiflux wearer) joins abuff, not the DMG% bracket.
        set_abuff = abuff + tagged[1]
        eff_def = formulas.effective_def(
            boss.base_def, buckets[_B_PEN_RATIO] + pen_ratio_const,
            buckets[_B_PEN_FLAT], def_red=def_red_const,
        )
        defense = level_coef / (level_coef + eff_def)
        value = (mult_of_ap(ap_total) * atk_total * formulas.ap_mult(ap_total)
                 * bonus * set_abuff * defense * outer)
        return value, None

    def linearize(x0, lo, hi):
        """``(c0, w)`` with ``log(evaluate) <= c0 + w · x`` on the box
        (§17): the anomaly product is mult(AP) × ATK × AP/100 × DMG% ×
        Anomaly Buff × DEF × the constant outer zones, and every factor
        but ATK and DEF is affine in the buckets."""
        if outer <= 0:
            return None
        lb = LogLinear(_N_LINEAR)
        try:
            lb.add_const(math.log(outer) - math.log(100.0))
            lb.add_log_affine(mult_affine, x0)
            lb.add_log_affine(affine(ap_const, (_B_AP, 1.0)), x0)
            u, v = _atk_factors(agent_plus_engine_atk, kit_atk_pct)
            lb.add_log_product(u, v, kit_flat_atk, x0, lo, hi)
            attr_term = ((_B_ATTR_DMG, 1.0),) if attr_dmg_applies else ()
            lb.add_log_affine(affine(1.0 + bonus_const, (_T_DMG, 1.0),
                                     (_B_GEN_DMG, 1.0), *attr_term), x0)
            lb.add_log_affine(affine(abuff, (_T_ANOM, 1.0)), x0)
            _add_def_zone(lb, boss.base_def, def_red_const, pen_ratio_const,
                          level_coef, lo, hi)
        except LinearBoundError:
            return None
        return lb.c0, lb.w

    # HP/DEF/Impact/Energy-Regen/CRIT totals are read only by the min-stat
    # constraints (inert for anomaly damage; anomaly crit comes from the
    # kit, never from disc CRIT).
    stats = _StatModel(
        atk_base=agent_plus_engine_atk, kit_atk_pct=kit_atk_pct,
        kit_flat_atk=kit_flat_atk, hp_base=agent.total_base_hp(),
        core_hp_pct=agent.core_bonus_hp_pct,
        kit_hp_mult=1.0 + kit["hp_pct"], def_base=agent.base_def,
        crit_rate_base=consts.base_crit_rate + agent.core_bonus_crit_rate,
        crit_rate_extra=config.external_crit_rate + kit["crit_rate"],
        crit_dmg_base=consts.base_crit_dmg + agent.core_bonus_crit_dmg,
        crit_dmg_extra=config.external_crit_dmg + kit["crit_dmg"],
        deferred_cd=(), pen_ratio_base=pen_ratio_const,
        impact_base=agent.base_impact + agent.core_bonus_impact,
        er_base=agent.base_energy_regen + agent.core_bonus_energy_regen,
        ap_base=ap_const,
        am_base=agent.base_anomaly_mastery + agent.core_bonus_anomaly_mastery,
        am_gate=_am_gate(data["disc_sets"]),
    )
    damage_indices = (_ANOMALY_DAMAGE_INDICES if attr_dmg_applies
                      else _ANOMALY_DAMAGE_INDICES_NO_ATTR)
    return _ModeModel(damage_indices, evaluate, linearize, stats,
                      anomaly_sets=True)


def _optimize(
    config: CalcConfig,
    options: OptimizeOptions,
    *,
    objectives: tuple,
    constraint_stats: tuple,
    calc_name: str,
    calc,
    build_model,
    consts: Constants | None,
    disc_data: DiscData | None,
    bosses: dict[str, Boss] | None,
    agents: dict[str, Agent] | None,
    engines: dict[str, Engine] | None,
    disc_sets: dict[str, DiscSet] | None,
    anomaly_data: AnomalyData | None,
    user_discs: dict[str, Disc] | None,
    _prune: bool,
    _bound: bool,
) -> OptimizeResult:
    """The optimizer every mode shares (plan §2-§6, §11, §15, §17).

    ``calc(config, data)`` is the mode's exact calculation (named
    ``calc_name`` in the §4 re-check error) and ``build_model`` its
    :class:`_ModeModel` factory; everything else is identical across
    modes. See :func:`optimize` for the contract.
    """
    required = _validate_options(options, objectives, constraint_stats)

    consts = consts if consts is not None else load_constants()
    disc_data = disc_data if disc_data is not None else load_disc_data()
    bosses = bosses if bosses is not None else load_bosses()
    agents = agents if agents is not None else load_agents()
    engines = engines if engines is not None else load_engines()
    disc_sets = disc_sets if disc_sets is not None else load_disc_sets()
    if user_discs is None:
        user_discs = load_user_discs(disc_data)

    data = dict(consts=consts, disc_data=disc_data, bosses=bosses,
                agents=agents, engines=engines, disc_sets=disc_sets)

    for key in required:
        if key not in disc_sets:
            raise OptimizeError(
                f"Unknown required set '{key}'; options: {sorted(disc_sets)}"
            )
    _validate_slot_mains(options, disc_data)

    # --- Baseline: validates the whole config (discs, mode inputs, agent
    # specialty) — the same errors the calculation would raise ------------
    baseline_results = calc(config, data)
    equipped = {d.slot: d for d in config.discs}
    agent = agents[config.agent_key]
    # Lumiflux (Attribute Mutation): the element this agent's damage
    # is dealt as — her own attribute for everyone else.
    attr = attack_element(agent, config)
    # Team-element gate (§18): 4pc effects no squad member can trigger
    # count 0 stacks in EVERY build. The baseline's own entered stacks of
    # such a set are dropped too (and the baseline re-run), so it is
    # valued on the same footing as the candidates it is compared with.
    gated = _gated_sets(disc_sets, _squad_elements(attr, config, agents))
    if any(key in gated for key in config.set_stacks):
        config = replace(config, set_stacks={
            key: stacks for key, stacks in config.set_stacks.items()
            if key not in gated
        })
        baseline_results = calc(config, data)
    baseline_value = getattr(baseline_results, options.objective)
    # Self-scaling kit inputs (the MAIN's own panel values) become
    # minimum-stat constraints — the buff amount is a constant, so no
    # build may fall short of the panel the user claimed.
    options = _apply_scaling_input_minimums(agent, config, options,
                                            constraint_stats)

    # --- Constant factors (disc-independent; the baseline validated them) -
    engine_key = (config.engine_key if config.engine_key is not None
                  else agent.default_engine)
    engine = engines[engine_key]
    boss = bosses[config.boss_name]
    rank = _resolve_engine_rank(engine, config.engine_rank)
    model = build_model(config, options.objective, agent, attr, engine, boss,
                        rank, data, anomaly_data)
    evaluate = model.evaluate
    stats = model.stats

    def stat_total(stat: str, buckets: list[float], _aux=None) -> float:
        """One constraint / maximize stat's final total for a bucket
        vector (the totals the results panel reports)."""
        return stats.total(stat, buckets)

    # --- Candidates per slot (plan §2): inventory + equipped virtuals -----
    watched = _watched_stats(options)
    # A stat-gated 4pc (Branch & Blade Song's AM >= 115) makes Anomaly
    # Mastery decide damage whenever such a set can be worn: an AM disc
    # must not be pruned as dominated, or the gate could never flip.
    gated_sets = {key for key, entry in disc_sets.items()
                  if entry.bonus_4pc is not None
                  and entry.bonus_4pc.base_min_stat is not None}
    if gated_sets and any(
            d.disc_set in gated_sets
            for d in (*user_discs.values(), *equipped.values())):
        watched.setdefault("Anomaly Mastery", 0.0)
    dominance_indices = _dominance_indices(model.damage_indices, watched)
    candidates, pruned_count = _gather_candidates(
        equipped, user_discs, disc_data, attr, options,
        dominance_indices, _prune,
    )

    slots = sorted(candidates)

    # Required sets: enough slots must hold a piece — each set on its own,
    # and both together (one piece per slot; Hall's condition, exact for
    # two sets).
    holders = {
        key: {slot for slot in slots
              if any(c.disc.disc_set == key for c in candidates[slot])}
        for key in required
    }
    for key, pieces in required.items():
        if len(holders[key]) < pieces:
            raise OptimizeError(
                f"Cannot build {pieces} pieces of '{disc_sets[key].name}': "
                f"only {len(holders[key])} slot(s) have a saved piece of it "
                f"(locked slots included)"
            )
    if len(required) > 1:
        either = set().union(*holders.values())
        if len(either) < sum(required.values()):
            names = " + ".join(f"{pieces}pc {disc_sets[key].name}"
                               for key, pieces in required.items())
            raise OptimizeError(
                f"Cannot build {names} at once: only {len(either)} "
                f"slot(s) have a saved piece of either (locked slots "
                f"included)"
            )

    base_buckets = _bucket_vector(engine.advanced_stat)
    panel_consts = _panel_constants(agent, engine, consts)
    feasible = _make_feasible(options, panel_consts, stat_total)

    # --- Per-set precomputation (§5 policy) --------------------------------
    baseline_counts: dict[str, int] = {}
    for disc in config.discs:
        if disc.disc_set is not None:
            baseline_counts[disc.disc_set] = (
                baseline_counts.get(disc.disc_set, 0) + 1
            )

    def assumed_stacks(key: str) -> int:
        entry = disc_sets[key]
        # Wearer-side modeled 4pc: the classic bonus, or a squad-facing
        # part (the wearer is a squad member too — King of the Summit).
        max_stacks = set_4pc_max_stacks(entry)
        if not max_stacks or key in gated:
            return 0
        if baseline_counts.get(key, 0) >= 4:
            return config.set_stacks.get(key, 0)
        return max_stacks if options.set_assumption == "max" else 0

    set_2pc: dict[str, tuple[float, ...]] = {}
    set_4pc: dict[str, tuple[float, ...]] = {}
    set_stacks_used: dict[str, int] = {}
    set_tag_bonus: dict[str, tuple[float, float]] = {}
    candidate_sets = {
        c.disc.disc_set
        for slot_list in candidates.values() for c in slot_list
        if c.disc.disc_set is not None
    }
    teammate_present = bool(config.supports)
    for key in candidate_sets:
        entry = disc_sets[key]
        # Anomaly hits carry no skill tag, so no tag-gated 2pc DMG% (Shadow
        # Harmony) applies there.
        set_2pc[key] = (
            _set_2pc_buckets(entry, attr) if model.anomaly_sets
            else _set_2pc_buckets(entry, attr, config.skill_tag,
                                  config.counts_as_aftershock))
        stacks = assumed_stacks(key)
        set_stacks_used[key] = stacks
        # Wearer-side 4pc stats at the assumed stacks (mirrors
        # set_bonus_stats: classic bonus incl. the at-max extra — Yunkui
        # Tales' Sheer DMG% lands in _B_SHEER_DMG, which only the sheer
        # fast path reads — or a squad-facing part mapped to a build stat),
        # incl. the unconditional base and the attribute gate. A stat-gated
        # base (Branch & Blade Song) goes to its own bucket, credited only
        # when the combination clears the gate (_StatModel.gated_crit_dmg).
        vector = list(_bucket_vector(set_4pc_stats(entry, stacks, attr)))
        if (entry.bonus_4pc is not None
                and entry.bonus_4pc.base_min_stat is not None):
            vector[_B_GATED_CD] += sum(entry.bonus_4pc.base.values())
        set_4pc[key] = tuple(vector)
        # 4pc DMG% additives in the ordinary bonus bracket, folded per combo:
        # skill-tag-gated ones (Puffer Electro's Ultimate +20%; a windowed
        # one only while its config toggle is on) and a set's
        # auto 4pc DMG% (Wuthering Salon +18%; Phaethon's Melody needs a
        # teammate) — the auto part mirrors _kit_contributions, which is why
        # the mode's kit excludes it (discs=[]).
        toggles = config.set_toggles.get(key, {})
        tag_skill = sum(
            b.value for b in entry.bonus_4pc_dmg
            if (b.toggle is None or toggles.get(b.toggle))
            and ((config.skill_tag is not None
                  and b.skill_tag == config.skill_tag)
                 or (config.counts_as_aftershock
                     and b.skill_tag == "aftershock"))
        )
        auto = 0.0
        if entry.auto_4pc_dmg and (
                not entry.auto_4pc_dmg_needs_teammate or teammate_present):
            auto = entry.auto_4pc_dmg
        # Second channel (anomaly modes): a 4pc part in the separate
        # Anomaly Buff bracket, optionally gated to the wearer's attribute
        # (Feathered Fate is Lumiflux-only). Mirrors _kit_contributions
        # exactly, or the section-4 re-check fails.
        anom = 0.0
        if model.anomaly_sets and entry.auto_4pc_anomaly_dmg and (
                entry.auto_4pc_attribute is None
                or entry.auto_4pc_attribute == agent.attribute):
            anom = entry.auto_4pc_anomaly_dmg
        set_tag_bonus[key] = (tag_skill + auto, anom)

    def combo_set_stacks(counts: dict[str, int]) -> dict[str, int]:
        """The ``CalcConfig.set_stacks`` this combination is evaluated at."""
        return {
            key: set_stacks_used[key]
            for key, count in counts.items()
            if count >= 4 and set_stacks_used[key]
        }

    # --- Branch-and-bound DFS (§11 E2, §17), shared search core -----------
    outcome = _search_maximizing(
        candidates, slots, equipped, base_buckets, evaluate, feasible,
        stat_total, set_2pc, set_4pc, set_tag_bonus, candidate_sets,
        required, options, _bound, model.linearize, stats,
    )
    heap, evals, exhausted = outcome.heap, outcome.evals, outcome.exhausted

    # --- Baseline feasibility (constraints may exclude the baseline) ------
    # After the search: a maximize-stat run's damage floor (§15 E5) is
    # only known once the first pass has found the ceiling.
    base_bb = list(base_buckets)
    for disc in config.discs:
        for k, v in enumerate(_disc_buckets(disc, disc_data, attr)):
            base_bb[k] += v
    _fold_sets(base_bb, baseline_counts, set_2pc, set_4pc, set_tag_bonus)
    _, baseline_aux = evaluate(base_bb, (0.0, 0.0))
    baseline_stat = (stat_total(options.maximize_stat, base_bb)
                     if options.maximize_stat else None)
    baseline_shortfalls: list = []
    baseline_feasible = (
        outcome.check_baseline(base_bb, baseline_aux, baseline_value,
                               baseline_shortfalls)
        and all(baseline_counts.get(key, 0) >= pieces
                for key, pieces in required.items())
        and _baseline_mains_ok(equipped, options.slot_main_stats,
                               attr)
        and (not options.sets_only or _sets_coverage_ok(config.discs))
    )

    def verify(combo: tuple[_Candidate, ...]) -> BuildOption:
        """Re-run one build through the calculation (exactness, §4)."""
        discs = [c.disc for c in combo]
        buckets, tagged, counts = _combo_buckets(
            combo, base_buckets, set_2pc, set_4pc, set_tag_bonus)
        fast_value, _aux = evaluate(buckets, tagged)
        stacks = combo_set_stacks(counts)
        results = calc(replace(config, discs=discs, set_stacks=stacks), data)
        exact = getattr(results, options.objective)
        if abs(fast_value - exact) > _RECHECK_RTOL * max(1.0, abs(exact)):
            raise OptimizeError(
                f"Internal error: fast evaluation ({fast_value!r}) disagrees "
                f"with {calc_name}() ({exact!r}) — please report this build"
            )
        return BuildOption(
            value=exact,
            delta=(exact - baseline_value) / baseline_value
            if baseline_value else 0.0,
            discs=tuple(discs),
            disc_ids=tuple(c.disc_id for c in combo),
            set_stacks=stacks,
            changed_slots=tuple(
                c.disc.slot for c in combo
                if equipped.get(c.disc.slot) != c.disc
            ),
            results=results,
            stat_value=(stat_total(options.maximize_stat, buckets)
                        if options.maximize_stat else None),
        )

    # Highest value first; on exact ties the earlier build wins, and the
    # baseline is the first build seeded, so a tie resolves toward the
    # current discs. Greedy seeds the DFS found again are de-duplicated.
    ranked = sorted(heap, key=lambda item: (-item[0], -item[1]))
    seen_combos: set = set()
    unique = []
    for value, _neg_counter, combo in ranked:
        combo_key = tuple(id(c) for c in combo)
        if combo_key in seen_combos:
            continue
        seen_combos.add(combo_key)
        unique.append((value, combo))
    options_verified = [verify(combo) for _value, combo in unique]

    if not options_verified and not baseline_feasible:
        raise _infeasible_error(
            _baseline_reasons(
                baseline_shortfalls,
                discs=config.discs,
                equipped=equipped,
                requirements=required,
                baseline_counts=baseline_counts,
                disc_sets=disc_sets,
                slot_main_stats=options.slot_main_stats,
                attribute=attr,
                sets_only=options.sets_only,
            ),
            exhausted, options.combo_budget,
        )

    best = options_verified[0] if options_verified else None
    # Compared by the SEARCH's ranking key — damage, or (maximize mode)
    # the stat with damage as the tie-break. Comparing damage there would
    # call the baseline a winner against a build that deliberately traded
    # a little damage for more of the maximized stat.
    baseline_rank = outcome.rank(baseline_value, baseline_stat)
    baseline_wins = baseline_feasible and (
        best is None
        or outcome.rank(best.value, best.stat_value)
        <= baseline_rank * (1.0 + _RECHECK_RTOL)
    )
    # A budget-exhausted search proves nothing about optimality: the
    # baseline may still be reported as the best FOUND, but never as
    # "already optimal".
    already_optimal = baseline_wins and not exhausted
    if baseline_wins:
        best = BuildOption(
            value=baseline_value,
            delta=0.0,
            discs=tuple(config.discs),
            disc_ids=tuple(
                next((i for i, d in user_discs.items() if d == disc), None)
                for disc in config.discs
            ),
            set_stacks=dict(config.set_stacks),
            changed_slots=(),
            results=baseline_results,
            stat_value=baseline_stat,
        )
    # Runners-up: every verified build that actually differs from the
    # baseline and isn't the reported best.
    alternatives = tuple(
        option for option in options_verified
        if option is not best and option.changed_slots
    )[: options.top_n - 1]

    return OptimizeResult(
        objective=options.objective,
        set_assumption=options.set_assumption,
        baseline_value=baseline_value,
        baseline_feasible=baseline_feasible,
        already_optimal=already_optimal,
        best=best,
        alternatives=alternatives,
        min_stats=dict(options.min_stats),
        required_4pc=options.required_4pc,
        required_2pc=options.required_2pc,
        slot_main_stats=dict(options.slot_main_stats),
        sets_only=options.sets_only,
        budget_exhausted=exhausted,
        min_panel_stats=dict(options.min_panel_stats),
        maximize_stat=options.maximize_stat,
        maximize_tolerance=(options.maximize_tolerance
                            if options.maximize_stat else 0.0),
        damage_ceiling=outcome.ceiling,
        damage_floor=outcome.floor,
        combos_evaluated=evals,
        candidates_per_slot={s: len(candidates[s]) for s in slots},
        discs_pruned=pruned_count,
        # Only the gated sets this search could actually have worn.
        gated_sets={
            key: element for key, element in gated.items()
            if key in candidate_sets or key in baseline_counts
            or key in required
        },
    )


def optimize(
    config: CalcConfig,
    options: OptimizeOptions | None = None,
    *,
    consts: Constants | None = None,
    disc_data: DiscData | None = None,
    bosses: dict[str, Boss] | None = None,
    agents: dict[str, Agent] | None = None,
    engines: dict[str, Engine] | None = None,
    disc_sets: dict[str, DiscSet] | None = None,
    user_discs: dict[str, Disc] | None = None,
    _prune: bool = True,
    _bound: bool = True,
) -> OptimizeResult:
    """Search every legal combination of the user's discs for the best build.

    ``config`` is the exact configuration the user just calculated; only
    its ``discs`` (and, per the §5 policy, ``set_stacks``) vary during the
    search. The keyword data arguments allow the UI server/tests to inject
    already-loaded data, same as :func:`~.api.calculate`. ``_prune`` /
    ``_bound`` exist for tests proving dominance pruning and the
    branch-and-bound cut never change the answer.

    Raises:
        OptimizeError: unknown objective/assumption/constraint, bad
            options, a search exceeding ``options.combo_budget``
            evaluations, or ``min_stats`` no combination can meet.
        CalcError / DiscError / AgentError: the baseline config itself is
            invalid (same errors ``calculate`` would raise).
    """
    return _optimize(
        config, options if options is not None else OptimizeOptions(),
        objectives=OBJECTIVES, constraint_stats=CONSTRAINT_STATS,
        calc_name="calculate",
        calc=lambda cfg, data: calculate(cfg, **data),
        build_model=_direct_model,
        consts=consts, disc_data=disc_data, bosses=bosses, agents=agents,
        engines=engines, disc_sets=disc_sets, anomaly_data=None,
        user_discs=user_discs, _prune=_prune, _bound=_bound,
    )


def optimize_anomaly(
    config: CalcConfig,
    options: OptimizeOptions | None = None,
    *,
    consts: Constants | None = None,
    disc_data: DiscData | None = None,
    bosses: dict[str, Boss] | None = None,
    agents: dict[str, Agent] | None = None,
    engines: dict[str, Engine] | None = None,
    disc_sets: dict[str, DiscSet] | None = None,
    anomaly_data: AnomalyData | None = None,
    user_discs: dict[str, Disc] | None = None,
    _prune: bool = True,
    _bound: bool = True,
) -> OptimizeResult:
    """Best-build search over the saved inventory for an ANOMALY objective.

    The anomaly-mode twin of :func:`optimize`: same disc-only search, same
    §5 set policy, same min-stat / required-4pc constraints, same exactness
    re-check — but it maximizes one of :data:`ANOMALY_OBJECTIVES`
    (proc/tick or full-duration, normal or stunned) and its fast path
    reproduces :func:`~.api.calculate_anomaly` for the config's mode (plain
    anomaly / Abloom / Disorder / Polarity Disorder / Vortex).

    Anomaly damage is monotone in every tracked bucket (ATK, PEN,
    Attribute DMG%, and — unlike direct hits — Anomaly Proficiency; the
    AP-scaled burst mults of Abloom and Polarity Disorder only *increase*
    with AP), so the shared branch-and-bound core is valid unchanged. Every
    reported build is re-run through ``calculate_anomaly`` and must match to
    1e-9, so the number is exactly what the user gets after equipping it.

    The optimizer targets the **fully-buffed** anomaly number (the "optimal"
    end of the UI's floor–optimal range): buildup dilution is not part of
    the objective. A config carrying ``buildup_segments`` or a non-zero
    ``unbuffed_share`` is rejected — those describe a weighted buildup, not a
    single build to optimize.

    Raises:
        OptimizeError: bad options/objective, buildup dilution present, an
            over-budget search, or constraints no combination can meet.
        CalcError / DiscError / AgentError: the baseline config itself is
            invalid (same errors ``calculate_anomaly`` would raise).
    """
    options = options if options is not None else OptimizeOptions(objective="full")
    _validate_options(options, ANOMALY_OBJECTIVES)
    if config.buildup_segments or config.unbuffed_share:
        raise OptimizeError(
            "optimize_anomaly targets the fully-buffed anomaly number; "
            "buildup segments / unbuffed_share (buildup dilution) are not "
            "supported — clear them and optimize again"
        )
    anomaly_data = anomaly_data if anomaly_data is not None else load_anomalies()
    return _optimize(
        config, options,
        objectives=ANOMALY_OBJECTIVES, constraint_stats=CONSTRAINT_STATS,
        calc_name="calculate_anomaly",
        calc=lambda cfg, data: calculate_anomaly(cfg, **data,
                                                 anomaly_data=anomaly_data),
        build_model=_anomaly_model,
        consts=consts, disc_data=disc_data, bosses=bosses, agents=agents,
        engines=engines, disc_sets=disc_sets, anomaly_data=anomaly_data,
        user_discs=user_discs, _prune=_prune, _bound=_bound,
    )


def optimize_sheer(
    config: CalcConfig,
    options: OptimizeOptions | None = None,
    *,
    consts: Constants | None = None,
    disc_data: DiscData | None = None,
    bosses: dict[str, Boss] | None = None,
    agents: dict[str, Agent] | None = None,
    engines: dict[str, Engine] | None = None,
    disc_sets: dict[str, DiscSet] | None = None,
    user_discs: dict[str, Disc] | None = None,
    _prune: bool = True,
    _bound: bool = True,
) -> OptimizeResult:
    """Best-build search over the saved inventory for SHEER (Rupture) damage.

    The sheer-mode twin of :func:`optimize`: same disc-only search, same
    §5 set policy, same min-stat / required-4pc constraints, same exactness
    re-check — but its fast path reproduces :func:`~.api.calculate_sheer`.
    Sheer crits like a direct hit, so it maximizes the same
    :data:`OBJECTIVES`; the deltas vs the direct fast path are the base
    stat (Sheer Force = ATK × the global conversion + HP × the agent's HP
    conversion + flat SF, so the HP buckets matter), the dedicated
    Sheer DMG% bracket, and the absent DEF zone (defense = 1.0 — PEN does
    nothing, which the dominance indices encode by omission).

    Sheer damage is monotone in every tracked bucket (ATK, HP, CRIT,
    DMG%, Sheer DMG% all only increase it), so the shared branch-and-bound
    core is valid unchanged. Every reported build is re-run through
    ``calculate_sheer`` and must match to 1e-9.

    Constraints come from :data:`SHEER_CONSTRAINT_STATS` — the shared
    base-stat list plus the finished Sheer Force panel
    :class:`~.api.SheerResults` reports.

    Raises:
        OptimizeError: bad options/objective/constraint, an over-budget
            search, or constraints no combination can meet.
        CalcError / DiscError / AgentError: the baseline config itself is
            invalid (same errors ``calculate_sheer`` would raise, including
            a non-rupture agent).
    """
    return _optimize(
        config, options if options is not None else OptimizeOptions(),
        objectives=OBJECTIVES, constraint_stats=SHEER_CONSTRAINT_STATS,
        calc_name="calculate_sheer",
        calc=lambda cfg, data: calculate_sheer(cfg, **data),
        build_model=_sheer_model,
        consts=consts, disc_data=disc_data, bosses=bosses, agents=agents,
        engines=engines, disc_sets=disc_sets, anomaly_data=None,
        user_discs=user_discs, _prune=_prune, _bound=_bound,
    )
