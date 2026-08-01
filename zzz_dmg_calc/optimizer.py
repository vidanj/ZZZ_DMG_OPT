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

Search (plan §6 + §11 E2): per-slot dominance pruning, then an exact
**branch-and-bound** depth-first search. Damage is monotone in every
stat bucket, so a partial build whose optimistic upper bound (component-
wise max of everything the remaining slots could still add, plus every
candidate set bonus at once) cannot beat the current top-N threshold is
skipped whole. The winner and every reported runner-up are re-run
through :func:`~.api.calculate` and must match to 1e-9 (the exactness
guarantee, plan §4), so the reported numbers are exactly what the user
gets after equipping the build.

Set-bonus policy per candidate combination (plan §5): 2-piece bonuses
always apply; a modeled 4-piece effect uses the user's entered stacks
when the set is also 4-piece in the baseline build, and otherwise —
depending on ``OptimizeOptions.set_assumption`` — ``max_stacks``
(``"max"``, default) or 0 (``"current"``).

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
from dataclasses import dataclass, field, replace
from itertools import product

from . import formulas
from .agent import Agent, load_agents
from .anomalies import AnomalyData, load_anomalies
from .api import (
    POLARITY_DISORDER_FRACTION, VIVIAN_ABLOOM_M2_FACTOR, VIVIAN_ABLOOM_MV,
    AnomalyResults, CalcConfig, CalcResults, attack_element, calculate,
    calculate_anomaly, calculate_sheer, _engine_buff_bonus,
    _engine_passive_bonuses, _kit_contributions, _resolve_engine_rank,
)
from .constants import Constants, load_constants
from .discs import (
    Disc, DiscData, DiscSet, disc_stats, load_disc_data, load_disc_sets,
    load_user_discs, set_4pc_max_stacks, set_4pc_stats,
)
from .enemies import Boss, load_bosses
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
    _B_DEF_PCT, _B_DEF_FLAT, _B_IMPACT, _B_ER, _B_AM_PCT = range(18)

_N_BUCKETS = 18
_DAMAGE_INDICES = tuple(range(8))

#: Anomaly damage-relevant buckets: the ATK / PEN / Attribute-DMG% / AP
#: stats. CRIT (indices 3-4) never affects an anomaly (they cannot crit);
#: AP (index 8) DOES, unlike direct hits. ``_B_ATTR_DMG`` is included here
#: but dropped by the caller when the burst deals an element other than the
#: agent's own attribute (cross-element Disorder/Vortex): the fast path
#: mirrors ``calculate_anomaly``, which credits the build's Attribute DMG%
#: only when ``dealt_element == agent.attribute``.
_ANOMALY_DAMAGE_INDICES = (_B_ATK_PCT, _B_COMBAT_ATK_PCT, _B_ATK_FLAT,
                           _B_PEN_RATIO, _B_PEN_FLAT, _B_ATTR_DMG, _B_AP)
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
                         _B_SHEER_DMG)
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
    # agent._fold_stat), so it shares the _B_ATTR_DMG bucket.
    "DMG%": _B_ATTR_DMG,
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


class _BudgetExhausted(Exception):
    """Internal: the DFS hit the evaluation budget — the heap holds the
    best builds found so far (anytime behavior, not an error)."""


#: Pay for the tighter count-aware set bound only when the quick static
#: bound lands within this factor of the incumbent threshold — nodes far
#: above it descend anyway, so the second bound would be wasted work.
_TIGHT_BOUND_MARGIN = 1.5

#: Wide searches first run an INTERNAL fast pass capped at this many
#: candidates per slot (budget-sliced): it finds set-coherent optima the
#: greedy ascent cannot reach (completing a 4pc takes coordinated swaps),
#: giving the full DFS a near-optimal incumbent before the main budget.
_SEED_STAGE_CAP = 15


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
            slot (by solo value; the equipped disc and the required 4pc
            set survive the cut). APPROXIMATE by design: the true
            optimum may use a disc outside the cap. 0 = exact search.
        min_panel_stats: like ``min_stats``, but checked against the
            PANEL totals (agent base + core + engine + discs + set
            stats) — no kit/team/combat brackets. This is where the
            self-scaling kit inputs land automatically: Astra claiming
            ``initial_atk`` 3,430 must reach panel ATK 3,430, and the
            combat total (which contains the very flat ATK that claim
            grants — circular) must not satisfy it.
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
    """

    objective: str = "average"
    set_assumption: str = "max"
    locked_slots: frozenset[int] = frozenset()
    top_n: int = 5
    combo_budget: int = DEFAULT_COMBO_BUDGET
    min_stats: dict[str, float] = field(default_factory=dict)
    required_4pc: str | None = None
    slot_main_stats: dict[int, str] = field(default_factory=dict)
    sets_only: bool = False
    candidate_cap: int = 0
    min_panel_stats: dict[str, float] = field(default_factory=dict)
    maximize_stat: str | None = None
    maximize_tolerance: float = DEFAULT_MAXIMIZE_TOLERANCE


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


def _set_2pc_buckets(entry: DiscSet, attribute: str) -> tuple[float, ...]:
    """One set's 2-piece bucket vector for an agent attacking as
    ``attribute``.

    Mirrors ``set_bonus_stats``' element gate: an element-specific 2pc
    (Fanged Metal = Physical DMG +10%) is worth nothing when the damage
    is dealt as another element, so the optimizer must not value it —
    otherwise it recommends an off-element set for a phantom +10% (and
    the §4 re-check would fail, since ``calculate`` drops it).
    """
    stats = dict(entry.bonus_2pc)
    if (entry.bonus_2pc_attribute is not None
            and entry.bonus_2pc_attribute != attribute):
        stats.pop("Attribute DMG%", None)
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
) -> str | None:
    """Validate the pre-load parts of ``options`` (shared by the search
    entry points); return the ``required_4pc`` key (or None).

    Raises:
        OptimizeError: unknown objective/assumption, bad top_n/budget,
            invalid locked slots or min-stat constraints, or a malformed
            ``required_4pc``.
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
    required = options.required_4pc
    if required is not None and (
            not isinstance(required, str) or not required.strip()):
        raise OptimizeError("'required_4pc' must be a set key or None")
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
    return required


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
    team buffs — mirroring ``_kit_contributions``) and, for each that
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
    own_by_name = {b.name: b for b in agent.team_buffs}
    for name, stacks in config.own_team_buffs.items():
        buff = own_by_name.get(name)
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
    required: str | None,
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
    if required is not None and baseline_counts.get(required, 0) < 4:
        name = disc_sets[required].name if required in disc_sets else required
        reasons.append(
            f"it wears {baseline_counts.get(required, 0)} piece(s) of "
            f"{name}, not the required 4"
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
    required: str | None,
    pairs_only: bool,
    top_n: int,
    budget: int,
    bound: bool,
    slot_cap: int = 0,
) -> tuple[list, int, bool]:
    """Branch-and-bound DFS over the per-slot candidates (plan §11 E2).

    ``pairs_only`` (2pc priority): only accept leaves where every set
    worn has >= 2 pieces (the callers already excluded set-less
    candidates, so singleton pieces are the only thing to reject here).

    ``slot_cap`` > 0 truncates each slot to its best ``slot_cap``
    candidates by solo value (equipped disc always kept; the required
    4pc set keeps a piece per slot) — the user-facing "fast search",
    APPROXIMATE by design.

    Returns ``(heap, evals, exhausted)``; ``exhausted`` means the budget
    ran out and the heap holds the best builds found so far, with no
    completeness claim (anytime behavior — no exception).

    Mode-agnostic search core shared by :func:`optimize` (direct hits) and
    :func:`optimize_anomaly`. ``evaluate(buckets, tagged) -> (value, aux)``
    and ``feasible(buckets, aux) -> bool`` carry all the mode-specific
    damage math; everything here — candidate ordering, the suffix/static
    upper bounds, the count-aware set bound, the top-N heap and the
    evaluation budget — is identical across modes because anomaly damage is
    monotone in every tracked bucket, exactly like direct-hit damage.

    The heap holds ``(value, -counter, combo)`` tuples, ranked and
    de-duplicated by the caller (greedy seeds may also be found by the
    DFS).
    """
    n_buckets = len(base_buckets)

    # Candidate order: the equipped disc stays FIRST (the very first build
    # evaluated is the baseline, so exact ties resolve toward it); the
    # rest are sorted by their solo value so a strong incumbent forms
    # early and the bound cuts sooner.
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
    # index 0, and the required 4pc set keeps its best piece per slot so
    # the requirement can't become spuriously unreachable.
    if slot_cap:
        for slot in slots:
            full = candidates[slot]
            if len(full) <= slot_cap:
                continue
            kept = full[:slot_cap]
            if required is not None and not any(
                    c.disc.disc_set == required for c in kept):
                extra = next((c for c in full[slot_cap:]
                              if c.disc.disc_set == required), None)
                if extra is not None:
                    kept = kept + [extra]
            candidates[slot] = kept

    # suffix_max[i][k]: the most bucket k can still gain from slots[i:].
    suffix_max = [[0.0] * n_buckets for _ in range(len(slots) + 1)]
    for i in range(len(slots) - 1, -1, -1):
        for k in range(n_buckets):
            suffix_max[i][k] = suffix_max[i + 1][k] + max(
                c.buckets[k] for c in candidates[slots[i]]
            )
    # suffix_set_max[i][key]: how many MORE pieces of ``key`` slots[i:] can
    # still equip (at most one per slot) — bounds reachable set bonuses.
    suffix_set_max: list[dict[str, int]] = [{} for _ in range(len(slots) + 1)]
    for i in range(len(slots) - 1, -1, -1):
        counts = dict(suffix_set_max[i + 1])
        for key in {c.disc.disc_set for c in candidates[slots[i]]
                    if c.disc.disc_set is not None}:
            counts[key] = counts.get(key, 0) + 1
        suffix_set_max[i] = counts
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

    heap: list[tuple[float, int, tuple[_Candidate, ...]]] = []
    heap_keys: set[tuple[int, ...]] = set()   # combos currently in the heap
    counter = 0
    evals = 0
    probes = 0     # interior bound evaluations — same cost as a leaf

    def offer(value: float, combo: tuple[_Candidate, ...]) -> None:
        """Push one acceptable build into the top-N heap, deduplicated —
        a greedy seed the DFS finds again must not crowd out a genuine
        alternative."""
        nonlocal counter
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
    work_buckets = list(base_buckets)
    work_counts: dict[str, int] = {}
    combo_stack: list[_Candidate] = []
    slot_entries = [
        [(c, tuple((k, v) for k, v in enumerate(c.buckets) if v),
          c.disc.disc_set) for c in candidates[slot]]
        for slot in slots
    ]
    n_slots = len(slots)

    static_ub_vec = [
        [suffix_max[i][k] + static_set_ub[k] for k in range(n_buckets)]
        for i in range(len(slots) + 1)
    ]
    set_2pc_sparse = {
        key: tuple((k, v) for k, v in enumerate(vec) if v)
        for key, vec in set_2pc.items()
    }
    set_4pc_sparse = {
        key: tuple((k, v) for k, v in enumerate(vec) if v)
        for key, vec in set_4pc.items()
    }

    def node_set_ub(i: int) -> tuple[list[float], float]:
        """Set-bonus upper bound for completions of the current partial
        build: fold every set at the piece count it can still reach,
        then cap componentwise by the static top-3+4pc bound."""
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

    def combo_value(combo) -> tuple[float, bool]:
        """One full build's (value, acceptable) — the leaf math, reused
        by the greedy warm start below."""
        buckets = list(base_buckets)
        counts: dict[str, int] = {}
        for cand in combo:
            for k, v in enumerate(cand.buckets):
                buckets[k] += v
            key = cand.disc.disc_set
            if key is not None:
                counts[key] = counts.get(key, 0) + 1
        tagged = _fold_sets(buckets, counts, set_2pc, set_4pc, set_tag_bonus)
        value, aux = evaluate(buckets, tagged)
        acceptable = (
            feasible(buckets, aux)
            and (required is None or counts.get(required, 0) >= 4)
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

    def dfs(i: int) -> None:
        nonlocal evals, probes
        # Every node visit counts toward the budget (bound probes and
        # reachability walks cost real time too) — so WALL TIME is capped
        # even when the bounds prune almost every leaf.
        probes += 1
        if evals + probes >= budget:
            raise _BudgetExhausted
        # Required-4pc reachability: exact, not just a bound — a branch
        # that can no longer collect 4 pieces has no feasible leaves.
        if required is not None and (
            work_counts.get(required, 0)
            + suffix_set_max[i].get(required, 0) < 4
        ):
            return
        # 2pc-priority reachability: a singleton set no later slot can
        # complete has no acceptable leaves (exact, like the 4pc check).
        if pairs_only and any(
            count == 1 and not suffix_set_max[i].get(key)
            for key, count in work_counts.items()
        ):
            return
        if i == n_slots:
            if pairs_only and any(c < 2 for c in work_counts.values()):
                return
            if evals >= budget:
                # Anytime behavior: stop and report the best found so far.
                raise _BudgetExhausted
            evals += 1
            leaf = list(work_buckets)
            tagged = _fold_sets(leaf, work_counts,
                                set_2pc, set_4pc, set_tag_bonus)
            value, aux = evaluate(leaf, tagged)
            if not feasible(leaf, aux):
                return
            offer(value, tuple(combo_stack))
            return
        if bound:
            threshold = heap[0][0] if len(heap) == top_n else None
            static_vec = static_ub_vec[i]
            quick = [work_buckets[k] + static_vec[k]
                     for k in range(n_buckets)]
            quick_value, quick_aux = evaluate(quick, static_tag_ub)
            if not feasible(quick, quick_aux):
                return           # not even the optimistic completion fits
            if threshold is not None and quick_value <= threshold:
                return           # cannot beat the current top-N
            # The quick bound passed — pay for the tighter count-aware
            # set bound only when the quick bound is CLOSE to the
            # threshold (a node far above it descends anyway, so the
            # second bound would be wasted arithmetic).
            if (threshold is None
                    or quick_value <= threshold * _TIGHT_BOUND_MARGIN):
                set_ub, tag_ub = node_set_ub(i)
                suffix = suffix_max[i]
                ub = [work_buckets[k] + suffix[k] + set_ub[k]
                      for k in range(n_buckets)]
                ub_value, ub_aux = evaluate(ub, tag_ub)
                if not feasible(ub, ub_aux):
                    return
                if threshold is not None and ub_value <= threshold:
                    return
        for cand, sparse, key in slot_entries[i]:
            for k, v in sparse:
                work_buckets[k] += v
            if key is not None:
                work_counts[key] = work_counts.get(key, 0) + 1
            combo_stack.append(cand)
            dfs(i + 1)
            combo_stack.pop()
            if key is not None:
                work_counts[key] -= 1
                if not work_counts[key]:
                    del work_counts[key]
            for k, v in sparse:
                work_buckets[k] -= v

    exhausted = False
    if slots:
        seed_incumbents()
        # Two-stage warm start (wide searches only): the internal fast
        # pass runs this same function over the top candidates per slot
        # on a slice of the budget; its winners join the heap as seeds.
        if (not slot_cap and bound and any(
                len(candidates[slot]) > _SEED_STAGE_CAP for slot in slots)):
            stage_heap, stage_evals, _ = _search(
                {slot: list(candidates[slot]) for slot in slots},
                list(slots), equipped, base_buckets, evaluate, feasible,
                set_2pc, set_4pc, set_tag_bonus, candidate_sets, required,
                pairs_only, top_n, min(budget // 2, 2_500_000), True,
                slot_cap=_SEED_STAGE_CAP,
            )
            probes += stage_evals
            for value, _neg_counter, combo in sorted(stage_heap,
                                                     reverse=True):
                offer(value, combo)
    try:
        dfs(0)
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
    required: str | None,
    options: OptimizeOptions,
    bound: bool,
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
    def run(cands, ev, fe, budget):
        return _search(
            cands, slots, equipped, base_buckets, ev, fe,
            set_2pc, set_4pc, set_tag_bonus, candidate_sets, required,
            options.sets_only, options.top_n, budget, bound,
            slot_cap=options.candidate_cap,
        )

    heap, evals, exhausted = run(candidates, evaluate, feasible,
                                 options.combo_budget)

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

    # A fresh candidate dict: _search reorders (and, under a candidate
    # cap, truncates) the lists it is given.
    pass2 = {slot: list(candidates[slot]) for slot in slots}
    heap2, evals2, exhausted2 = run(pass2, evaluate_stat, feasible_stat,
                                    options.combo_budget)
    return _SearchOutcome(
        heap=heap2, evals=evals + evals2,
        exhausted=exhausted or exhausted2,
        # The baseline's damage is its VERIFIED value, so the floor is
        # judged against the same number the user was shown.
        check_baseline=lambda buckets, aux, damage, report=None:
            feasible_stat(buckets, (damage, aux), report),
        rank=rank, ceiling=ceiling, floor=floor,
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
    options = options if options is not None else OptimizeOptions()
    required = _validate_options(options, OBJECTIVES)

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

    if required is not None and required not in disc_sets:
        raise OptimizeError(
            f"Unknown required set '{required}'; options: {sorted(disc_sets)}"
        )
    _validate_slot_mains(options, disc_data)

    # --- Baseline: validates the whole config (discs included) ------------
    baseline_results = calculate(config, **data)
    baseline_value = getattr(baseline_results, options.objective)
    equipped = {d.slot: d for d in config.discs}
    agent = agents[config.agent_key]
    # Lumiflux (Attribute Mutation): the element this agent's damage
    # is dealt as — her own attribute for everyone else.
    attr = attack_element(agent, config)
    # Self-scaling kit inputs (the MAIN's own panel values) become
    # minimum-stat constraints — the buff amount is a constant, so no
    # build may fall short of the panel the user claimed.
    options = _apply_scaling_input_minimums(agent, config, options,
                                            CONSTRAINT_STATS)

    # --- Candidates per slot (plan §2): inventory + equipped virtuals -----
    dominance_indices = _dominance_indices(
        _DAMAGE_INDICES, _watched_stats(options))
    candidates, pruned_count = _gather_candidates(
        equipped, user_discs, disc_data, attr, options,
        dominance_indices, _prune,
    )

    slots = sorted(candidates)

    if required is not None:
        equippable = sum(
            1 for slot in slots
            if any(c.disc.disc_set == required for c in candidates[slot])
        )
        if equippable < 4:
            raise OptimizeError(
                f"Cannot build 4 pieces of '{disc_sets[required].name}': "
                f"only {equippable} slot(s) have a saved piece of it "
                f"(locked slots included)"
            )

    # --- Constant factors (disc-independent, mirrors api.calculate) -------
    engine_key = (config.engine_key if config.engine_key is not None
                  else agent.default_engine)
    engine = engines[engine_key]
    boss = bosses[config.boss_name]
    rank = _resolve_engine_rank(engine, config.engine_rank)
    # Kit is computed with discs=[] so a set's DISC-dependent auto 4pc DMG%
    # (Wuthering Salon, Phaethon's Melody) is excluded here and folded per
    # combo via set_tag_bonus instead — otherwise a combination that newly
    # forms such a set would be mis-valued (its auto DMG% would be missing).
    # ``mode``/``dealt_element`` MUST mirror calculate() (api.py) — element-
    # gated kit effects (e.g. Rina's +10% electric core passive, "squad
    # Electric DMG") are dropped without the dealt element, silently omitting
    # them from every fast-path build.
    kit = _kit_contributions(agent, replace(config, discs=[]),
                             agents, disc_sets, engines,
                             mode="direct", dealt_element=attr)

    base_buckets = _bucket_vector(engine.advanced_stat)
    panel_consts = _panel_constants(agent, engine, consts)

    base_crit_rate = consts.base_crit_rate + agent.core_bonus_crit_rate
    base_crit_dmg = consts.base_crit_dmg + agent.core_bonus_crit_dmg
    agent_plus_engine_atk = agent.total_base_atk() + engine.base_atk
    kit_atk_pct = kit["atk_pct"]
    kit_flat_atk = kit["flat_atk"]
    # Engine always-on CRIT Rate (Qingming Birdcage) joins the crit total
    # exactly as in calculate(); same for crit_dmg-bracket conditional
    # parts (Heartstring Nocturne).
    crit_rate_const = (config.external_crit_rate + kit["crit_rate"]
                       + engine.passive_crit(rank))
    crit_dmg_const = (
        config.external_crit_dmg + kit["crit_dmg"]
        + _engine_buff_bonus(engine, config.engine_buff_stacks,
                             attr, rank, bracket="crit_dmg",
                             skill_tag=config.skill_tag,
                             counts_as_aftershock=config.counts_as_aftershock)
    )
    # Kit/support PEN Ratio (e.g. Rina's squad PEN Ratio) — a constant that
    # calculate() folds into the DEF zone (build.pen_ratio + kit pen_ratio);
    # the fast path must too, or a support granting PEN Ratio under-values
    # every build's DefMult (PEN interacts non-linearly, so it is not a flat
    # scalar — hence the §4 re-check catches the omission). Same for enemy
    # DEF reduction (Qingyi M1), a separate factor in the DEF zone.
    pen_ratio_const = kit["pen_ratio"] + agent.core_bonus_pen_ratio
    def_red_const = kit["def_red"]
    # AP is inert for direct-hit damage, but the constraint / maximize-stat
    # total must be the SAME number the other modes report (engine passive
    # and kit/external AP included) — an agent whose other hit scales with
    # AP is exactly why a direct-hit search would ask for it.
    ap_const = (agent.base_anomaly_proficiency
                + agent.core_bonus_anomaly_proficiency
                + engine.passive_ap(rank)
                + kit["anomaly_proficiency"]
                + config.external_anomaly_proficiency)
    am_const = agent.base_anomaly_mastery + agent.core_bonus_anomaly_mastery
    # HP/DEF/Impact/Energy-Regen panel pieces, read only by the min-stat
    # constraints (inert for direct-hit damage): HP composes like the
    # sheer panel; the others are agent base × (1 + % total).
    hp_base = agent.total_base_hp()
    core_hp_pct = agent.core_bonus_hp_pct
    kit_hp_mult = 1.0 + kit["hp_pct"]
    def_base = agent.base_def
    impact_base = agent.base_impact + agent.core_bonus_impact
    er_base = agent.base_energy_regen + agent.core_bonus_energy_regen
    bonus_const = (
        sum(config.external_dmg_bonuses)
        + sum(_engine_passive_bonuses(engine, attr, rank))
        + _engine_buff_bonus(engine, config.engine_buff_stacks,
                             attr, rank,
                             skill_tag=config.skill_tag,
                             counts_as_aftershock=config.counts_as_aftershock)
        + sum(kit["dmg_bonus"])
    )
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
    stunned = options.objective.endswith("_stunned")
    stun = formulas.stun_mult(
        stunned, stun_base,
        list(config.external_daze_bonuses) + kit["daze_bonus"],
    )
    crit_variant = options.objective.replace("_stunned", "")
    const_mult = res * taken * stun * config.skill_multiplier

    def evaluate(buckets: list[float], tagged: tuple[float, float]) -> tuple[float, float]:
        """Fast path: (objective value, final ATK) of one bucket total."""
        atk_pre_combat = (
            agent_plus_engine_atk * (1.0 + buckets[_B_ATK_PCT])
            + buckets[_B_ATK_FLAT]
        )
        atk_total = (
            atk_pre_combat
            * (1.0 + buckets[_B_COMBAT_ATK_PCT] + kit_atk_pct)
            + kit_flat_atk
        )
        if crit_variant == "non_crit":
            crit_mult = 1.0
        else:
            crit_dmg = base_crit_dmg + buckets[_B_CRIT_DMG] + crit_dmg_const
            if crit_variant == "crit":
                crit_mult = 1.0 + crit_dmg
            else:
                crit_rate = (base_crit_rate + buckets[_B_CRIT_RATE]
                             + crit_rate_const)
                crit_mult = 1.0 + min(max(crit_rate, 0.0), crit_cap) * crit_dmg
        # Only the DMG% channel matters here: the anomaly-bracket
        # channel exists in anomaly modes only.
        bonus = 1.0 + buckets[_B_ATTR_DMG] + bonus_const + tagged[0]
        eff_def = formulas.effective_def(
            boss.base_def, buckets[_B_PEN_RATIO] + pen_ratio_const,
            buckets[_B_PEN_FLAT], def_red=def_red_const
        )
        defense = level_coef / (level_coef + eff_def)
        return atk_total * crit_mult * bonus * defense * const_mult, atk_total

    def stat_total(stat: str, buckets: list[float], atk_total: float) -> float:
        """One constraint / maximize stat's final total for a bucket
        vector (the totals the results panel reports)."""
        if stat == "ATK":
            return atk_total
        if stat == "HP":
            return (hp_base * (1.0 + core_hp_pct + buckets[_B_HP_PCT])
                    + buckets[_B_HP_FLAT]) * kit_hp_mult
        if stat == "DEF":
            return (def_base * (1.0 + buckets[_B_DEF_PCT])
                    + buckets[_B_DEF_FLAT])
        if stat == "CRIT Rate":
            return base_crit_rate + buckets[_B_CRIT_RATE] + crit_rate_const
        if stat == "CRIT DMG":
            return base_crit_dmg + buckets[_B_CRIT_DMG] + crit_dmg_const
        if stat == "PEN Ratio":
            return buckets[_B_PEN_RATIO] + pen_ratio_const
        if stat == "PEN":
            return buckets[_B_PEN_FLAT]
        if stat == "Impact":
            return impact_base * (1.0 + buckets[_B_IMPACT])
        if stat == "Energy Regen":
            return er_base * (1.0 + buckets[_B_ER])
        if stat == "Anomaly Proficiency":
            return ap_const + buckets[_B_AP]
        # "Anomaly Mastery" (keys validated above)
        return (am_const + buckets[_B_AM]) * (1.0 + buckets[_B_AM_PCT])

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
        if not max_stacks:
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
        set_2pc[key] = _set_2pc_buckets(entry, attr)
        stacks = assumed_stacks(key)
        set_stacks_used[key] = stacks
        # Wearer-side 4pc stats at the assumed stacks (mirrors
        # set_bonus_stats: classic bonus incl. at-max extra, or a
        # squad-facing part mapped to a build stat — Yunkui Tales' Sheer
        # DMG% lands in a bucket only the sheer fast path reads, so it
        # stays inert in this mode).
        set_4pc[key] = _bucket_vector(set_4pc_stats(entry, stacks))
        # 4pc DMG% additives in the ordinary bonus bracket, folded per combo:
        # skill-tag-gated ones (Puffer Electro's Ultimate +20%) and a set's
        # auto 4pc DMG% (Wuthering Salon +18%; Phaethon's Melody needs a
        # teammate) — the auto part mirrors _kit_contributions, which is why
        # ``kit`` above excludes it (discs=[]).
        tag_skill = sum(
            b.value for b in entry.bonus_4pc_dmg
            if (config.skill_tag is not None
                and b.skill_tag == config.skill_tag)
            or (config.counts_as_aftershock
                and b.skill_tag == "aftershock")
        )
        auto = 0.0
        if entry.auto_4pc_dmg and (
                not entry.auto_4pc_dmg_needs_teammate or teammate_present):
            auto = entry.auto_4pc_dmg
        set_tag_bonus[key] = (tag_skill + auto, 0.0)

    def combo_set_stacks(counts: dict[str, int]) -> dict[str, int]:
        """The ``CalcConfig.set_stacks`` this combination is evaluated at."""
        return {
            key: set_stacks_used[key]
            for key, count in counts.items()
            if count >= 4 and set_stacks_used[key]
        }

    # --- Branch-and-bound DFS (§11 E2), shared search core -----------------
    outcome = _search_maximizing(
        candidates, slots, equipped, base_buckets, evaluate, feasible,
        stat_total, set_2pc, set_4pc, set_tag_bonus, candidate_sets,
        required, options, _bound,
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
    _, baseline_atk = evaluate(base_bb, (0.0, 0.0))
    baseline_stat = (stat_total(options.maximize_stat, base_bb, baseline_atk)
                     if options.maximize_stat else None)
    baseline_shortfalls: list = []
    baseline_feasible = (
        outcome.check_baseline(base_bb, baseline_atk, baseline_value,
                               baseline_shortfalls)
        and (required is None or baseline_counts.get(required, 0) >= 4)
        and _baseline_mains_ok(equipped, options.slot_main_stats,
                               attr)
        and (not options.sets_only or _sets_coverage_ok(config.discs))
    )

    def verify(combo: tuple[_Candidate, ...]) -> BuildOption:
        """Re-run one build through calculate() (exactness guarantee, §4)."""
        discs = [c.disc for c in combo]
        buckets, tagged, counts = _combo_buckets(
            combo, base_buckets, set_2pc, set_4pc, set_tag_bonus)
        fast_value, aux = evaluate(buckets, tagged)
        stacks = combo_set_stacks(counts)
        results = calculate(
            replace(config, discs=discs, set_stacks=stacks), **data
        )
        exact = getattr(results, options.objective)
        if abs(fast_value - exact) > _RECHECK_RTOL * max(1.0, abs(exact)):
            raise OptimizeError(
                f"Internal error: fast evaluation ({fast_value!r}) disagrees "
                f"with calculate() ({exact!r}) — please report this build"
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
            stat_value=(stat_total(options.maximize_stat, buckets, aux)
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
                required=required,
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
        required_4pc=required,
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
    required = _validate_options(options, ANOMALY_OBJECTIVES)
    if config.buildup_segments or config.unbuffed_share:
        raise OptimizeError(
            "optimize_anomaly targets the fully-buffed anomaly number; "
            "buildup segments / unbuffed_share (buildup dilution) are not "
            "supported — clear them and optimize again"
        )

    consts = consts if consts is not None else load_constants()
    disc_data = disc_data if disc_data is not None else load_disc_data()
    bosses = bosses if bosses is not None else load_bosses()
    agents = agents if agents is not None else load_agents()
    engines = engines if engines is not None else load_engines()
    disc_sets = disc_sets if disc_sets is not None else load_disc_sets()
    anomaly_data = anomaly_data if anomaly_data is not None else load_anomalies()
    if user_discs is None:
        user_discs = load_user_discs(disc_data)

    data = dict(consts=consts, disc_data=disc_data, bosses=bosses,
                agents=agents, engines=engines, disc_sets=disc_sets)

    if required is not None and required not in disc_sets:
        raise OptimizeError(
            f"Unknown required set '{required}'; options: {sorted(disc_sets)}"
        )

    _validate_slot_mains(options, disc_data)

    # --- Baseline: validates the whole config (mode, discs, anomaly) -------
    baseline_results = calculate_anomaly(config, **data, anomaly_data=anomaly_data)
    baseline_value = getattr(baseline_results, options.objective)
    equipped = {d.slot: d for d in config.discs}
    agent = agents[config.agent_key]
    # Lumiflux (Attribute Mutation): the element this agent's damage
    # is dealt as — her own attribute for everyone else.
    attr = attack_element(agent, config)
    # Self-scaling kit inputs (the MAIN's own panel values) become
    # minimum-stat constraints — the buff amount is a constant, so no
    # build may fall short of the panel the user claimed.
    options = _apply_scaling_input_minimums(agent, config, options,
                                            CONSTRAINT_STATS)
    engine_key = (config.engine_key if config.engine_key is not None
                  else agent.default_engine)
    engine = engines[engine_key]
    boss = bosses[config.boss_name]
    rank = _resolve_engine_rank(engine, config.engine_rank)

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
        anomaly = anomaly_data.anomalies[attr]
        kind, dealt_element, hits = "anomaly", attr, anomaly.hits
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

    # --- Candidates per slot (plan §2): inventory + equipped virtuals -----
    damage_indices = (_ANOMALY_DAMAGE_INDICES if attr_dmg_applies
                      else _ANOMALY_DAMAGE_INDICES_NO_ATTR)
    dominance_indices = _dominance_indices(
        damage_indices, _watched_stats(options))
    candidates, pruned_count = _gather_candidates(
        equipped, user_discs, disc_data, attr, options,
        dominance_indices, _prune,
    )

    slots = sorted(candidates)

    if required is not None:
        equippable = sum(
            1 for slot in slots
            if any(c.disc.disc_set == required for c in candidates[slot])
        )
        if equippable < 4:
            raise OptimizeError(
                f"Cannot build 4 pieces of '{disc_sets[required].name}': "
                f"only {equippable} slot(s) have a saved piece of it "
                f"(locked slots included)"
            )

    # --- Constant factors (disc-independent, mirrors calculate_anomaly) ----
    # Kit is computed with discs=[] so its DISC-dependent part (a set's auto
    # 4pc DMG%, e.g. Wuthering Salon) is excluded here and folded per combo
    # via set_tag_bonus instead — the rest of the kit is disc-independent.
    kit = _kit_contributions(
        agent, replace(config, discs=[]), agents, disc_sets, engines,
        mode=kind, dealt_element=dealt_element,
    )

    base_buckets = _bucket_vector(engine.advanced_stat)
    panel_consts = _panel_constants(agent, engine, consts)

    base_crit_rate = consts.base_crit_rate + agent.core_bonus_crit_rate
    base_crit_dmg = consts.base_crit_dmg + agent.core_bonus_crit_dmg
    agent_plus_engine_atk = agent.total_base_atk() + engine.base_atk
    kit_atk_pct = kit["atk_pct"]
    kit_flat_atk = kit["flat_atk"]
    pen_ratio_const = kit["pen_ratio"] + agent.core_bonus_pen_ratio
    def_red_const = kit["def_red"]
    crit_rate_const = config.external_crit_rate + kit["crit_rate"]
    crit_dmg_const = config.external_crit_dmg + kit["crit_dmg"]
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
    am_const = agent.base_anomaly_mastery + agent.core_bonus_anomaly_mastery
    # HP/DEF/Impact/Energy-Regen panel pieces, read only by the min-stat
    # constraints (inert for anomaly damage).
    hp_base = agent.total_base_hp()
    core_hp_pct = agent.core_bonus_hp_pct
    kit_hp_mult = 1.0 + kit["hp_pct"]
    def_base = agent.base_def
    impact_base = agent.base_impact + agent.core_bonus_impact
    er_base = agent.base_energy_regen + agent.core_bonus_energy_regen

    engine_buff_dmg = _engine_buff_bonus(
        engine, config.engine_buff_stacks, dealt_element, rank, mode=kind,
    )
    bonus_const = (
        sum(config.external_dmg_bonuses)
        + sum(_engine_passive_bonuses(engine, dealt_element, rank))
        + engine_buff_dmg
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
    luminize_res = 0.50 if (config.luminize is not None
                            and config.luminize.res_ignore) else 0.0
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
    stunned = options.objective.endswith("_stunned")
    stun = formulas.stun_mult(
        stunned, stun_base,
        list(config.external_daze_bonuses) + kit["daze_bonus"],
    )
    # "full" multiplies the per-proc value by the anomaly's proc count.
    hits_factor = hits if options.objective.startswith("full") else 1
    # Refringe (Lumiflux teammate) is a kit constant in its own zone —
    # disc-independent, so it folds straight into the outer product.
    refringe = formulas.refringe_mult([kit["refringe"]])
    outer = level_mult * res * taken * stun * hits_factor * refringe

    # --- Per-proc / burst multiplier (constant, or AP-scaled per build) ----
    disorder_additive = config.external_disorder_mult_add + kit["disorder_mult_add"]
    if config.luminize is not None:
        # Constant: it depends on the PROVIDER's AP, not on this build.
        provider = agents[config.luminize.provider_key]
        luminize_mult = provider.luminize.multiplier(
            config.luminize.skill, config.luminize.anomaly_proficiency,
            m4=config.luminize.m4)
        def mult_of_ap(ap: float) -> float:
            return luminize_mult
    elif kind == "vortex":
        const_mult = formulas.burst_conversion_mult(
            rule.mult, rule.time_mult, rule.window,
            config.disorder_elapsed_seconds, extra_mult=disorder_additive,
        )
        def mult_of_ap(ap: float) -> float:
            return const_mult
    elif kind == "disorder" and config.polarity_disorder:
        base_disorder = formulas.burst_conversion_mult(
            rule.base, rule.time_mult, rule.window,
            config.disorder_elapsed_seconds, extra_mult=0.0,
        )
        polarity_coeff = 0.05 + config.polarity_special_level * 0.0225
        def mult_of_ap(ap: float) -> float:
            return (POLARITY_DISORDER_FRACTION * base_disorder
                    + disorder_additive + polarity_coeff * (ap / 100.0))
    elif kind == "disorder":
        const_mult = formulas.burst_conversion_mult(
            rule.base, rule.time_mult, rule.window,
            config.disorder_elapsed_seconds, extra_mult=disorder_additive,
        )
        def mult_of_ap(ap: float) -> float:
            return const_mult
    elif config.abloom:
        mv = VIVIAN_ABLOOM_MV[dealt_element]
        m2 = VIVIAN_ABLOOM_M2_FACTOR if config.mindscapes.get(2) else 1.0
        abloom_base = anomaly.mult * (mv / 100.0) * m2
        def mult_of_ap(ap: float) -> float:
            return abloom_base * (ap / 10.0)
    else:
        const_mult = (config.anomaly_mult_override
                      if config.anomaly_mult_override is not None
                      else anomaly.mult)
        def mult_of_ap(ap: float) -> float:
            return const_mult

    def evaluate(buckets: list[float], tagged: tuple[float, float]) -> tuple[float, tuple]:
        """Fast path: (objective value, (final ATK, total AP)) of a build."""
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
        bonus = 1.0 + bonus_const + tagged[0]
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
        return value, (atk_total, ap_total)

    def stat_total(stat: str, buckets: list[float], aux: tuple) -> float:
        """One constraint / maximize stat's final total for a bucket
        vector (the totals the results panel reports)."""
        atk_total, ap_total = aux
        if stat == "ATK":
            return atk_total
        if stat == "HP":
            return (hp_base * (1.0 + core_hp_pct + buckets[_B_HP_PCT])
                    + buckets[_B_HP_FLAT]) * kit_hp_mult
        if stat == "DEF":
            return (def_base * (1.0 + buckets[_B_DEF_PCT])
                    + buckets[_B_DEF_FLAT])
        if stat == "CRIT Rate":
            return base_crit_rate + buckets[_B_CRIT_RATE] + crit_rate_const
        if stat == "CRIT DMG":
            return base_crit_dmg + buckets[_B_CRIT_DMG] + crit_dmg_const
        if stat == "PEN Ratio":
            return buckets[_B_PEN_RATIO] + pen_ratio_const
        if stat == "PEN":
            return buckets[_B_PEN_FLAT]
        if stat == "Impact":
            return impact_base * (1.0 + buckets[_B_IMPACT])
        if stat == "Energy Regen":
            return er_base * (1.0 + buckets[_B_ER])
        if stat == "Anomaly Proficiency":
            return ap_total
        # "Anomaly Mastery" (keys validated above)
        return (am_const + buckets[_B_AM]) * (1.0 + buckets[_B_AM_PCT])

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
        if not max_stacks:
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
        set_2pc[key] = _set_2pc_buckets(entry, attr)
        stacks = assumed_stacks(key)
        set_stacks_used[key] = stacks
        # Wearer-side 4pc stats at the assumed stacks (mirrors
        # set_bonus_stats: classic bonus incl. at-max extra, or a
        # squad-facing part mapped to a build stat).
        set_4pc[key] = _bucket_vector(set_4pc_stats(entry, stacks))
        # 4pc DMG% additives that land in the ordinary bonus bracket: the
        # skill-tag-gated ones (none for anomaly — skill_tag is unset) plus
        # an auto 4pc DMG% (e.g. Wuthering Salon), which needs a teammate
        # for some sets. Folded per combo, mirroring _kit_contributions.
        tag_skill = sum(
            b.value for b in entry.bonus_4pc_dmg
            if (config.skill_tag is not None
                and b.skill_tag == config.skill_tag)
            or (config.counts_as_aftershock
                and b.skill_tag == "aftershock")
        )
        auto = 0.0
        if entry.auto_4pc_dmg and (
                not entry.auto_4pc_dmg_needs_teammate or teammate_present):
            auto = entry.auto_4pc_dmg
        # Second channel: a 4pc part in the separate Anomaly Buff
        # bracket, optionally gated to the wearer's attribute
        # (Feathered Fate is Lumiflux-only). Mirrors
        # _kit_contributions exactly, or the section-4 re-check fails.
        anom = 0.0
        if entry.auto_4pc_anomaly_dmg and (
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

    # --- Branch-and-bound DFS (§11 E2), shared search core -----------------
    outcome = _search_maximizing(
        candidates, slots, equipped, base_buckets, evaluate, feasible,
        stat_total, set_2pc, set_4pc, set_tag_bonus, candidate_sets,
        required, options, _bound,
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
    baseline_stat = (stat_total(options.maximize_stat, base_bb, baseline_aux)
                     if options.maximize_stat else None)
    baseline_shortfalls: list = []
    baseline_feasible = (
        outcome.check_baseline(base_bb, baseline_aux, baseline_value,
                               baseline_shortfalls)
        and (required is None or baseline_counts.get(required, 0) >= 4)
        and _baseline_mains_ok(equipped, options.slot_main_stats,
                               attr)
        and (not options.sets_only or _sets_coverage_ok(config.discs))
    )

    def verify(combo: tuple[_Candidate, ...]) -> BuildOption:
        """Re-run one build through calculate_anomaly (exactness, §4)."""
        discs = [c.disc for c in combo]
        buckets, tagged, counts = _combo_buckets(
            combo, base_buckets, set_2pc, set_4pc, set_tag_bonus)
        fast_value, aux = evaluate(buckets, tagged)
        stacks = combo_set_stacks(counts)
        results = calculate_anomaly(
            replace(config, discs=discs, set_stacks=stacks),
            **data, anomaly_data=anomaly_data,
        )
        exact = getattr(results, options.objective)
        if abs(fast_value - exact) > _RECHECK_RTOL * max(1.0, abs(exact)):
            raise OptimizeError(
                f"Internal error: fast evaluation ({fast_value!r}) disagrees "
                f"with calculate_anomaly() ({exact!r}) — please report this "
                f"build"
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
            stat_value=(stat_total(options.maximize_stat, buckets, aux)
                        if options.maximize_stat else None),
        )

    ranked = sorted(heap, key=lambda item: (-item[0], -item[1]))
    options_verified = [verify(combo) for _value, _counter, combo in ranked]

    if not options_verified and not baseline_feasible:
        raise _infeasible_error(
            _baseline_reasons(
                baseline_shortfalls,
                discs=config.discs,
                equipped=equipped,
                required=required,
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
        required_4pc=required,
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
    options = options if options is not None else OptimizeOptions()
    required = _validate_options(options, OBJECTIVES, SHEER_CONSTRAINT_STATS)

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

    if required is not None and required not in disc_sets:
        raise OptimizeError(
            f"Unknown required set '{required}'; options: {sorted(disc_sets)}"
        )

    _validate_slot_mains(options, disc_data)

    # --- Baseline: validates the whole config (agent specialty included) ---
    baseline_results = calculate_sheer(config, **data)
    baseline_value = getattr(baseline_results, options.objective)
    equipped = {d.slot: d for d in config.discs}
    agent = agents[config.agent_key]
    # Lumiflux (Attribute Mutation): the element this agent's damage
    # is dealt as — her own attribute for everyone else.
    attr = attack_element(agent, config)
    # Self-scaling kit inputs (the MAIN's own panel values) become
    # minimum-stat constraints — the buff amount is a constant, so no
    # build may fall short of the panel the user claimed.
    options = _apply_scaling_input_minimums(agent, config, options,
                                            SHEER_CONSTRAINT_STATS)

    # --- Candidates per slot (plan §2): inventory + equipped virtuals -----
    # HP only matters when the agent converts it to Sheer Force; for a
    # pure-ATK rupture agent the HP buckets must NOT weaken the dominance
    # pruning (an HP-heavy disc adds nothing there). An HP constraint
    # still forces them into the comparison, so a disc kept alive only by
    # the constraint is never discarded.
    damage_indices = _SHEER_DAMAGE_INDICES + (
        _SHEER_HP_INDICES if agent.sheer_force_hp_conversion > 0 else ()
    )
    dominance_indices = _dominance_indices(
        damage_indices, _watched_stats(options))
    candidates, pruned_count = _gather_candidates(
        equipped, user_discs, disc_data, attr, options,
        dominance_indices, _prune,
    )

    slots = sorted(candidates)

    if required is not None:
        equippable = sum(
            1 for slot in slots
            if any(c.disc.disc_set == required for c in candidates[slot])
        )
        if equippable < 4:
            raise OptimizeError(
                f"Cannot build 4 pieces of '{disc_sets[required].name}': "
                f"only {equippable} slot(s) have a saved piece of it "
                f"(locked slots included)"
            )

    # --- Constant factors (disc-independent, mirrors calculate_sheer) ------
    engine_key = (config.engine_key if config.engine_key is not None
                  else agent.default_engine)
    engine = engines[engine_key]
    boss = bosses[config.boss_name]
    rank = _resolve_engine_rank(engine, config.engine_rank)
    # Kit is computed with discs=[] so a set's DISC-dependent auto 4pc DMG%
    # is excluded here and folded per combo via set_tag_bonus instead.
    # ``mode``/``dealt_element`` MUST mirror calculate_sheer — element-gated
    # kit/engine effects stay element-checked in sheer mode.
    kit = _kit_contributions(agent, replace(config, discs=[]),
                             agents, disc_sets, engines,
                             mode="sheer", dealt_element=attr)

    base_buckets = _bucket_vector(engine.advanced_stat)
    panel_consts = _panel_constants(agent, engine, consts)

    base_crit_rate = consts.base_crit_rate + agent.core_bonus_crit_rate
    base_crit_dmg = consts.base_crit_dmg + agent.core_bonus_crit_dmg
    agent_plus_engine_atk = agent.total_base_atk() + engine.base_atk
    kit_atk_pct = kit["atk_pct"]
    kit_flat_atk = kit["flat_atk"]
    # HP side of Sheer Force: sheet HP% scales the base bucket only, flat
    # HP is added after it, and kit/team HP% (Lucia, Dreamlit Hearth)
    # multiplies the FINISHED panel — the combat HP bracket, mirroring
    # aggregate_build + calculate_sheer.
    hp_base = agent.total_base_hp()
    core_hp_pct = agent.core_bonus_hp_pct
    kit_hp_mult = 1.0 + kit["hp_pct"]
    sf_atk_conv = consts.sheer_force_atk_conversion
    sf_hp_conv = agent.sheer_force_hp_conversion
    flat_sf_const = config.sheer_force_flat + kit["sheer_force"]
    # Constraint-only panel pieces (inert for Sheer damage): DEF / Impact /
    # Energy Regen / PEN / AP / AM totals, composed like the other modes.
    def_base = agent.base_def
    impact_base = agent.base_impact + agent.core_bonus_impact
    er_base = agent.base_energy_regen + agent.core_bonus_energy_regen
    pen_ratio_const = kit["pen_ratio"] + agent.core_bonus_pen_ratio
    ap_const = (agent.base_anomaly_proficiency
                + agent.core_bonus_anomaly_proficiency
                + engine.passive_ap(rank)
                + kit["anomaly_proficiency"]
                + config.external_anomaly_proficiency)
    am_const = agent.base_anomaly_mastery + agent.core_bonus_anomaly_mastery
    crit_rate_const = (config.external_crit_rate + kit["crit_rate"]
                       + engine.passive_crit(rank))
    crit_dmg_const = (
        config.external_crit_dmg + kit["crit_dmg"]
        + _engine_buff_bonus(engine, config.engine_buff_stacks,
                             attr, rank, mode="sheer",
                             bracket="crit_dmg",
                             skill_tag=config.skill_tag,
                             counts_as_aftershock=config.counts_as_aftershock)
    )
    bonus_const = (
        sum(config.external_dmg_bonuses)
        + sum(_engine_passive_bonuses(engine, attr, rank))
        + _engine_buff_bonus(engine, config.engine_buff_stacks,
                             attr, rank, mode="sheer",
                             skill_tag=config.skill_tag,
                             counts_as_aftershock=config.counts_as_aftershock)
        + sum(kit["dmg_bonus"])
    )
    # Dedicated Sheer DMG bracket constants (manual entries + kit + engine
    # buff parts — e.g. Qingming Birdcage's tag-gated Sheer DMG stacks);
    # the disc side (Yunkui Tales 4pc at max stacks) arrives per combo via
    # the _B_SHEER_DMG bucket.
    sheer_const = (
        sum(config.external_sheer_dmg)
        + sum(kit["sheer_dmg"])
        + _engine_buff_bonus(engine, config.engine_buff_stacks,
                             attr, rank, mode="sheer",
                             bracket="sheer_dmg",
                             skill_tag=config.skill_tag,
                             counts_as_aftershock=config.counts_as_aftershock)
    )
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
    stunned = options.objective.endswith("_stunned")
    stun = formulas.stun_mult(
        stunned, stun_base,
        list(config.external_daze_bonuses) + kit["daze_bonus"],
    )
    crit_variant = options.objective.replace("_stunned", "")
    # NO DEF zone: Sheer damage ignores enemy DEF (calculate_sheer uses
    # defense = 1.0), so no level coefficient / PEN factors appear here.
    const_mult = res * taken * stun * config.skill_multiplier

    def evaluate(buckets: list[float], tagged: tuple[float, float]) -> tuple[float, tuple]:
        """Fast path: (objective value, (final ATK, HP, Sheer Force))."""
        atk_pre_combat = (
            agent_plus_engine_atk * (1.0 + buckets[_B_ATK_PCT])
            + buckets[_B_ATK_FLAT]
        )
        atk_total = (
            atk_pre_combat
            * (1.0 + buckets[_B_COMBAT_ATK_PCT] + kit_atk_pct)
            + kit_flat_atk
        )
        hp_total = (
            hp_base * (1.0 + core_hp_pct + buckets[_B_HP_PCT])
            + buckets[_B_HP_FLAT]
        ) * kit_hp_mult
        sf = atk_total * sf_atk_conv + hp_total * sf_hp_conv + flat_sf_const
        if crit_variant == "non_crit":
            crit_mult = 1.0
        else:
            crit_dmg = base_crit_dmg + buckets[_B_CRIT_DMG] + crit_dmg_const
            if crit_variant == "crit":
                crit_mult = 1.0 + crit_dmg
            else:
                crit_rate = (base_crit_rate + buckets[_B_CRIT_RATE]
                             + crit_rate_const)
                crit_mult = 1.0 + min(max(crit_rate, 0.0), crit_cap) * crit_dmg
        # Only the DMG% channel matters here: the anomaly-bracket
        # channel exists in anomaly modes only.
        bonus = 1.0 + buckets[_B_ATTR_DMG] + bonus_const + tagged[0]
        sheer_mult = 1.0 + sheer_const + buckets[_B_SHEER_DMG]
        value = sf * crit_mult * bonus * sheer_mult * const_mult
        return value, (atk_total, hp_total, sf)

    def stat_total(stat: str, buckets: list[float], aux: tuple) -> float:
        """One constraint / maximize stat's final total for a bucket
        vector (the totals SheerResults reports)."""
        atk_total, hp_total, sf = aux
        if stat == "ATK":
            return atk_total
        if stat == "HP":
            return hp_total
        if stat == "Sheer Force":
            return sf
        if stat == "DEF":
            return (def_base * (1.0 + buckets[_B_DEF_PCT])
                    + buckets[_B_DEF_FLAT])
        if stat == "CRIT Rate":
            return base_crit_rate + buckets[_B_CRIT_RATE] + crit_rate_const
        if stat == "CRIT DMG":
            return base_crit_dmg + buckets[_B_CRIT_DMG] + crit_dmg_const
        if stat == "PEN Ratio":
            return buckets[_B_PEN_RATIO] + pen_ratio_const
        if stat == "PEN":
            return buckets[_B_PEN_FLAT]
        if stat == "Impact":
            return impact_base * (1.0 + buckets[_B_IMPACT])
        if stat == "Energy Regen":
            return er_base * (1.0 + buckets[_B_ER])
        if stat == "Anomaly Proficiency":
            return ap_const + buckets[_B_AP]
        # "Anomaly Mastery" (keys validated above)
        return (am_const + buckets[_B_AM]) * (1.0 + buckets[_B_AM_PCT])

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
        if not max_stacks:
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
        set_2pc[key] = _set_2pc_buckets(entry, attr)
        stacks = assumed_stacks(key)
        set_stacks_used[key] = stacks
        # Wearer-side 4pc stats at the assumed stacks (mirrors
        # set_bonus_stats: classic bonus incl. the at-max extra — Yunkui
        # Tales' Sheer DMG% lands in _B_SHEER_DMG, which THIS mode's fast
        # path reads — or a squad-facing part mapped to a build stat).
        set_4pc[key] = _bucket_vector(set_4pc_stats(entry, stacks))
        # 4pc DMG% additives in the ordinary bonus bracket, folded per combo:
        # skill-tag-gated ones and a set's auto 4pc DMG% — both land in the
        # same additive DMG% bracket in sheer mode too (a tag-gated Sheer
        # DMG% set doesn't exist today; the §4 re-check would hard-fail
        # rather than mis-rank if one were ever added to the data).
        tag_skill = sum(
            b.value for b in entry.bonus_4pc_dmg
            if (config.skill_tag is not None
                and b.skill_tag == config.skill_tag)
            or (config.counts_as_aftershock
                and b.skill_tag == "aftershock")
        )
        auto = 0.0
        if entry.auto_4pc_dmg and (
                not entry.auto_4pc_dmg_needs_teammate or teammate_present):
            auto = entry.auto_4pc_dmg
        set_tag_bonus[key] = (tag_skill + auto, 0.0)

    def combo_set_stacks(counts: dict[str, int]) -> dict[str, int]:
        """The ``CalcConfig.set_stacks`` this combination is evaluated at."""
        return {
            key: set_stacks_used[key]
            for key, count in counts.items()
            if count >= 4 and set_stacks_used[key]
        }

    # --- Branch-and-bound DFS (§11 E2), shared search core -----------------
    outcome = _search_maximizing(
        candidates, slots, equipped, base_buckets, evaluate, feasible,
        stat_total, set_2pc, set_4pc, set_tag_bonus, candidate_sets,
        required, options, _bound,
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
    baseline_stat = (stat_total(options.maximize_stat, base_bb, baseline_aux)
                     if options.maximize_stat else None)
    baseline_shortfalls: list = []
    baseline_feasible = (
        outcome.check_baseline(base_bb, baseline_aux, baseline_value,
                               baseline_shortfalls)
        and (required is None or baseline_counts.get(required, 0) >= 4)
        and _baseline_mains_ok(equipped, options.slot_main_stats,
                               attr)
        and (not options.sets_only or _sets_coverage_ok(config.discs))
    )

    def verify(combo: tuple[_Candidate, ...]) -> BuildOption:
        """Re-run one build through calculate_sheer (exactness, §4)."""
        discs = [c.disc for c in combo]
        buckets, tagged, counts = _combo_buckets(
            combo, base_buckets, set_2pc, set_4pc, set_tag_bonus)
        fast_value, aux = evaluate(buckets, tagged)
        stacks = combo_set_stacks(counts)
        results = calculate_sheer(
            replace(config, discs=discs, set_stacks=stacks), **data
        )
        exact = getattr(results, options.objective)
        if abs(fast_value - exact) > _RECHECK_RTOL * max(1.0, abs(exact)):
            raise OptimizeError(
                f"Internal error: fast evaluation ({fast_value!r}) disagrees "
                f"with calculate_sheer() ({exact!r}) — please report this "
                f"build"
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
            stat_value=(stat_total(options.maximize_stat, buckets, aux)
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
                required=required,
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
        required_4pc=required,
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
    )
