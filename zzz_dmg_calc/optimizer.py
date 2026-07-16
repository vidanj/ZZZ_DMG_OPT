"""Best-build search over the user's saved disc inventory.

Design: DOCS/optimizer_plan.md (§11 for the E1-E3 extensions). Given a
:class:`~.api.CalcConfig` the user just calculated, find the combination
of their SAVED discs (``data/user_discs.json``, plus the currently
equipped discs as unsaved virtual candidates) that maximizes one field of
the direct-hit results — or confirm the current discs are already
optimal. Everything except ``config.discs`` / ``config.set_stacks`` is
held constant.

Entry point::

    optimize(config, options) -> OptimizeResult

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
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field, replace

from . import formulas
from .agent import Agent, load_agents
from .anomalies import AnomalyData, load_anomalies
from .api import (
    POLARITY_DISORDER_FRACTION, VIVIAN_ABLOOM_M2_FACTOR, VIVIAN_ABLOOM_MV,
    AnomalyResults, CalcConfig, CalcResults, calculate, calculate_anomaly,
    _engine_buff_bonus, _engine_passive_bonuses, _kit_contributions,
    _resolve_engine_rank,
)
from .constants import Constants, load_constants
from .discs import (
    Disc, DiscData, DiscSet, disc_stats, load_disc_data, load_disc_sets,
    load_user_discs,
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

#: Stats ``OptimizeOptions.min_stats`` may constrain (§11 E3), defined as
#: the final-build totals the results panel reports: ATK is the finished
#: panel (combat + kit brackets included); CRIT Rate/DMG include external
#: and kit conditionals; AP/AM are the build aggregation's totals
#: (agent base + core + engine + discs + sets).
CONSTRAINT_STATS = ("ATK", "CRIT Rate", "CRIT DMG", "PEN Ratio", "PEN",
                    "Anomaly Proficiency", "Anomaly Mastery")

#: Default cap on builds *evaluated* by the branch-and-bound search —
#: keeps the local server responsive on adversarial inventories (§11 E2;
#: the raw cartesian product no longer matters).
DEFAULT_COMBO_BUDGET = 5_000_000

#: Relative tolerance of the fast-path vs calculate() re-check (plan §4).
_RECHECK_RTOL = 1e-9

#: Stat buckets tracked per candidate. Indices 0-7 are the direct-hit
#: damage-relevant stats (the fast path and the dominance pruning share
#: these vectors); 8-9 exist only for min-stat constraints and join the
#: dominance comparison only when constrained (§11 E3). Routing mirrors
#: ``agent._fold_stat``; every other stat (HP, DEF, Impact, ...) affects
#: neither the damage nor a supported constraint and is ignored, which
#: is also what makes pruning effective (plan §6).
_B_ATK_PCT, _B_COMBAT_ATK_PCT, _B_ATK_FLAT, _B_CRIT_RATE, _B_CRIT_DMG, \
    _B_PEN_RATIO, _B_PEN_FLAT, _B_ATTR_DMG, _B_AP, _B_AM = range(10)

_N_BUCKETS = 10
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

#: Bucket a min-stat constraint watches, when it maps to a single bucket
#: (ATK is the composed panel total, handled separately in ``feasible``).
_CONSTRAINT_STAT_INDEX = {
    "CRIT Rate": _B_CRIT_RATE,
    "CRIT DMG": _B_CRIT_DMG,
    "PEN Ratio": _B_PEN_RATIO,
    "PEN": _B_PEN_FLAT,
    "Anomaly Proficiency": _B_AP,
    "Anomaly Mastery": _B_AM,
}

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
}

#: Extra dominance indices a constraint on a stat requires (stats whose
#: buckets are already damage-relevant need nothing extra).
_CONSTRAINT_EXTRA_INDICES = {
    "Anomaly Proficiency": _B_AP,
    "Anomaly Mastery": _B_AM,
}


class OptimizeError(ValueError):
    """Raised for invalid optimizer options, an over-budget search, or
    constraints no combination can meet."""


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
        combo_budget: Abort after evaluating this many builds — the
            error tells the user to lock slots (§11 E2).
        min_stats: stat name -> minimum final-build total the build must
            reach (keys from :data:`CONSTRAINT_STATS`). If the baseline
            itself misses a minimum, the best *feasible* build may deal
            less damage than the baseline (negative delta).
        required_4pc: Set key the build MUST wear >= 4 pieces of
            (§11b E4) — the user decides which 4pc passive to build
            around (synergy the model can't value); ``None`` = no
            requirement. Any registry set is allowed, modeled 4pc or
            not. Feasibility semantics mirror ``min_stats``.
    """

    objective: str = "average"
    set_assumption: str = "max"
    locked_slots: frozenset[int] = frozenset()
    top_n: int = 5
    combo_budget: int = DEFAULT_COMBO_BUDGET
    min_stats: dict[str, float] = field(default_factory=dict)
    required_4pc: str | None = None


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
    """

    value: float
    delta: float
    discs: tuple[Disc, ...]
    disc_ids: tuple[str | None, ...]
    set_stacks: dict[str, int]
    changed_slots: tuple[int, ...]
    results: CalcResults


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


def _validate_options(options: OptimizeOptions, objectives: tuple) -> str | None:
    """Validate the pre-load parts of ``options`` (shared by both search
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
    bad_slots = [s for s in options.locked_slots if s not in (1, 2, 3, 4, 5, 6)]
    if bad_slots:
        raise OptimizeError(f"Invalid locked slots: {sorted(bad_slots)}")
    for stat, minimum in options.min_stats.items():
        if stat not in CONSTRAINT_STATS:
            raise OptimizeError(
                f"Unknown constraint stat '{stat}'; "
                f"options: {list(CONSTRAINT_STATS)}"
            )
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) \
                or minimum < 0:
            raise OptimizeError(
                f"Minimum for '{stat}' must be a number >= 0, "
                f"got {minimum!r}"
            )
    required = options.required_4pc
    if required is not None and (
            not isinstance(required, str) or not required.strip()):
        raise OptimizeError("'required_4pc' must be a set key or None")
    return required


def _fold_sets(
    buckets: list[float], counts: dict[str, int],
    set_2pc: dict[str, tuple[float, ...]],
    set_4pc: dict[str, tuple[float, ...]],
    set_tag_bonus: dict[str, float],
) -> float:
    """Add set bonuses for ``counts`` into ``buckets`` in place; return the
    tagged-DMG% total (skill-tag 4pc DMG and, in anomaly mode, auto 4pc
    DMG additives) that lands in the additive DMG% bracket at >= 4 pieces.
    """
    tagged = 0.0
    for key, count in counts.items():
        if count >= 2:
            for k, v in enumerate(set_2pc[key]):
                buckets[k] += v
        if count >= 4:
            for k, v in enumerate(set_4pc[key]):
                buckets[k] += v
            tagged += set_tag_bonus[key]
    return tagged


def _search(
    candidates: dict[int, list[_Candidate]],
    slots: list[int],
    equipped: dict[int, Disc],
    base_buckets: tuple[float, ...],
    evaluate,
    feasible,
    set_2pc: dict[str, tuple[float, ...]],
    set_4pc: dict[str, tuple[float, ...]],
    set_tag_bonus: dict[str, float],
    candidate_sets: set[str],
    required: str | None,
    top_n: int,
    budget: int,
    bound: bool,
) -> tuple[list, int]:
    """Branch-and-bound DFS over the per-slot candidates (plan §11 E2).

    Mode-agnostic search core shared by :func:`optimize` (direct hits) and
    :func:`optimize_anomaly`. ``evaluate(buckets, tagged) -> (value, aux)``
    and ``feasible(buckets, aux) -> bool`` carry all the mode-specific
    damage math; everything here — candidate ordering, the suffix/static
    upper bounds, the count-aware set bound, the top-N heap and the
    evaluation budget — is identical across modes because anomaly damage is
    monotone in every tracked bucket, exactly like direct-hit damage.

    Returns ``(heap, evals)``: the raw top-N heap of
    ``(value, -counter, combo)`` tuples (ranked by the caller) and the
    number of leaf builds evaluated.

    Raises:
        OptimizeError: the search exceeded ``budget`` evaluated builds.
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
        return evaluate(merged, 0.0)[0]

    for slot in slots:
        head = candidates[slot][:1] if equipped.get(slot) is not None else []
        tail = candidates[slot][len(head):]
        tail.sort(key=solo_value, reverse=True)
        candidates[slot] = head + tail

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
    static_tag_ub = max(
        (set_tag_bonus[key] for key in candidate_sets), default=0.0
    )

    heap: list[tuple[float, int, tuple[_Candidate, ...]]] = []
    counter = 0
    evals = 0
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
        tag_ub = 0.0
        for key, limit in suffix_sets.items():
            count = limit + work_counts.get(key, 0)
            if count >= 2:
                for k, v in set_2pc_sparse[key]:
                    ub[k] += v
            if count >= 4:
                for k, v in set_4pc_sparse[key]:
                    ub[k] += v
                tag_ub += set_tag_bonus[key]
        for key, count in work_counts.items():
            if key in suffix_sets:
                continue        # already counted above
            if count >= 2:
                for k, v in set_2pc_sparse[key]:
                    ub[k] += v
            if count >= 4:
                for k, v in set_4pc_sparse[key]:
                    ub[k] += v
                tag_ub += set_tag_bonus[key]
        for k in range(n_buckets):
            if ub[k] > static_set_ub[k]:
                ub[k] = static_set_ub[k]
        return ub, min(tag_ub, static_tag_ub)

    def dfs(i: int) -> None:
        nonlocal counter, evals
        # Required-4pc reachability: exact, not just a bound — a branch
        # that can no longer collect 4 pieces has no feasible leaves.
        if required is not None and (
            work_counts.get(required, 0)
            + suffix_set_max[i].get(required, 0) < 4
        ):
            return
        if i == n_slots:
            evals += 1
            if evals > budget:
                raise OptimizeError(
                    f"Search exceeded the evaluation budget "
                    f"({budget:,} builds); lock some slots to narrow "
                    f"the search"
                )
            leaf = list(work_buckets)
            tagged = _fold_sets(leaf, work_counts,
                                set_2pc, set_4pc, set_tag_bonus)
            value, aux = evaluate(leaf, tagged)
            if not feasible(leaf, aux):
                return
            counter += 1
            if len(heap) < top_n:
                heapq.heappush(heap, (value, -counter, tuple(combo_stack)))
            elif value > heap[0][0]:
                heapq.heapreplace(heap, (value, -counter, tuple(combo_stack)))
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
            # set bound before descending.
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

    dfs(0)
    return heap, evals


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

    # --- Baseline: validates the whole config (discs included) ------------
    baseline_results = calculate(config, **data)
    baseline_value = getattr(baseline_results, options.objective)
    equipped = {d.slot: d for d in config.discs}
    agent = agents[config.agent_key]

    # --- Candidates per slot (plan §2): inventory + equipped virtuals -----
    dominance_indices = _DAMAGE_INDICES + tuple(
        _CONSTRAINT_EXTRA_INDICES[stat]
        for stat in options.min_stats if stat in _CONSTRAINT_EXTRA_INDICES
    )
    candidates: dict[int, list[_Candidate]] = {}
    pruned_count = 0
    for slot in (1, 2, 3, 4, 5, 6):
        current = equipped.get(slot)
        slot_candidates: list[_Candidate] = []
        if current is not None:
            current_id = next(
                (i for i, d in user_discs.items() if d == current), None
            )
            slot_candidates.append(_Candidate(
                current_id, current,
                _disc_buckets(current, disc_data, agent.attribute),
            ))
        if slot not in options.locked_slots:
            for disc_id, disc in user_discs.items():
                if disc.slot != slot or disc == current:
                    continue
                slot_candidates.append(_Candidate(
                    disc_id, disc,
                    _disc_buckets(disc, disc_data, agent.attribute),
                ))
        if _prune and len(slot_candidates) > 1:
            before = len(slot_candidates)
            slot_candidates = _prune_dominated(slot_candidates,
                                               dominance_indices)
            pruned_count += before - len(slot_candidates)
        if slot_candidates:
            candidates[slot] = slot_candidates

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
                             mode="direct", dealt_element=agent.attribute)

    base_buckets = _bucket_vector(engine.advanced_stat)

    base_crit_rate = consts.base_crit_rate + agent.core_bonus_crit_rate
    base_crit_dmg = consts.base_crit_dmg + agent.core_bonus_crit_dmg
    agent_plus_engine_atk = agent.total_base_atk() + engine.base_atk
    kit_atk_pct = kit["atk_pct"]
    kit_flat_atk = kit["flat_atk"]
    # Engine always-on CRIT Rate (Qingming Birdcage) joins the crit total
    # exactly as in calculate().
    crit_rate_const = (config.external_crit_rate + kit["crit_rate"]
                       + engine.passive_crit(rank))
    crit_dmg_const = config.external_crit_dmg + kit["crit_dmg"]
    # Kit/support PEN Ratio (e.g. Rina's squad PEN Ratio) — a constant that
    # calculate() folds into the DEF zone (build.pen_ratio + kit pen_ratio);
    # the fast path must too, or a support granting PEN Ratio under-values
    # every build's DefMult (PEN interacts non-linearly, so it is not a flat
    # scalar — hence the §4 re-check catches the omission). Same for enemy
    # DEF reduction (Qingyi M1), a separate factor in the DEF zone.
    pen_ratio_const = kit["pen_ratio"]
    def_red_const = kit["def_red"]
    ap_const = (agent.base_anomaly_proficiency
                + agent.core_bonus_anomaly_proficiency)
    am_const = agent.base_anomaly_mastery + agent.core_bonus_anomaly_mastery
    bonus_const = (
        sum(config.external_dmg_bonuses)
        + sum(_engine_passive_bonuses(engine, agent.attribute, rank))
        + _engine_buff_bonus(engine, config.engine_buff_stacks,
                             agent.attribute, rank,
                             skill_tag=config.skill_tag,
                             counts_as_aftershock=config.counts_as_aftershock)
        + sum(kit["dmg_bonus"])
    )
    level_coef = consts.level_coefficient(agent.level)
    crit_cap = consts.crit_rate_cap
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
    stunned = options.objective.endswith("_stunned")
    stun = formulas.stun_mult(
        stunned, stun_base,
        list(config.external_daze_bonuses) + kit["daze_bonus"],
    )
    crit_variant = options.objective.replace("_stunned", "")
    const_mult = res * taken * stun * config.skill_multiplier

    def evaluate(buckets: list[float], tagged: float) -> tuple[float, float]:
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
        bonus = 1.0 + buckets[_B_ATTR_DMG] + bonus_const + tagged
        eff_def = formulas.effective_def(
            boss.base_def, buckets[_B_PEN_RATIO] + pen_ratio_const,
            buckets[_B_PEN_FLAT], def_red=def_red_const
        )
        defense = level_coef / (level_coef + eff_def)
        return atk_total * crit_mult * bonus * defense * const_mult, atk_total

    def feasible(buckets: list[float], atk_total: float) -> bool:
        """Whether a bucket total meets every ``min_stats`` minimum."""
        for stat, minimum in options.min_stats.items():
            if stat == "ATK":
                total = atk_total
            elif stat == "CRIT Rate":
                total = base_crit_rate + buckets[_B_CRIT_RATE] + crit_rate_const
            elif stat == "CRIT DMG":
                total = base_crit_dmg + buckets[_B_CRIT_DMG] + crit_dmg_const
            elif stat == "PEN Ratio":
                total = buckets[_B_PEN_RATIO] + pen_ratio_const
            elif stat == "PEN":
                total = buckets[_B_PEN_FLAT]
            elif stat == "Anomaly Proficiency":
                total = ap_const + buckets[_B_AP]
            else:   # "Anomaly Mastery" (keys validated above)
                total = am_const + buckets[_B_AM]
            if total < minimum - 1e-12:
                return False
        return True

    # --- Per-set precomputation (§5 policy) --------------------------------
    baseline_counts: dict[str, int] = {}
    for disc in config.discs:
        if disc.disc_set is not None:
            baseline_counts[disc.disc_set] = (
                baseline_counts.get(disc.disc_set, 0) + 1
            )

    def assumed_stacks(key: str) -> int:
        entry = disc_sets[key]
        if entry.bonus_4pc is None:
            return 0
        if baseline_counts.get(key, 0) >= 4:
            return config.set_stacks.get(key, 0)
        return entry.bonus_4pc.max_stacks if options.set_assumption == "max" else 0

    set_2pc: dict[str, tuple[float, ...]] = {}
    set_4pc: dict[str, tuple[float, ...]] = {}
    set_stacks_used: dict[str, int] = {}
    set_tag_bonus: dict[str, float] = {}
    candidate_sets = {
        c.disc.disc_set
        for slot_list in candidates.values() for c in slot_list
        if c.disc.disc_set is not None
    }
    teammate_present = bool(config.supports)
    for key in candidate_sets:
        entry = disc_sets[key]
        set_2pc[key] = _bucket_vector(entry.bonus_2pc)
        stacks = assumed_stacks(key)
        set_stacks_used[key] = stacks
        four_stats: dict[str, float] = {}
        if entry.bonus_4pc is not None and stacks:
            four_stats[entry.bonus_4pc.stat] = entry.bonus_4pc.per_stack * stacks
            # Extra part granted only at MAX stacks (mirrors set_bonus_stats;
            # stats without a bucket — Yunkui Tales' Sheer DMG% — drop out
            # in _bucket_vector, as before).
            b4 = entry.bonus_4pc
            if b4.at_max_extra_stat is not None and stacks == b4.max_stacks:
                four_stats[b4.at_max_extra_stat] = (
                    four_stats.get(b4.at_max_extra_stat, 0.0)
                    + b4.at_max_extra_value
                )
        set_4pc[key] = _bucket_vector(four_stats)
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
        set_tag_bonus[key] = tag_skill + auto

    def combo_set_stacks(counts: dict[str, int]) -> dict[str, int]:
        """The ``CalcConfig.set_stacks`` this combination is evaluated at."""
        return {
            key: set_stacks_used[key]
            for key, count in counts.items()
            if count >= 4 and set_stacks_used[key]
        }

    # --- Baseline feasibility (constraints may exclude the baseline) ------
    base_bb = list(base_buckets)
    for disc in config.discs:
        for k, v in enumerate(_disc_buckets(disc, disc_data, agent.attribute)):
            base_bb[k] += v
    _fold_sets(base_bb, baseline_counts, set_2pc, set_4pc, set_tag_bonus)
    _, baseline_atk = evaluate(base_bb, 0.0)
    baseline_feasible = feasible(base_bb, baseline_atk) and (
        required is None or baseline_counts.get(required, 0) >= 4
    )

    # --- Branch-and-bound DFS (§11 E2), shared search core -----------------
    heap, evals = _search(
        candidates, slots, equipped, base_buckets, evaluate, feasible,
        set_2pc, set_4pc, set_tag_bonus, candidate_sets, required,
        options.top_n, options.combo_budget, _bound,
    )

    def verify(combo: tuple[_Candidate, ...], fast_value: float) -> BuildOption:
        """Re-run one build through calculate() (exactness guarantee, §4)."""
        discs = [c.disc for c in combo]
        counts: dict[str, int] = {}
        for disc in discs:
            if disc.disc_set is not None:
                counts[disc.disc_set] = counts.get(disc.disc_set, 0) + 1
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
        )

    # Highest value first; on exact ties the earlier build wins, and the
    # baseline is the first build evaluated, so a tie resolves toward the
    # current discs.
    ranked = sorted(heap, key=lambda item: (-item[0], -item[1]))
    options_verified = [verify(combo, value) for value, _, combo in ranked]

    if not options_verified and not baseline_feasible:
        parts = []
        if required is not None:
            parts.append(f"the required 4-piece set "
                         f"({disc_sets[required].name})")
        if options.min_stats:
            parts.append("the minimum-stat constraints")
        raise OptimizeError(
            f"No combination of your saved discs meets "
            f"{' and '.join(parts)}; relax them and optimize again"
        )

    best = options_verified[0] if options_verified else None
    already_optimal = baseline_feasible and (
        best is None
        or best.value <= baseline_value * (1.0 + _RECHECK_RTOL)
    )
    if already_optimal:
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

    # --- Baseline: validates the whole config (mode, discs, anomaly) -------
    baseline_results = calculate_anomaly(config, **data, anomaly_data=anomaly_data)
    baseline_value = getattr(baseline_results, options.objective)
    equipped = {d.slot: d for d in config.discs}
    agent = agents[config.agent_key]
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
        abloom_key = (config.abloom_element or agent.attribute).lower()
        anomaly = anomaly_data.anomalies[abloom_key]
        kind, dealt_element, hits = "anomaly", abloom_key, 1
    else:
        anomaly = anomaly_data.anomalies[agent.attribute]
        kind, dealt_element, hits = "anomaly", agent.attribute, anomaly.hits
    # Build Attribute DMG% (disc + set + engine advanced) only counts when
    # the burst deals the agent's OWN attribute — mirrors calculate_anomaly.
    attr_dmg_applies = (dealt_element == agent.attribute)

    # --- Candidates per slot (plan §2): inventory + equipped virtuals -----
    damage_indices = (_ANOMALY_DAMAGE_INDICES if attr_dmg_applies
                      else _ANOMALY_DAMAGE_INDICES_NO_ATTR)
    dominance_indices = damage_indices + tuple(
        _CONSTRAINT_STAT_INDEX[stat]
        for stat in options.min_stats
        if stat in _CONSTRAINT_STAT_INDEX
        and _CONSTRAINT_STAT_INDEX[stat] not in damage_indices
    )
    candidates: dict[int, list[_Candidate]] = {}
    pruned_count = 0
    for slot in (1, 2, 3, 4, 5, 6):
        current = equipped.get(slot)
        slot_candidates: list[_Candidate] = []
        if current is not None:
            current_id = next(
                (i for i, d in user_discs.items() if d == current), None
            )
            slot_candidates.append(_Candidate(
                current_id, current,
                _disc_buckets(current, disc_data, agent.attribute),
            ))
        if slot not in options.locked_slots:
            for disc_id, disc in user_discs.items():
                if disc.slot != slot or disc == current:
                    continue
                slot_candidates.append(_Candidate(
                    disc_id, disc,
                    _disc_buckets(disc, disc_data, agent.attribute),
                ))
        if _prune and len(slot_candidates) > 1:
            before = len(slot_candidates)
            slot_candidates = _prune_dominated(slot_candidates,
                                               dominance_indices)
            pruned_count += before - len(slot_candidates)
        if slot_candidates:
            candidates[slot] = slot_candidates

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

    base_crit_rate = consts.base_crit_rate + agent.core_bonus_crit_rate
    base_crit_dmg = consts.base_crit_dmg + agent.core_bonus_crit_dmg
    agent_plus_engine_atk = agent.total_base_atk() + engine.base_atk
    kit_atk_pct = kit["atk_pct"]
    kit_flat_atk = kit["flat_atk"]
    pen_ratio_const = kit["pen_ratio"]
    def_red_const = kit["def_red"]
    crit_rate_const = config.external_crit_rate + kit["crit_rate"]
    crit_dmg_const = config.external_crit_dmg + kit["crit_dmg"]
    ap_const = (agent.base_anomaly_proficiency
                + agent.core_bonus_anomaly_proficiency
                + engine.passive_ap(rank)
                + kit["anomaly_proficiency"]
                + config.external_anomaly_proficiency)
    am_const = agent.base_anomaly_mastery + agent.core_bonus_anomaly_mastery

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
    stunned = options.objective.endswith("_stunned")
    stun = formulas.stun_mult(
        stunned, stun_base,
        list(config.external_daze_bonuses) + kit["daze_bonus"],
    )
    # "full" multiplies the per-proc value by the anomaly's proc count.
    hits_factor = hits if options.objective.startswith("full") else 1
    outer = level_mult * res * taken * stun * hits_factor

    # --- Per-proc / burst multiplier (constant, or AP-scaled per build) ----
    disorder_additive = config.external_disorder_mult_add + kit["disorder_mult_add"]
    if kind == "vortex":
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

    def evaluate(buckets: list[float], tagged: float) -> tuple[float, tuple]:
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
        bonus = 1.0 + bonus_const + tagged
        if attr_dmg_applies:
            bonus += buckets[_B_ATTR_DMG]
        eff_def = formulas.effective_def(
            boss.base_def, buckets[_B_PEN_RATIO] + pen_ratio_const,
            buckets[_B_PEN_FLAT], def_red=def_red_const,
        )
        defense = level_coef / (level_coef + eff_def)
        value = (mult_of_ap(ap_total) * atk_total * formulas.ap_mult(ap_total)
                 * bonus * abuff * defense * outer)
        return value, (atk_total, ap_total)

    def feasible(buckets: list[float], aux: tuple) -> bool:
        """Whether a bucket total meets every ``min_stats`` minimum."""
        atk_total, ap_total = aux
        for stat, minimum in options.min_stats.items():
            if stat == "ATK":
                total = atk_total
            elif stat == "CRIT Rate":
                total = base_crit_rate + buckets[_B_CRIT_RATE] + crit_rate_const
            elif stat == "CRIT DMG":
                total = base_crit_dmg + buckets[_B_CRIT_DMG] + crit_dmg_const
            elif stat == "PEN Ratio":
                total = buckets[_B_PEN_RATIO] + pen_ratio_const
            elif stat == "PEN":
                total = buckets[_B_PEN_FLAT]
            elif stat == "Anomaly Proficiency":
                total = ap_total
            else:   # "Anomaly Mastery" (keys validated above)
                total = am_const + buckets[_B_AM]
            if total < minimum - 1e-12:
                return False
        return True

    # --- Per-set precomputation (§5 policy) --------------------------------
    baseline_counts: dict[str, int] = {}
    for disc in config.discs:
        if disc.disc_set is not None:
            baseline_counts[disc.disc_set] = (
                baseline_counts.get(disc.disc_set, 0) + 1
            )

    def assumed_stacks(key: str) -> int:
        entry = disc_sets[key]
        if entry.bonus_4pc is None:
            return 0
        if baseline_counts.get(key, 0) >= 4:
            return config.set_stacks.get(key, 0)
        return entry.bonus_4pc.max_stacks if options.set_assumption == "max" else 0

    set_2pc: dict[str, tuple[float, ...]] = {}
    set_4pc: dict[str, tuple[float, ...]] = {}
    set_stacks_used: dict[str, int] = {}
    set_tag_bonus: dict[str, float] = {}
    candidate_sets = {
        c.disc.disc_set
        for slot_list in candidates.values() for c in slot_list
        if c.disc.disc_set is not None
    }
    teammate_present = bool(config.supports)
    for key in candidate_sets:
        entry = disc_sets[key]
        set_2pc[key] = _bucket_vector(entry.bonus_2pc)
        stacks = assumed_stacks(key)
        set_stacks_used[key] = stacks
        four_stats: dict[str, float] = {}
        if entry.bonus_4pc is not None and stacks:
            four_stats[entry.bonus_4pc.stat] = entry.bonus_4pc.per_stack * stacks
            # Extra part granted only at MAX stacks (mirrors set_bonus_stats;
            # stats without a bucket — Yunkui Tales' Sheer DMG% — drop out
            # in _bucket_vector, as before).
            b4 = entry.bonus_4pc
            if b4.at_max_extra_stat is not None and stacks == b4.max_stacks:
                four_stats[b4.at_max_extra_stat] = (
                    four_stats.get(b4.at_max_extra_stat, 0.0)
                    + b4.at_max_extra_value
                )
        set_4pc[key] = _bucket_vector(four_stats)
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
        set_tag_bonus[key] = tag_skill + auto

    def combo_set_stacks(counts: dict[str, int]) -> dict[str, int]:
        """The ``CalcConfig.set_stacks`` this combination is evaluated at."""
        return {
            key: set_stacks_used[key]
            for key, count in counts.items()
            if count >= 4 and set_stacks_used[key]
        }

    # --- Baseline feasibility (constraints may exclude the baseline) ------
    base_bb = list(base_buckets)
    for disc in config.discs:
        for k, v in enumerate(_disc_buckets(disc, disc_data, agent.attribute)):
            base_bb[k] += v
    _fold_sets(base_bb, baseline_counts, set_2pc, set_4pc, set_tag_bonus)
    _, baseline_aux = evaluate(base_bb, 0.0)
    baseline_feasible = feasible(base_bb, baseline_aux) and (
        required is None or baseline_counts.get(required, 0) >= 4
    )

    # --- Branch-and-bound DFS (§11 E2), shared search core -----------------
    heap, evals = _search(
        candidates, slots, equipped, base_buckets, evaluate, feasible,
        set_2pc, set_4pc, set_tag_bonus, candidate_sets, required,
        options.top_n, options.combo_budget, _bound,
    )

    def verify(combo: tuple[_Candidate, ...], fast_value: float) -> BuildOption:
        """Re-run one build through calculate_anomaly (exactness, §4)."""
        discs = [c.disc for c in combo]
        counts: dict[str, int] = {}
        for disc in discs:
            if disc.disc_set is not None:
                counts[disc.disc_set] = counts.get(disc.disc_set, 0) + 1
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
        )

    ranked = sorted(heap, key=lambda item: (-item[0], -item[1]))
    options_verified = [verify(combo, value) for value, _, combo in ranked]

    if not options_verified and not baseline_feasible:
        parts = []
        if required is not None:
            parts.append(f"the required 4-piece set "
                         f"({disc_sets[required].name})")
        if options.min_stats:
            parts.append("the minimum-stat constraints")
        raise OptimizeError(
            f"No combination of your saved discs meets "
            f"{' and '.join(parts)}; relax them and optimize again"
        )

    best = options_verified[0] if options_verified else None
    already_optimal = baseline_feasible and (
        best is None
        or best.value <= baseline_value * (1.0 + _RECHECK_RTOL)
    )
    if already_optimal:
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
        combos_evaluated=evals,
        candidates_per_slot={s: len(candidates[s]) for s in slots},
        discs_pruned=pruned_count,
    )
