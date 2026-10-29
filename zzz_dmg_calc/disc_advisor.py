"""Disc contribution and "replace this disc first" advisor.

Design: DOCS/disc_advice_plan.md. For the build the user just calculated,
report how much each equipped disc contributes to the damage and which
slot is the best farming target — the slot with the highest expected
damage gain per S-rank drop.

Entry points::

    analyze_discs(config, objective) -> DiscAdvice          # direct hits
    analyze_discs_anomaly(config, objective) -> DiscAdvice  # anomaly family
    analyze_discs_sheer(config, objective) -> DiscAdvice    # Sheer (Rupture)

Method: the optimizer's exact fast path (one bucket vector per disc, see
:func:`~.optimizer._prepare_fast_path`) values a whole build in
microseconds, so every possible replacement disc can be valued exactly.
Set bonuses are folded once for the equipped set counts and held fixed:
a replacement keeps the disc's set whenever a set bonus depends on it.

Per equipped disc, with ``D`` the baseline objective value:

- **Damage share** ``1 − V(build without the disc) / D`` and **substat
  share** ``1 − V(disc with its main stat only) / D`` — leave-one-out
  losses. Damage multiplies, so the shares do not add up to 100%.
- **Replacement model**: a new S-rank disc for the same slot with the
  same main stat (and element), whose substats follow the drop model
  (``data/disc_drops.json``, :mod:`.disc_odds`), enumerated exactly.
  Substats this mode's damage never reads collapse into one outcome.
- **P(beat)**: chance the new disc deals more damage; **percentile** is
  its complement. **Drops needed** = 1 / (P(set) · P(slot) · P(main) ·
  P(beat)); **gain per drop** = P(set) · P(slot) · P(main) ·
  E[max(V/D − 1, 0)]. P(set) is 1 / ``sets_per_source`` when the disc
  is *load-bearing* (its set has exactly 2 or 4 pieces, so losing it
  breaks a bonus), else 1 — any set will do.
- The recommendation is the slot with the highest gain per drop.

Replacements must keep the automatic panel minimums of the main's
self-scaling kit inputs (Astra's claimed initial ATK) that the current
build meets — the optimizer's rule; outcomes below them never count.

Exactness (optimizer §4): the fast-path baseline must equal the mode's
calculation to 1e-9, and each slot's best possible replacement is
rebuilt as a real :class:`~.discs.Disc` and re-run through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .agent import Agent, load_agents
from .anomalies import AnomalyData, load_anomalies
from .api import CalcConfig, calculate, calculate_anomaly, calculate_sheer
from .constants import Constants, load_constants
from .disc_odds import main_stat_probability, substat_distribution
from .discs import (
    Disc, DiscData, DiscSet, DropModel, load_disc_data, load_disc_sets,
    load_drop_model, set_4pc_max_stacks,
)
from .enemies import Boss, load_bosses
from .engines import Engine, load_engines
from .optimizer import (
    ANOMALY_OBJECTIVES, CONSTRAINT_STATS, OBJECTIVES, SHEER_CONSTRAINT_STATS,
    OptimizeOptions, _CONSTRAINT_BUCKETS, _anomaly_model, _bucket_vector,
    _direct_model, _disc_buckets, _fold_sets, _make_feasible, _panel_feasible,
    _prepare_fast_path, _set_vectors, _sheer_model,
)

#: Relative tolerance: a replacement must beat the baseline by more than
#: this to count as an upgrade (exact ties are not upgrades), and the fast
#: path must agree with the exact calculation within it.
_RTOL = 1e-9


class AdviceError(ValueError):
    """Raised for an unknown objective, a config the advisor can't value,
    or a fast-path / calculation disagreement (an internal error)."""


@dataclass(frozen=True)
class SlotAdvice:
    """One equipped disc's contribution and upgrade outlook.

    Attributes:
        slot: Partition 1-6.
        disc: The equipped disc.
        damage_share: Fraction of the damage lost without the disc.
        substat_share: Fraction lost if the disc kept only its main stat.
        useful_rolls: Rolls on substats this mode's damage reads.
        total_rolls: All substat rolls on the disc.
        load_bearing: Losing the disc would break a set bonus, so the
            replacement must come from the same set.
        p_set: Chance a drop from the disc's farming source has a usable
            set (1 / sets_per_source when load-bearing, else 1).
        p_slot: Chance a drop is this slot.
        p_main: Chance a drop of this slot has the disc's main stat.
        p_beat: Chance a new same-slot, same-main disc deals more damage.
        percentile: Chance such a disc is no better (``1 - p_beat``).
        avg_gain_when_beat: Mean fractional gain over the upgrades
            (0 when none exists).
        expected_gain: E[max(V/D - 1, 0)] over same-slot, same-main discs.
        ceiling_gain: Fractional gain of the best possible replacement
            (<= 0 when the disc already is the best roll).
        ceiling_disc: That best possible replacement.
        drops_needed: Expected S-rank drops until an upgrade, or ``None``
            when none can happen.
        gain_per_drop: Expected fractional gain per S-rank drop — the
            ranking key.
    """

    slot: int
    disc: Disc
    damage_share: float
    substat_share: float
    useful_rolls: int
    total_rolls: int
    load_bearing: bool
    p_set: float
    p_slot: float
    p_main: float
    p_beat: float
    percentile: float
    avg_gain_when_beat: float
    expected_gain: float
    ceiling_gain: float
    ceiling_disc: Disc | None
    drops_needed: float | None
    gain_per_drop: float


@dataclass(frozen=True)
class DiscAdvice:
    """Outcome of :func:`analyze_discs` and its mode twins.

    ``recommended_slot`` is the equipped slot with the highest gain per
    drop (``None`` when no disc can be improved). ``empty_slots`` have no
    disc at all — any disc there is an upgrade, so they are not ranked.
    ``relevant_substats`` are the substats this mode's damage reads.
    ``gated_sets``: equipped sets whose 4pc no squad member can trigger
    (credited at 0 stacks, optimizer §18).
    """

    objective: str
    baseline_value: float
    slots: tuple[SlotAdvice, ...]
    recommended_slot: int | None
    empty_slots: tuple[int, ...]
    relevant_substats: tuple[str, ...]
    provisional: bool
    sets_per_source: int
    gated_sets: dict[str, str] = field(default_factory=dict)


def _analyze(
    config: CalcConfig,
    objective: str,
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
    drop_model: DropModel | None,
) -> DiscAdvice:
    """The analysis every mode shares (see the module docstring)."""
    if objective not in objectives:
        raise AdviceError(
            f"Unknown objective '{objective}'; options: {list(objectives)}"
        )
    consts = consts if consts is not None else load_constants()
    disc_data = disc_data if disc_data is not None else load_disc_data()
    bosses = bosses if bosses is not None else load_bosses()
    agents = agents if agents is not None else load_agents()
    engines = engines if engines is not None else load_engines()
    disc_sets = disc_sets if disc_sets is not None else load_disc_sets()
    if drop_model is None:
        drop_model = load_drop_model(disc_data)
    data = dict(consts=consts, disc_data=disc_data, bosses=bosses,
                agents=agents, engines=engines, disc_sets=disc_sets)

    fast = _prepare_fast_path(
        config, OptimizeOptions(objective=objective), data=data, calc=calc,
        build_model=build_model, anomaly_data=anomaly_data,
        constraint_stats=constraint_stats)
    config, model, attr = fast.config, fast.model, fast.attr
    evaluate = model.evaluate
    baseline = fast.baseline_value
    if baseline <= 0:
        raise AdviceError(
            "The build deals no damage for this objective — nothing to "
            "compare discs against"
        )

    # --- The equipped build, with its set bonuses held fixed --------------
    counts: dict[str, int] = {}
    for disc in config.discs:
        if disc.disc_set is not None:
            counts[disc.disc_set] = counts.get(disc.disc_set, 0) + 1

    def stacks_for(key: str) -> int:
        # The optimizer's rule for a set the baseline already wears.
        if not set_4pc_max_stacks(disc_sets[key]) or key in fast.gated:
            return 0
        return config.set_stacks.get(key, 0)

    set_2pc, set_4pc, _stacks, set_tag = _set_vectors(
        counts, disc_sets=disc_sets, attr=attr, agent=fast.agent,
        config=config, model=model, stacks_for=stacks_for)
    build = list(fast.base_buckets)
    disc_vectors: dict[int, tuple[float, ...]] = {}
    for disc in config.discs:
        vector = _disc_buckets(disc, disc_data, attr)
        disc_vectors[disc.slot] = vector
        for k, v in enumerate(vector):
            build[k] += v
    tagged = _fold_sets(build, counts, set_2pc, set_4pc, set_tag)
    fast_value, _aux = evaluate(build, tagged)
    if abs(fast_value - baseline) > _RTOL * max(1.0, abs(baseline)):
        raise AdviceError(
            f"Internal error: fast evaluation ({fast_value!r}) disagrees "
            f"with {calc_name}() ({baseline!r}) — please report this build"
        )

    # --- Panel minimums the build meets, and the substats that matter -----
    kept_minimums = {
        stat: minimum
        for stat, minimum in fast.options.min_panel_stats.items()
        if _panel_feasible({stat: minimum}, fast.panel_consts, build)
    }
    feasible = (
        _make_feasible(replace(fast.options, min_stats={},
                               min_panel_stats=kept_minimums),
                       fast.panel_consts, fast.stat_total)
        if kept_minimums else None)
    watched = set(model.damage_indices)
    for stat in kept_minimums:
        watched.update(_CONSTRAINT_BUCKETS.get(stat, ()))
    unit = {stat: [(k, v) for k, v in enumerate(_bucket_vector({stat: roll}))
                   if v]
            for stat, roll in disc_data.substat_rolls.items()}
    relevant = frozenset(stat for stat, parts in unit.items()
                         if any(k in watched for k, _v in parts))

    # --- Each equipped disc ----------------------------------------------
    p_slot = 1.0 / len(disc_data.main_stats)
    distributions: dict[str, dict] = {}
    slots: list[SlotAdvice] = []
    for disc in sorted(config.discs, key=lambda d: d.slot):
        full = disc_vectors[disc.slot]
        subs = _bucket_vector({stat: rolls * disc_data.substat_rolls[stat]
                               for stat, rolls in disc.substats.items()})
        without = [b - f for b, f in zip(build, full)]
        main_only = [b - s for b, s in zip(build, subs)]
        damage_share = 1.0 - evaluate(without, tagged)[0] / baseline
        substat_share = 1.0 - evaluate(main_only, tagged)[0] / baseline

        if disc.main_stat not in distributions:
            distributions[disc.main_stat] = substat_distribution(
                drop_model.substat_weights, disc.main_stat,
                drop_model.initial_substats, drop_model.upgrades, relevant,
                disc_data.substats_per_disc)
        p_beat = expected_gain = 0.0
        best_value = best_example = None
        for key, (p, example) in distributions[disc.main_stat].items():
            vector = list(main_only)
            for stat, rolls in key:
                for k, v in unit[stat]:
                    vector[k] += rolls * v
            value, aux = evaluate(vector, tagged)
            if feasible is not None and not feasible(vector, aux):
                continue
            gain = value / baseline - 1.0
            if gain > _RTOL:
                p_beat += p
                expected_gain += p * gain
            if best_value is None or value > best_value:
                best_value, best_example = value, example

        ceiling_disc = None
        ceiling_gain = 0.0
        if best_value is not None:
            ceiling_disc = Disc(slot=disc.slot, main_stat=disc.main_stat,
                                substats=dict(best_example),
                                disc_set=disc.disc_set, element=disc.element)
            discs = [ceiling_disc if d.slot == disc.slot else d
                     for d in config.discs]
            exact = getattr(calc(replace(config, discs=discs), data),
                            objective)
            if abs(best_value - exact) > _RTOL * max(1.0, abs(exact)):
                raise AdviceError(
                    f"Internal error: fast evaluation ({best_value!r}) "
                    f"disagrees with {calc_name}() ({exact!r}) for a slot "
                    f"{disc.slot} replacement — please report this build"
                )
            ceiling_gain = exact / baseline - 1.0

        load_bearing = counts.get(disc.disc_set, 0) in (2, 4)
        p_set = 1.0 / drop_model.sets_per_source if load_bearing else 1.0
        element = None
        if disc.main_stat == "Attribute DMG%":
            # A legacy element-less disc counts as matching (optimizer §11).
            element = disc.element if disc.element is not None else attr
        p_main = main_stat_probability(
            drop_model.main_stat_weights[disc.slot], disc.main_stat, element)
        per_drop = p_set * p_slot * p_main
        slots.append(SlotAdvice(
            slot=disc.slot,
            disc=disc,
            damage_share=damage_share,
            substat_share=substat_share,
            useful_rolls=sum(rolls for stat, rolls in disc.substats.items()
                             if stat in relevant),
            total_rolls=sum(disc.substats.values()),
            load_bearing=load_bearing,
            p_set=p_set,
            p_slot=p_slot,
            p_main=p_main,
            p_beat=p_beat,
            percentile=1.0 - p_beat,
            avg_gain_when_beat=expected_gain / p_beat if p_beat else 0.0,
            expected_gain=expected_gain,
            ceiling_gain=ceiling_gain,
            ceiling_disc=ceiling_disc,
            drops_needed=(1.0 / (per_drop * p_beat)
                          if per_drop * p_beat > 0 else None),
            gain_per_drop=per_drop * expected_gain,
        ))

    ranked = [s for s in slots if s.gain_per_drop > 0]
    recommended = (max(ranked, key=lambda s: (s.gain_per_drop, s.p_beat,
                                              -s.slot)).slot
                   if ranked else None)
    equipped_slots = {d.slot for d in config.discs}
    return DiscAdvice(
        objective=objective,
        baseline_value=baseline,
        slots=tuple(slots),
        recommended_slot=recommended,
        empty_slots=tuple(s for s in sorted(disc_data.main_stats)
                          if s not in equipped_slots),
        relevant_substats=tuple(sorted(relevant)),
        provisional=drop_model.provisional,
        sets_per_source=drop_model.sets_per_source,
        gated_sets={key: element for key, element in fast.gated.items()
                    if key in counts},
    )


def analyze_discs(
    config: CalcConfig,
    objective: str = "average",
    *,
    consts: Constants | None = None,
    disc_data: DiscData | None = None,
    bosses: dict[str, Boss] | None = None,
    agents: dict[str, Agent] | None = None,
    engines: dict[str, Engine] | None = None,
    disc_sets: dict[str, DiscSet] | None = None,
    drop_model: DropModel | None = None,
) -> DiscAdvice:
    """Disc contributions and upgrade odds for a DIRECT-hit objective
    (one of :data:`~.optimizer.OBJECTIVES`). The keyword data arguments
    let the server/tests inject already-loaded data, as in
    :func:`~.api.calculate`.

    Raises:
        AdviceError: unknown objective, a zero-damage baseline, or an
            internal fast-path disagreement.
        CalcError / DiscError / AgentError: the config itself is invalid
            (the same errors ``calculate`` would raise).
    """
    return _analyze(
        config, objective,
        objectives=OBJECTIVES, constraint_stats=CONSTRAINT_STATS,
        calc_name="calculate",
        calc=lambda cfg, data: calculate(cfg, **data),
        build_model=_direct_model,
        consts=consts, disc_data=disc_data, bosses=bosses, agents=agents,
        engines=engines, disc_sets=disc_sets, anomaly_data=None,
        drop_model=drop_model,
    )


def analyze_discs_anomaly(
    config: CalcConfig,
    objective: str = "full",
    *,
    consts: Constants | None = None,
    disc_data: DiscData | None = None,
    bosses: dict[str, Boss] | None = None,
    agents: dict[str, Agent] | None = None,
    engines: dict[str, Engine] | None = None,
    disc_sets: dict[str, DiscSet] | None = None,
    anomaly_data: AnomalyData | None = None,
    drop_model: DropModel | None = None,
) -> DiscAdvice:
    """The anomaly-family twin of :func:`analyze_discs` (objectives from
    :data:`~.optimizer.ANOMALY_OBJECTIVES`). Like the anomaly optimizer
    it values the fully-buffed number, so buildup dilution is rejected.

    Raises:
        AdviceError: as :func:`analyze_discs`, plus buildup segments or a
            non-zero ``unbuffed_share`` on the config.
        CalcError / DiscError / AgentError: the config itself is invalid.
    """
    if config.buildup_segments or config.unbuffed_share:
        raise AdviceError(
            "Disc advice values the fully-buffed anomaly number; buildup "
            "segments / unbuffed_share (buildup dilution) are not "
            "supported — clear them and analyze again"
        )
    anomaly_data = anomaly_data if anomaly_data is not None else load_anomalies()
    return _analyze(
        config, objective,
        objectives=ANOMALY_OBJECTIVES, constraint_stats=CONSTRAINT_STATS,
        calc_name="calculate_anomaly",
        calc=lambda cfg, data: calculate_anomaly(cfg, **data,
                                                 anomaly_data=anomaly_data),
        build_model=_anomaly_model,
        consts=consts, disc_data=disc_data, bosses=bosses, agents=agents,
        engines=engines, disc_sets=disc_sets, anomaly_data=anomaly_data,
        drop_model=drop_model,
    )


def analyze_discs_sheer(
    config: CalcConfig,
    objective: str = "average",
    *,
    consts: Constants | None = None,
    disc_data: DiscData | None = None,
    bosses: dict[str, Boss] | None = None,
    agents: dict[str, Agent] | None = None,
    engines: dict[str, Engine] | None = None,
    disc_sets: dict[str, DiscSet] | None = None,
    drop_model: DropModel | None = None,
) -> DiscAdvice:
    """The Sheer (Rupture) twin of :func:`analyze_discs` — same crit
    objectives, valued through :func:`~.api.calculate_sheer`.

    Raises:
        AdviceError / CalcError / DiscError / AgentError: as
            :func:`analyze_discs` (including a non-rupture agent).
    """
    return _analyze(
        config, objective,
        objectives=OBJECTIVES, constraint_stats=SHEER_CONSTRAINT_STATS,
        calc_name="calculate_sheer",
        calc=lambda cfg, data: calculate_sheer(cfg, **data),
        build_model=_sheer_model,
        consts=consts, disc_data=disc_data, bosses=bosses, agents=agents,
        engines=engines, disc_sets=disc_sets, anomaly_data=None,
        drop_model=drop_model,
    )
