"""Pure probability functions for freshly farmed drive discs.

Every function here takes plain values (weights, probabilities, stat names)
and returns plain values — no I/O, no game data. The weights come from
``data/disc_drops.json`` via :func:`~.discs.load_drop_model`; the advisor
(:mod:`.disc_advisor`) composes these pieces (DOCS/disc_advice_plan.md).

Roll model (S-rank, confirmed mechanics — the weights are provisional):

- The disc starts with ``k`` substats (probability ``initial[k]``), drawn
  one at a time by weight **without replacement** from the legal pool
  (every substat except the main stat).
- It then gets ``upgrades`` upgrades. The first ``n - k`` reveal the
  missing substats (drawn the same way); each remaining upgrade adds one
  roll to one of the ``n`` substats, picked **uniformly**.
- Every substat therefore ends with ``1 + (upgrades it received)`` rolls.

Because the reveal draws use the same weights as the initial draws, the
final substat *set* is distributed like ``n`` weighted draws without
replacement whatever ``k`` is; only the number of free upgrades
(``upgrades - (n - k)``) depends on the start.
"""

from __future__ import annotations

from itertools import combinations, permutations
from math import factorial, prod
from typing import Iterator

#: A collapsed outcome key: sorted ``(substat, rolls)`` pairs of the
#: *relevant* substats only (irrelevant ones are dropped).
OutcomeKey = tuple[tuple[str, int], ...]


def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Every way to split ``total`` upgrades over ``parts`` substats."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def _multinomial(counts: tuple[int, ...]) -> float:
    """Probability of exactly ``counts`` when each upgrade picks one of
    ``len(counts)`` substats uniformly."""
    total = sum(counts)
    return (factorial(total) / prod(factorial(c) for c in counts)
            / len(counts) ** total)


def _subset_probability(subset: tuple[str, ...],
                        weights: dict[str, float]) -> float:
    """Probability that ``len(subset)`` weighted draws without
    replacement from ``weights`` pick exactly ``subset`` (any order)."""
    pool_total = sum(weights.values())
    probability = 0.0
    for order in permutations(subset):
        p = 1.0
        remaining = pool_total
        for stat in order:
            p *= weights[stat] / remaining
            remaining -= weights[stat]
        probability += p
    return probability


def substat_distribution(
    weights: dict[str, float],
    main_stat: str,
    initial: dict[int, float],
    upgrades: int,
    relevant: frozenset[str] | set[str],
    substats_per_disc: int = 4,
) -> dict[OutcomeKey, tuple[float, dict[str, int]]]:
    """The exact Lv. 15 substat distribution of a new disc.

    Args:
        weights: substat name -> relative draw weight (the whole pool;
            ``main_stat`` is removed here, as the game never rolls it).
        main_stat: the disc's main stat name.
        initial: starting substat count -> probability.
        upgrades: upgrades on the way to max level.
        relevant: substats the caller can tell apart. Outcomes that differ
            only in the other substats collapse into one key.
        substats_per_disc: substats a max-level disc carries.

    Returns:
        key -> ``(probability, example)``. ``example`` is one full
        substat -> rolls dict that produces the key (a legal disc's
        substats, for re-checking a collapsed outcome). Probabilities sum
        to 1.
    """
    pool = {stat: w for stat, w in weights.items() if stat != main_stat}
    probs: dict[OutcomeKey, float] = {}
    examples: dict[OutcomeKey, dict[str, int]] = {}
    for subset in combinations(sorted(pool), substats_per_disc):
        p_subset = _subset_probability(subset, pool)
        for start, p_start in initial.items():
            free = upgrades - (substats_per_disc - start)
            for counts in _compositions(free, substats_per_disc):
                p = p_start * p_subset * _multinomial(counts)
                rolls = {stat: 1 + c for stat, c in zip(subset, counts)}
                key = tuple(sorted((stat, r) for stat, r in rolls.items()
                                   if stat in relevant))
                if key not in probs:
                    probs[key] = 0.0
                    examples[key] = rolls
                probs[key] += p
    return {key: (probs[key], examples[key]) for key in probs}


def main_stat_probability(
    slot_weights: dict[tuple[str, str | None], float],
    main_stat: str,
    element: str | None = None,
) -> float:
    """Probability that a new disc of the slot rolls ``main_stat``.

    ``slot_weights`` maps ``(main stat, element)`` to a relative weight,
    where ``element`` is set only for the per-element ``Attribute DMG%``
    variants. Pass ``element`` for those (the variant must match); any
    other main ignores it. Returns 0 for a main the slot cannot roll.
    """
    key = (main_stat, element if (main_stat, element) in slot_weights
           else None)
    weight = slot_weights.get(key, 0.0)
    return weight / sum(slot_weights.values()) if weight else 0.0
