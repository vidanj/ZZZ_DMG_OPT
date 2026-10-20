"""Linear upper bounds on log(damage) for the optimizer's branch-and-bound.

Pure math, like :mod:`.formulas`: plain numbers in, plain numbers out, no
I/O and no game data. The optimizer (DOCS/optimizer_plan.md §17) uses
these pieces to build, for one search, a linear function

    ``L(x) = c0 + Σ w[i] · x[i]``

with ``log(objective(x)) <= L(x)`` for EVERY stat vector ``x`` inside a
box ``lo <= x <= hi`` (the box contains every build the search can
reach). Because ``L`` is linear, the best completion of a partial build is
just the sum of each remaining slot's best disc score ``w · v`` — no
phantom disc that carries every main stat at once, which is what made the
older per-stat bound ~2x loose.

The objective is a product of zones, so ``log`` of it is a sum and each
zone is bounded on its own:

- a positive affine zone (DMG% bracket, AP, ...): ``log`` is concave, so
  its tangent at the reference point ``x0`` lies above it everywhere
  (:meth:`LogLinear.add_log_affine`);
- the ATK zone ``u·v + K`` (sheet ATK × combat ATK% bracket + flat):
  concave in log when ``u·v >= K`` holds on the whole box, else a
  McCormick plane (:meth:`LogLinear.add_log_product`);
- average crit ``1 + min(rate, cap)·dmg``: ``log`` of it is not concave
  where ``rate·dmg < 1``, so a plane with slopes chosen to minimize the
  worst gap, lifted by a RIGOROUS bound of that gap over the box
  (:func:`average_crit_log_upper`; ~2% slack where a McCormick plane
  left ~17%);
- the DEF zone: convex in PEN, so the chord over the box
  (:meth:`LogLinear.add_log_def`).

Every function either returns a VALID bound or raises
:class:`LinearBoundError` (a precondition failed); the optimizer then
falls back to its older per-stat bound, so exactness never depends on a
case being supported.

Affine functions are ``(const, ((index, coef), ...))`` tuples over the
stat vector.
"""

from __future__ import annotations

import math

#: ``(const, ((index, coef), ...))`` — ``const + Σ coef · x[index]``.
Affine = tuple[float, tuple[tuple[int, float], ...]]


class LinearBoundError(ValueError):
    """A precondition for a valid linear bound does not hold."""


def affine(const: float, *terms: tuple[int, float]) -> Affine:
    """Build an affine function, dropping zero coefficients."""
    return (float(const), tuple((i, float(c)) for i, c in terms if c))


def affine_at(f: Affine, x) -> float:
    """``f(x)``."""
    const, terms = f
    return const + sum(coef * x[i] for i, coef in terms)


def affine_range(f: Affine, lo, hi) -> tuple[float, float]:
    """The exact ``(min, max)`` of ``f`` over the box ``lo <= x <= hi``."""
    const, terms = f
    low = high = const
    for i, coef in terms:
        if coef >= 0:
            low += coef * lo[i]
            high += coef * hi[i]
        else:
            low += coef * hi[i]
            high += coef * lo[i]
    return low, high


def affine_combine(*parts: tuple[float, Affine], const: float = 0.0) -> Affine:
    """``const + Σ scale · f`` for ``(scale, f)`` pairs."""
    total = const
    coefs: dict[int, float] = {}
    for scale, (c, terms) in parts:
        total += scale * c
        for i, coef in terms:
            coefs[i] = coefs.get(i, 0.0) + scale * coef
    return affine(total, *coefs.items())


def mccormick_upper(u: Affine, v: Affine, x0, lo, hi) -> Affine:
    """Affine upper bound of ``u(x) · v(x)`` on the box (McCormick).

    Both planes are valid everywhere in the box —
    ``(u_hi − u)(v − v_lo) >= 0`` and ``(u − u_lo)(v_hi − v) >= 0`` —
    so the one lower at ``x0`` is returned. Exact when either factor is
    constant over the box.
    """
    u_lo, u_hi = affine_range(u, lo, hi)
    v_lo, v_hi = affine_range(v, lo, hi)
    plane_a = affine_combine((u_hi, v), (v_lo, u), const=-u_hi * v_lo)
    plane_b = affine_combine((u_lo, v), (v_hi, u), const=-u_lo * v_hi)
    if affine_at(plane_a, x0) <= affine_at(plane_b, x0):
        return plane_a
    return plane_b


def step_scaling_upper(
    base: float, threshold: float, per_step: float, per_value: float,
    cap: float | None, d_lo: float, d_hi: float, d0: float,
) -> tuple[float, float]:
    """Affine upper bound ``k0 + k1 · d`` of a kit scaling's value on
    ``[d_lo, d_hi]`` (``agent.EffectScaling.resolve``:
    ``base + floor(max(d − threshold, 0) / per_step) × per_value``,
    clamped to ``cap``).

    Dropping the floor gives the convex ``base + per_value ·
    (max(d − threshold, 0) + 1e-9) / per_step`` (the 1e-9 mirrors
    ``resolve``'s float guard), which its chord over the interval bounds;
    the cap alone is a second valid bound — the lower at ``d0`` wins.
    """
    if per_step <= 0 or per_value < 0:
        raise LinearBoundError("unsupported kit scaling shape")

    def relaxed(d: float) -> float:
        return base + per_value * (max(d - threshold, 0.0) + 1e-9) / per_step

    if d_hi - d_lo > 1e-12:
        slope = (relaxed(d_hi) - relaxed(d_lo)) / (d_hi - d_lo)
        k0, k1 = relaxed(d_lo) - slope * d_lo, slope
    else:
        k0, k1 = relaxed(d_hi), 0.0
    if cap is not None and cap < k0 + k1 * d0:
        return cap, 0.0
    return k0, k1


def log_affine_tangent(f: Affine, x0) -> Affine:
    """Tangent of ``log f`` at ``x0`` as an affine function (``f`` affine,
    positive at ``x0``): ``log f(x) <= log f0 + (f(x) − f0) / f0``
    wherever ``f > 0``, since ``log`` is concave."""
    f0 = affine_at(f, x0)
    if not f0 > 0:
        raise LinearBoundError("non-positive zone at the tangent point")
    return affine_combine((1.0 / f0, f), const=math.log(f0) - 1.0)


#: Grid points along CRIT Rate when bounding the average-crit zone's
#: worst gap (:func:`average_crit_log_upper`). The Lipschitz margin
#: shrinks as ``1 / n``; 512 keeps it near 0.1-0.3% of the zone.
CRIT_GRID = 512


def _average_crit_gap(a: float, b: float, r_lo: float, r_hi: float,
                      e_lo: float, e_hi: float, cap: float, r0: float,
                      e0: float, n: int) -> float:
    """A RIGOROUS upper bound of
    ``max over the box of log(1 + m(r)·e) − [φ0 + a(r − r0) + b(e − e0)]``
    (``m`` the capped rate, ``φ0 = log(1 + m(r0)·e0)``).

    For a fixed ``r`` the gap is concave in ``e``, so its maximum over
    ``[e_lo, e_hi]`` sits at the stationary point ``1/b − 1/m`` clamped
    to the interval — exact. Over ``r`` the gap is Lipschitz with
    constant ``G = max(|a|, |e_hi − a|)`` (its ``r``-slope lies in
    ``[−a, e_hi − a]``), so a grid of ``n + 1`` points plus ``G·h/2``
    bounds the true maximum.
    """
    phi0 = math.log(1.0 + min(max(r0, 0.0), cap) * e0)
    span = r_hi - r_lo
    steps = n if span > 1e-12 else 0
    worst = -math.inf
    for j in range(steps + 1):
        r = r_lo + span * j / steps if steps else r_lo
        m = min(max(r, 0.0), cap)
        if m > 0.0 and b > 0.0:
            e = 1.0 / b - 1.0 / m
            e = e_lo if e < e_lo else e_hi if e > e_hi else e
        elif b > 0.0:
            e = e_lo
        else:
            e = e_hi
        gap = (math.log(1.0 + m * e) - phi0 - a * (r - r0) - b * (e - e0))
        if gap > worst:
            worst = gap
    lipschitz = max(abs(a), abs(e_hi - a))
    worst += lipschitz * (span / steps / 2.0 if steps else span)
    return worst


def golden_min(fn, low: float, high: float, iterations: int = 24) -> float:
    """Minimizer of a convex 1-D ``fn`` on ``[low, high]`` (golden
    section). Used only to pick tight slopes; validity never depends on
    it."""
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = high - ratio * (high - low)
    x2 = low + ratio * (high - low)
    f1, f2 = fn(x1), fn(x2)
    for _ in range(iterations):
        if f1 <= f2:
            high, x2, f2 = x2, x1, f1
            x1 = high - ratio * (high - low)
            f1 = fn(x1)
        else:
            low, x1, f1 = x1, x2, f2
            x2 = low + ratio * (high - low)
            f2 = fn(x2)
    return x1 if f1 <= f2 else x2


def average_crit_log_upper(rate: Affine, cap: float, dmg: Affine,
                           x0, lo, hi) -> Affine:
    """Affine upper bound of ``log(1 + min(max(rate, 0), cap) · dmg)`` on
    the box — the average-crit zone in log space.

    ``log(1 + r·e)`` is not concave where ``r·e < 1``, so its tangent
    alone can undercut it in the low corner of the box. Instead: pick
    the plane slopes ``(a, b)`` that minimize the worst gap (convex in
    ``(a, b)``; coordinate golden section on a coarse grid), then lift
    the plane by the RIGOROUS gap bound of :func:`_average_crit_gap` at
    full resolution. Measured ~2% slack at the root box of a 653-disc
    inventory, vs ~17% for a McCormick plane.
    """
    r_lo, r_hi = affine_range(rate, lo, hi)
    e_lo, e_hi = affine_range(dmg, lo, hi)
    if r_lo < 0 or e_lo < 0:
        raise LinearBoundError("negative crit factor inside the search box")
    r0 = min(max(affine_at(rate, x0), r_lo), r_hi)
    e0 = min(max(affine_at(dmg, x0), e_lo), e_hi)
    m0 = min(r0, cap)
    denom = 1.0 + m0 * e0
    a = (e0 / denom) if r0 < cap else 0.0
    b = m0 / denom

    def coarse(a_try: float, b_try: float) -> float:
        return _average_crit_gap(a_try, b_try, r_lo, r_hi, e_lo, e_hi, cap,
                                 r0, e0, 48)

    a_cap = max(2.0 * a, e_hi)
    b_cap = max(2.0 * b, cap)
    for _ in range(3):
        a = golden_min(lambda t: coarse(t, b), 0.0, a_cap)
        b = golden_min(lambda t: coarse(a, t), 0.0, b_cap)
    gap = max(_average_crit_gap(a, b, r_lo, r_hi, e_lo, e_hi, cap, r0, e0,
                                CRIT_GRID), 0.0)
    phi0 = math.log(1.0 + m0 * e0)
    return affine_combine((a, rate), (b, dmg),
                          const=phi0 + gap - a * r0 - b * e0)


def crit_zone_log_upper(
    variant: str, rate: Affine, cap: float, dmg: Affine,
    deferred, x0, lo, hi,
) -> Affine | None:
    """Affine upper bound of ``log(crit multiplier)`` on the box, or
    ``None`` when the zone is the constant 1 (``variant ==
    "non_crit"``).

    ``variant`` ``"crit"`` is ``1 + e``, ``"average"`` is
    ``1 + min(max(rate, 0), cap) · e``, where ``e`` is the CRIT DMG
    total after effects that take a share of it (Anby S's Silver Star):
    ``e = dmg + Σ stacks · resolve(dmg)`` — replaced by its affine upper
    bound (the zone only grows with ``e``). ``deferred`` lists those
    effects as ``(stacks, base, threshold, per_step, per_value, cap)``.
    """
    if variant == "non_crit":
        return None
    d_lo, d_hi = affine_range(dmg, lo, hi)
    d0 = affine_at(dmg, x0)
    slope_total = 1.0
    const_total = 0.0
    for stacks, base, threshold, per_step, per_value, s_cap in deferred:
        if stacks < 0:
            raise LinearBoundError("negative stacks on a scaled effect")
        k0, k1 = step_scaling_upper(base, threshold, per_step, per_value,
                                    s_cap, d_lo, d_hi, d0)
        const_total += stacks * k0
        slope_total += stacks * k1
    dmg_upper = affine_combine((slope_total, dmg), const=const_total)
    if variant == "crit":
        return log_affine_tangent(
            affine_combine((1.0, dmg_upper), const=1.0), x0)
    return average_crit_log_upper(rate, cap, dmg_upper, x0, lo, hi)


class LogLinear:
    """Accumulates ``c0 + Σ w[i] · x[i]``, an upper bound on a log.

    Every ``add_*`` method adds a valid upper bound of one zone's log
    (over the box it is given), so the total bounds the log of the
    product of the zones.
    """

    def __init__(self, size: int) -> None:
        self.c0 = 0.0
        self.w = [0.0] * size

    def add_const(self, value: float) -> None:
        self.c0 += value

    def add_linear(self, f: Affine, scale: float = 1.0) -> None:
        const, terms = f
        self.c0 += scale * const
        for i, coef in terms:
            self.w[i] += scale * coef

    def add_log_affine(self, f: Affine, x0) -> None:
        """Tangent of ``log f`` at ``x0`` (``f`` affine and positive at
        ``x0``): ``log f(x) <= log f0 + (f(x) − f0) / f0`` wherever
        ``f > 0``, since ``log`` is concave."""
        self.add_linear(log_affine_tangent(f, x0))

    def add_log_product(self, u: Affine, v: Affine, k: float,
                        x0, lo, hi) -> None:
        """Upper bound of ``log(u·v + k)`` for affine ``u, v > 0`` on the
        box and ``k >= 0``.

        ``k == 0``: ``log u + log v``, two concave tangents.
        ``u_lo · v_lo >= k``: ``log(u·v + k)`` is concave in ``(u, v)``
        on the box (its Hessian is negative semidefinite exactly when
        ``u·v >= k``), so its tangent at ``x0`` bounds it.
        Otherwise: a McCormick plane bounds ``u·v``, then the tangent of
        ``log(plane + k)``.
        """
        u_lo, _ = affine_range(u, lo, hi)
        v_lo, _ = affine_range(v, lo, hi)
        if u_lo <= 0 or v_lo <= 0 or k < 0:
            raise LinearBoundError("non-positive ATK factor in the box")
        if k == 0:
            self.add_log_affine(u, x0)
            self.add_log_affine(v, x0)
            return
        if u_lo * v_lo >= k:
            u0, v0 = affine_at(u, x0), affine_at(v, x0)
            z0 = u0 * v0 + k
            # log z <= log z0 + (v0/z0)(u − u0) + (u0/z0)(v − v0)
            self.c0 += math.log(z0) - 2.0 * u0 * v0 / z0
            self.add_linear(u, v0 / z0)
            self.add_linear(v, u0 / z0)
            return
        plane = mccormick_upper(u, v, x0, lo, hi)
        self.add_log_affine(affine_combine((1.0, plane), const=k), x0)

    def add_log_def(self, s: Affine, armor: float, level_coef: float,
                    lo, hi) -> None:
        """Upper bound of ``log(L / (L + max(A − s, 0)))`` — the DEF zone,
        with ``A`` the enemy DEF after reduction and ``s = A·PEN Ratio +
        PEN`` (affine, increasing).

        As a function of ``s`` it is convex up to ``s = A`` and flat (0)
        after it, so the chord from ``s_lo`` to ``min(s_hi, A)`` —
        extended past ``A`` — lies above it on the whole box.
        """
        if level_coef <= 0:
            raise LinearBoundError("non-positive level coefficient")

        def psi(value: float) -> float:
            return (math.log(level_coef)
                    - math.log(level_coef + max(armor - value, 0.0)))

        s_lo, s_hi = affine_range(s, lo, hi)
        s_b = min(s_hi, armor)
        if s_b - s_lo <= 1e-12:
            # psi is non-decreasing and constant past A: its value at the
            # box's top (clipped to A) bounds the whole box.
            self.add_const(psi(max(s_b, s_lo)))
            return
        slope = (psi(s_b) - psi(s_lo)) / (s_b - s_lo)
        self.add_const(psi(s_lo) - slope * s_lo)
        self.add_linear(s, slope)

    def value_at(self, x) -> float:
        """``c0 + w · x``."""
        return self.c0 + sum(wi * xi for wi, xi in zip(self.w, x) if wi)
