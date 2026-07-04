"""Pure damage-formula functions for direct hits.

Every function here takes plain numbers (or iterables of numbers) and returns
a plain number. No I/O, no game data, no state — everything above this module
composes these pieces (see DOCS/zzz_dmg_calc_plan.md §2 for the math and §3
for the layering).

Full direct-hit formula::

    DMG = BaseDMG × CritMult × DmgBonusMult × DefMult × ResMult
          × DmgTakenMult × StunMult

Conventions:

- All percentage-like stats are **fractions**: 30% ATK is ``0.30``, CRIT DMG
  96% is ``0.96``.
- Multi-source brackets (``dmg_bonus_mult``, ``dmg_taken_mult``, the daze
  bonuses of ``stun_mult``) take *iterables of contributions* and sum them.
  This keeps the brackets composable so future mechanics (Windswept,
  Contamination, …) are new data entries, not new code paths.
"""

from __future__ import annotations

from typing import Iterable

# ---------------------------------------------------------------------------
# ATK aggregation
# ---------------------------------------------------------------------------


def atk_final(
    agent_base_atk: float,
    engine_base_atk: float,
    atk_pct_total: float,
    atk_flat_total: float,
) -> float:
    """Final ATK after combining the three ATK buckets.

    ``ATK_final = (agent_base + engine_base) × (1 + ATK%_total) + ATK_flat``

    ATK% only ever scales the combined *base* ATK (agent + W-Engine). Flat ATK
    from disc mains/substats is added afterwards and is never scaled — keep
    the buckets separate upstream.

    Args:
        agent_base_atk: Agent's own base ATK at Lv. 60.
        engine_base_atk: W-Engine base ATK at Lv. 60.
        atk_pct_total: Sum of all ATK% sources, as a fraction (0.30 for 30%).
        atk_flat_total: Sum of all flat ATK sources.
    """
    return (agent_base_atk + engine_base_atk) * (1.0 + atk_pct_total) + atk_flat_total


# ---------------------------------------------------------------------------
# Individual multiplier zones
# ---------------------------------------------------------------------------


def base_dmg(skill_multiplier: float, atk: float) -> float:
    """Base damage of a hit: ``SkillMultiplier × ATK_final``.

    Args:
        skill_multiplier: Skill's % of ATK as a fraction (2.5 for a 250% hit).
        atk: Final ATK from :func:`atk_final`.
    """
    return skill_multiplier * atk


def crit_mult_non_crit() -> float:
    """CRIT multiplier for a hit that does not crit: always ``1``."""
    return 1.0


def crit_mult_crit(crit_dmg: float) -> float:
    """CRIT multiplier for a guaranteed crit: ``1 + CRIT_DMG``.

    Args:
        crit_dmg: Total CRIT DMG as a fraction (0.96 for 96%).
    """
    return 1.0 + crit_dmg


def crit_mult_average(crit_rate: float, crit_dmg: float, crit_rate_cap: float = 1.0) -> float:
    """Expected-value CRIT multiplier: ``1 + min(rate, cap) × CRIT_DMG``.

    CRIT Rate is clamped to ``[0, crit_rate_cap]`` — over-capped CRIT Rate is
    wasted, and a negative rate cannot heal the hit below non-crit damage.

    Args:
        crit_rate: Total CRIT Rate as a fraction (0.62 for 62%).
        crit_dmg: Total CRIT DMG as a fraction.
        crit_rate_cap: Maximum effective CRIT Rate (from constants, 1.0).
    """
    effective_rate = min(max(crit_rate, 0.0), crit_rate_cap)
    return 1.0 + effective_rate * crit_dmg


def dmg_bonus_mult(bonuses: Iterable[float]) -> float:
    """DMG% bracket: ``1 + Σ(bonuses)``.

    All DMG% sources — elemental DMG%, skill-type DMG%, external buffs — are
    additive with each other inside this single bracket.

    Args:
        bonuses: Each contribution as a fraction (0.30 for a 30% bonus).
    """
    return 1.0 + sum(bonuses)


def effective_def(enemy_def: float, pen_ratio: float, pen_flat: float) -> float:
    """Enemy DEF after penetration, floored at 0.

    ``EffDEF = max(EnemyDEF × (1 − PEN_Ratio) − PEN_flat, 0)``

    Args:
        enemy_def: Enemy's DEF at its level.
        pen_ratio: Attacker's PEN Ratio as a fraction (0.24 for 24%).
        pen_flat: Attacker's flat PEN.
    """
    return max(enemy_def * (1.0 - pen_ratio) - pen_flat, 0.0)


def def_mult(level_coefficient: float, eff_def: float) -> float:
    """DEF multiplier zone: ``LevelCoef / (LevelCoef + EffDEF)``.

    With ``eff_def`` already floored at 0 (see :func:`effective_def`) the
    result is always in ``(0, 1]``.

    Args:
        level_coefficient: Attacker level factor (794 at Lv. 60).
        eff_def: Effective enemy DEF from :func:`effective_def`.
    """
    return level_coefficient / (level_coefficient + eff_def)


def res_mult(enemy_res: float, res_ignore: float = 0.0) -> float:
    """RES multiplier: ``1 − (EnemyRES − RES_ignore)``.

    A weakness is a negative RES (−0.20 → ×1.20); a resistance is positive
    (+0.20 → ×0.80). RES shred/ignore effects subtract from enemy RES.

    Args:
        enemy_res: Enemy's RES to the attack attribute as a fraction.
        res_ignore: Total RES shred/ignore applied to the enemy, as a fraction.
    """
    return 1.0 - (enemy_res - res_ignore)


def dmg_taken_mult(debuffs: Iterable[float] = ()) -> float:
    """"DMG taken" bracket: ``1 + Σ(debuffs)``.

    Separate multiplicative zone from :func:`dmg_bonus_mult`. Kept as a
    composable sum so future mechanics that add "DMG taken" entries
    (Windswept, Contamination, …) are data changes, not code changes.

    Args:
        debuffs: Each "increased DMG taken" contribution as a fraction.
    """
    return 1.0 + sum(debuffs)


def stun_mult(
    stunned: bool,
    stun_dmg_multiplier: float,
    daze_bonuses: Iterable[float] = (),
) -> float:
    """Stun multiplier zone.

    Not stunned → ``1``. Stunned → the enemy's Stun DMG Multiplier (typically
    1.5) plus any additive daze-vulnerability buffs.

    Args:
        stunned: Whether the enemy is currently stunned.
        stun_dmg_multiplier: Enemy's Stun DMG Multiplier from the boss DB.
        daze_bonuses: Additive daze-vulnerability contributions (fractions).
    """
    if not stunned:
        return 1.0
    return stun_dmg_multiplier + sum(daze_bonuses)


# ---------------------------------------------------------------------------
# Attribute Anomaly zones (Phase 5 — see DOCS/anomaly_plan.md)
# ---------------------------------------------------------------------------


def ap_mult(anomaly_proficiency: float) -> float:
    """Anomaly Proficiency multiplier: ``AP / 100``.

    ⚠️ Provisional form — one community source suggests ``1 + AP/100``;
    the first in-game Assault calibration discriminates (~2× apart). If it
    turns out to be the other form, only this function changes.

    Args:
        anomaly_proficiency: Total AP (e.g. 118 -> 1.18).
    """
    return anomaly_proficiency / 100.0


def anomaly_base_dmg(anomaly_mult: float, atk: float) -> float:
    """Base anomaly damage of one hit/tick/proc: ``mult × ATK``.

    Args:
        anomaly_mult: Per-proc multiplier as a fraction (7.13 for Assault).
        atk: Final ATK from :func:`atk_final` (incl. combat buffs).
    """
    return anomaly_mult * atk


def burst_conversion_mult(
    base: float,
    time_mult: float,
    window: float,
    elapsed_seconds: float,
    extra_mult: float = 0.0,
) -> float:
    """Disorder/Vortex burst multiplier (closed form, one function for both).

    ``mult_total = base + extra_mult + time_mult × max(0, window − elapsed)``

    Disorder uses the replaced element's rule (base 4.5, window 10);
    Vortex uses the infused element's rule (per-element mult, window 30).
    ``extra_mult`` carries additive "Disorder DMG Multiplier" buffs
    (Velina's consumed Windbite +1.50, Yuzuha M6 +1.05/stack).

    Form from the zenless-optimizer datamine (2026-07-03) — ⚠️ calibrate
    in-game before trusting (anomaly_plan.md §5e).

    Args:
        base: Element's base burst multiplier (fraction of ATK).
        time_mult: Multiplier per remaining second inside the window.
        window: Seconds after the converted anomaly's application during
            which the time term still contributes.
        elapsed_seconds: Time since the converted anomaly was applied
            (clamped >= 0).
        extra_mult: Additive burst-multiplier buffs (fraction).
    """
    elapsed = max(elapsed_seconds, 0.0)
    return base + extra_mult + time_mult * max(0.0, window - elapsed)


def anomaly_buff_mult(bonuses: Iterable[float]) -> float:
    """Anomaly/Disorder/Vortex "Buff Multiplier" bracket: ``1 + Σ(bonuses)``.

    A separate multiplicative zone from :func:`dmg_bonus_mult` for
    "Attribute Anomaly DMG +X%" / "Disorder DMG +X%" / "Windswept and
    Vortex DMG +X%" effects (Yuzuha's Additional Ability, Velina's kit,
    Joyau Doré) — the zenless-optimizer models these in their own bracket
    (``buff_mult_``), distinct from the ordinary DMG% bonuses.
    ⚠️ Bracket placement PROVISIONAL until popup-checked in-game.

    Args:
        bonuses: Each contribution as a fraction (0.10 for a 10% bonus).
    """
    return 1.0 + sum(bonuses)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def total_dmg(
    base: float,
    crit: float,
    dmg_bonus: float,
    defense: float,
    res: float,
    dmg_taken: float,
    stun: float,
) -> float:
    """Compose all multiplier zones into the final damage number.

    ``DMG = BaseDMG × CritMult × DmgBonusMult × DefMult × ResMult ×
    DmgTakenMult × StunMult``

    Each argument is the output of the corresponding zone function above.
    """
    return base * crit * dmg_bonus * defense * res * dmg_taken * stun
