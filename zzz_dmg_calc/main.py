"""Interactive CLI for the ZZZ DMG Optimizer.

This is the only module that talks to the user (plan §3): it gathers input,
builds a :class:`~.api.CalcConfig`, calls :func:`~.api.calculate`, and prints
the result table. All rules/validation live in the layers below — the CLI
just re-prompts when a layer rejects the input.

Run either way::

    python -m zzz_dmg_calc.main   # from the project root
    python run.py                 # root launcher
    python main.py                # directly from this folder also works

CLI flow (plan §6): boss -> agent -> engine -> disc mains (slots 4/5/6 pick
a type, 1/2/3 are fixed) -> substat rolls -> skill multiplier -> external
buffs -> output table (non-crit / crit / average, normal and stunned).
"""

from __future__ import annotations

try:
    from .api import CalcConfig, calculate
    from .agent import load_agents
    from .constants import load_constants
    from .discs import (
        Disc, DiscError, load_disc_data, load_disc_sets, load_loadouts,
        set_bonus_stats, validate_disc,
    )
    from .enemies import load_bosses
    from .engines import load_engines
except ImportError:
    # Executed directly as a script (``py main.py``): there is no parent
    # package, so relative imports fail. Put the project root on sys.path
    # and import the package absolutely instead.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from zzz_dmg_calc.api import CalcConfig, calculate
    from zzz_dmg_calc.agent import load_agents
    from zzz_dmg_calc.constants import load_constants
    from zzz_dmg_calc.discs import (
        Disc, DiscError, load_disc_data, load_disc_sets, load_loadouts,
        set_bonus_stats, validate_disc,
    )
    from zzz_dmg_calc.enemies import load_bosses
    from zzz_dmg_calc.engines import load_engines


# ---------------------------------------------------------------------------
# Small input helpers (re-prompt until valid)
# ---------------------------------------------------------------------------


def _choose(prompt: str, options: list[str]) -> str:
    """Numbered menu; returns the chosen option string."""
    for i, option in enumerate(options, start=1):
        print(f"  {i}. {option}")
    while True:
        raw = input(f"{prompt} [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("  Invalid choice, try again.")


def _ask_float(prompt: str, default: float = 0.0) -> float:
    """Float input; empty answer returns ``default``."""
    while True:
        raw = input(f"{prompt} [default {default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Not a number, try again.")


def _ask_percent(prompt: str, default: float = 0.0) -> float:
    """Percent input entered as human numbers (30 = 30%) -> fraction."""
    return _ask_float(f"{prompt} (in %, e.g. 30)", default * 100) / 100.0


# ---------------------------------------------------------------------------
# Disc entry
# ---------------------------------------------------------------------------


def _enter_disc(slot: int, disc_data, substats_per_disc: int,
                disc_sets) -> Disc | None:
    """Interactively build one disc; returns None if the slot is skipped."""
    options = sorted(disc_data.main_stats[slot])
    print(f"\n--- Disc slot {slot} ---")

    if input("Equip this slot? [Y/n]: ").strip().lower() == "n":
        return None

    if len(options) == 1:
        main = options[0]
        print(f"Main stat is fixed: {main} "
              f"{disc_data.main_stats[slot][main]}")
    else:
        main = _choose("Main stat type", options)

    set_labels = {s.name: key for key, s in disc_sets.items()}
    picked_set = _choose("Set", [*set_labels, "No set / not modeled"])
    disc_set = set_labels.get(picked_set)

    while True:
        print(f"Enter the {substats_per_disc} substats. For each one, type "
              f"the upgrade count the game shows as '+N' (Enter or 0 = "
              f"substat at its base value).")
        substats: dict[str, int] = {}
        for i in range(1, substats_per_disc + 1):
            stat = _choose(
                f"Substat {i}",
                sorted(s for s in disc_data.substat_rolls
                       if s != main and s not in substats),
            )
            upgrades = int(_ask_float(f"  Upgrades on {stat} (+N in game)", 0))
            # The first roll is implicit: '+N' in game = N + 1 total rolls.
            substats[stat] = upgrades + 1

        disc = Disc(slot=slot, main_stat=main, substats=substats,
                    disc_set=disc_set)
        try:
            validate_disc(disc, disc_data)
            return disc
        except DiscError as exc:
            print(f"  Invalid disc: {exc}. Re-enter this disc's substats.")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def run() -> None:
    """Full interactive session (plan §6)."""
    print("=== ZZZ DMG Optimizer — direct-hit damage (v1) ===")

    bosses = load_bosses()
    agents = load_agents()
    engines = load_engines()
    disc_data = load_disc_data()
    consts = load_constants()

    # 1. Boss
    print("\nTarget boss:")
    boss_name = _choose("Boss", list(bosses))

    # 2. Agent (v1: DUMMY only)
    print("\nAgent:")
    agent_key_by_label = {
        f"{a.name} [{a.attribute}]": key for key, a in agents.items()
    }
    agent_key = agent_key_by_label[_choose("Agent", list(agent_key_by_label))]

    # 2b. W-Engine (Enter = agent's default)
    default_engine = agents[agent_key].default_engine
    print(f"\nW-Engine (default: {engines[default_engine].name}):")
    engine_key_by_label = {
        f"{e.name} — {e.base_atk:g} ATK": key for key, e in engines.items()
    }
    labels = list(engine_key_by_label)
    if len(labels) == 1:
        engine_key = default_engine
        print(f"  Only one engine available: {engines[engine_key].name}")
    else:
        engine_key = engine_key_by_label[_choose("Engine", labels)]
    if engines[engine_key].passive_note:
        print(f"  Note: {engines[engine_key].passive_note}")

    # 3-4. Discs: saved loadout or manual entry
    loadouts = load_loadouts(disc_data)
    disc_sets = load_disc_sets()
    discs: list[Disc] = []
    manual = True
    if loadouts:
        labels = [
            f"Loadout '{name}' — {l.description}" for name, l in loadouts.items()
        ] + ["Enter discs manually", "No discs"]
        picked = _choose("\nDiscs", labels)
        if picked.startswith("Loadout "):
            loadout = list(loadouts.values())[labels.index(picked)]
            discs = list(loadout.discs)
            manual = False
            for disc in discs:
                set_tag = f" [{disc.disc_set}]" if disc.disc_set else ""
                print(f"  Slot {disc.slot}: {disc.main_stat} main{set_tag} | "
                      + ", ".join(f"{s} x{r}" for s, r in disc.substats.items())
                      + " (rolls)")
        elif picked == "No discs":
            manual = False
    if manual:
        for slot in sorted(disc_data.main_stats):
            disc = _enter_disc(slot, disc_data, disc_data.substats_per_disc,
                               disc_sets)
            if disc is not None:
                discs.append(disc)

    # 4b. Set bonuses: 2pc auto; modeled 4pc effects ask for active stacks.
    piece_counts: dict[str, int] = {}
    for disc in discs:
        if disc.disc_set:
            piece_counts[disc.disc_set] = piece_counts.get(disc.disc_set, 0) + 1
    set_stacks: dict[str, int] = {}
    for key, count in piece_counts.items():
        bonus = disc_sets[key].bonus_4pc
        if count >= 4 and bonus is not None:
            print(f"\n{disc_sets[key].name} 4-piece: {bonus.note}")
            stacks = int(_ask_float(
                f"Active stacks of +{bonus.per_stack:.0%} {bonus.stat} "
                f"(0-{bonus.max_stacks})", 0
            ))
            if stacks:
                set_stacks[key] = stacks
    if piece_counts:
        applied = set_bonus_stats(discs, disc_sets, set_stacks)
        summary = ", ".join(f"{s} +{v:.1%}" if v < 1 else f"{s} +{v:g}"
                            for s, v in applied.items())
        print(f"Set bonuses applied: {summary or 'none (need 2+ pieces)'}")

    # 5. Skill multiplier (total motion value — see README addendum)
    print("\nNote: enter the move's TOTAL motion value. Results are the")
    print("whole move; in-game popups are per hit and will show smaller")
    print("numbers that add up to the result.")
    skill = _ask_percent("Skill multiplier", 1.0)

    # 6. External buffs (all optional, default 0)
    print("\nExternal buffs (press Enter to skip):")
    dmg_bonus = _ask_percent("Extra DMG% bonuses (total)")
    crit_rate_buff = _ask_percent("Extra CRIT Rate from conditional buffs")
    crit_dmg_buff = _ask_percent(
        "Extra CRIT DMG from conditional buffs (e.g. core passive)"
    )
    res_shred = _ask_percent("Enemy RES shred/ignore (total)")
    dmg_taken = _ask_percent("'DMG taken' debuffs (total)")
    stun_mult = _ask_percent(
        "Stun DMG multiplier shown under the daze bar (Enter = boss default)",
        bosses[boss_name].stun_dmg_multiplier,
    )

    config = CalcConfig(
        agent_key=agent_key,
        engine_key=engine_key,
        boss_name=boss_name,
        skill_multiplier=skill,
        discs=discs,
        set_stacks=set_stacks,
        external_dmg_bonuses=[dmg_bonus] if dmg_bonus else [],
        external_crit_rate=crit_rate_buff,
        external_crit_dmg=crit_dmg_buff,
        external_res_shred=res_shred,
        external_dmg_taken=[dmg_taken] if dmg_taken else [],
        stun_multiplier_override=stun_mult,
    )
    results = calculate(
        config, consts=consts, disc_data=disc_data, bosses=bosses,
        agents=agents, engines=engines, disc_sets=disc_sets,
    )

    # 7. Output table
    print(f"\n=== Results vs {boss_name} ===")
    print(f"Final ATK: {results.atk_final:,.1f}   "
          f"CRIT: {results.crit_rate:.1%} / {results.crit_dmg:.1%}")
    print(f"Zones: DMG% x{results.dmg_bonus_mult:.3f} | "
          f"DEF x{results.def_mult:.4f} | RES x{results.res_mult:.2f} | "
          f"Taken x{results.dmg_taken_mult:.2f} | "
          f"Stun x{results.stun_mult:.2f}")
    header = f"{'Scenario':<12}{'Normal':>14}{'Stunned':>14}"
    print("\n" + header)
    print("-" * len(header))
    rows = (
        ("Non-crit", results.non_crit, results.non_crit_stunned),
        ("Crit", results.crit, results.crit_stunned),
        ("Average", results.average, results.average_stunned),
    )
    for label, normal, stunned in rows:
        print(f"{label:<12}{normal:>14,.1f}{stunned:>14,.1f}")


if __name__ == "__main__":
    run()
