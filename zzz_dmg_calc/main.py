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
    from .api import (
        BuildupSegment, CalcConfig, SupportConfig, calculate, calculate_anomaly,
    )
    from .agent import load_agents
    from .anomalies import load_anomalies
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
    from zzz_dmg_calc.api import (
        BuildupSegment, CalcConfig, SupportConfig, calculate, calculate_anomaly,
    )
    from zzz_dmg_calc.agent import load_agents
    from zzz_dmg_calc.anomalies import load_anomalies
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

    # Refinement rank is a runtime input; the data file only sets a default.
    engine = engines[engine_key]
    engine_rank = engine.refinement_rank
    if engine.max_rank > 1:
        engine_rank = int(_ask_float(
            f"Refinement rank (1-{engine.max_rank})", engine.refinement_rank
        ))
    for passive in engine.passive_dmg:
        element = passive.element or "all"
        print(f"  Auto-applied at R{engine_rank}: "
              f"+{passive.value(engine_rank):.0%} {element} DMG")

    # Rank-scaled engine conditional buff — prompted later: single stacks
    # value for direct hits, per-segment for anomalies.
    buff = engine.conditional_buff
    if buff is not None:
        print(f"\n{buff.name} (R{engine_rank}: "
              f"+{buff.per_stack(engine_rank):.0%} "
              f"{buff.element or 'DMG'} per stack)")
        if buff.note:
            print(f"  Note: {buff.note}")

    # On-field engine's OWN squad buffs — squad-wide, so the wearer gets
    # them too (e.g. Velina + Joyau Doré's +60 AP at 2 stacks).
    engine_squad_buffs: dict[str, int] = {}
    for squad_buff in engine.squad_buffs:
        per = (f"+{squad_buff.value(engine_rank):g}"
               if squad_buff.kind in ("flat_atk", "anomaly_proficiency")
               else f"+{squad_buff.value(engine_rank):.0%}")
        # Ramp-up (auto) squad buffs default to max — the range covers uptime.
        default = squad_buff.max_stacks if squad_buff.auto else 0
        tag = " [auto]" if squad_buff.auto else ""
        print(f"\n{squad_buff.name}{tag} — squad {squad_buff.kind} {per} "
              f"(applies to {agents[agent_key].name} too): {squad_buff.note}")
        if squad_buff.max_stacks == 1:
            yn = "Y/n" if squad_buff.auto else "y/N"
            answer = input(f"Active? [{yn}]: ").strip().lower()
            active = (answer == "y") or (squad_buff.auto and answer == "")
            if active:
                engine_squad_buffs[squad_buff.name] = 1
        else:
            stacks = int(_ask_float(
                f"Stacks (0-{squad_buff.max_stacks})", default))
            if stacks:
                engine_squad_buffs[squad_buff.name] = stacks

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
            # Ramp-up (auto) 4pc default to max — the range covers uptime.
            tag = " [auto]" if bonus.auto else ""
            print(f"\n{disc_sets[key].name} 4-piece{tag}: {bonus.note}")
            if bonus.max_stacks == 1:
                # On/off effect (e.g. Hormone Punk, Puffer Electro):
                # internally stacks 1/0, asked as a yes/no.
                yn = "Y/n" if bonus.auto else "y/N"
                answer = input(
                    f"Is +{bonus.per_stack:.0%} {bonus.stat} active? [{yn}]: "
                ).strip().lower()
                stacks = 1 if (answer == "y" or (bonus.auto and answer == "")) else 0
            else:
                stacks = int(_ask_float(
                    f"Active stacks of +{bonus.per_stack:.0%} {bonus.stat} "
                    f"(0-{bonus.max_stacks})", bonus.max_stacks if bonus.auto else 0
                ))
            if stacks:
                set_stacks[key] = stacks
    if piece_counts:
        applied = set_bonus_stats(discs, disc_sets, set_stacks)
        summary = ", ".join(f"{s} +{v:.1%}" if v < 1 else f"{s} +{v:g}"
                            for s, v in applied.items())
        print(f"Set bonuses applied: {summary or 'none (need 2+ pieces)'}")

    # 5. Calculation mode
    print("\nCalculation mode:")
    mode = _choose("Mode", ["Direct hit", "Anomaly proc", "Disorder", "Vortex"])
    mode_key = {"Direct hit": "direct", "Anomaly proc": "anomaly",
                "Disorder": "disorder", "Vortex": "vortex"}[mode]

    anomaly_data = load_anomalies()
    agent_attr = agents[agent_key].attribute

    skill = 1.0
    skill_tag = None
    disorder_replaced = None
    disorder_elapsed = 0.0
    vortex_infused = None
    anomaly_mult_override = None
    disorder_mult_add = 0.0
    if mode == "Direct hit":
        print("\nNote: enter the move's TOTAL motion value. Results are the")
        print("whole move; in-game popups are per hit and will show smaller")
        print("numbers that add up to the result.")
        skill = _ask_percent("Skill multiplier", 1.0)

        # Damage type: gates skill-type-conditional bonuses (e.g. Puffer
        # Electro 4pc's Ultimate DMG +20% applies only to Ultimate hits).
        print("\nDamage type of the move:")
        tag_by_label = {label: key for key, label in consts.skill_tags.items()}
        picked = _choose("Type", ["Untyped / not relevant", *tag_by_label])
        skill_tag = tag_by_label.get(picked)
    elif mode == "Anomaly proc":
        anomaly = anomaly_data.anomalies[agent_attr]
        print(f"\nAnomaly: {anomaly.name} [{agent_attr}]"
              + ("" if anomaly.supported else " — NOT YET SUPPORTED"))
        if anomaly.debuff_note:
            print(f"  Note: {anomaly.debuff_note}")
        override_pct = _ask_percent(
            "Anomaly multiplier OVERRIDE, replaces the base (Enter = normal "
            f"{anomaly.name} proc; Velina Ablooms: 145 Condensed / 255 "
            "Sweeping / 680 Ultimate - each a SEPARATE popup)",
            0.0,
        )
        anomaly_mult_override = override_pct if override_pct > 0 else None
    elif mode == "Disorder":
        options = [
            f"{a.name} [{e}]" for e, a in anomaly_data.anomalies.items()
            if a.supported and e != agent_attr
        ]
        picked = _choose("Anomaly being REPLACED", options)
        disorder_replaced = picked.split("[")[1].rstrip("]")
        disorder_elapsed = _ask_float(
            "Seconds elapsed since the replaced anomaly was applied", 5.0
        )
        disorder_mult_add = _ask_percent(
            "Additive Disorder base-mult increase not modeled elsewhere"
        )
    else:   # Vortex (second anomaly over an active Windswept)
        options = [
            f"{anomaly_data.anomalies[e].name} [{e}]"
            for e in sorted(anomaly_data.vortex)
        ]
        picked = _choose("Anomaly INFUSED into the Windswept", options)
        vortex_infused = picked.split("[")[1].rstrip("]")
        disorder_elapsed = _ask_float(
            "Seconds elapsed since the infused anomaly was applied", 0.0
        )
        disorder_mult_add = _ask_percent(
            "Additive Vortex base-mult increase, manual (Velina's Windbite "
            "is her 'Windbite: enhanced Vortex' team buff - don't re-enter here)"
        )

    # 6. Buffs. Attacker-side buffs snapshot during BUILDUP for anomalies
    # (mechanic discovery #2), so anomaly modes describe the buildup in
    # segments; direct hits take instantaneous values.
    dmg_bonus = 0.0
    crit_rate_buff = crit_dmg_buff = 0.0
    engine_buff_stacks = 0.0
    core_passive_active = False
    additional_ability_stacks = 0
    mindscape_stacks: dict[int, int] = {}
    scaling_inputs: dict[str, float] = {}
    supports: list[SupportConfig] = []
    buildup_segments: list[BuildupSegment] = []
    anomaly_buff_ext = 0.0

    # Agent kit conditionals (modeled core passive / mindscapes) apply in
    # EVERY mode — effects carry their own mode/element gates (agents.json).
    # Anomaly modes hold kit/team buff state constant over the buildup
    # (adopted concession); buildup segments cover what varies.
    agent_obj = agents[agent_key]

    # Owner-stat inputs for the agent's scaled kit effects (e.g. Zhao as
    # attacker: her core-passive CRIT scales with her Max HP; Velina's
    # core scales with her panel Energy Regen).
    kit_buffs = [b for b in (agent_obj.core_passive,
                             agent_obj.additional_ability) if b]
    kit_buffs += [m.buff for m in agent_obj.mindscapes.values() if m.buff]
    for kit_b in kit_buffs:
        for eff in kit_b.effects:
            if eff.scaling is not None and eff.scaling.input not in scaling_inputs:
                scaling_inputs[eff.scaling.input] = _ask_float(
                    f"{agent_obj.name}'s "
                    f"{eff.scaling.input.replace('_', ' ')}", 0
                )

    core = agent_obj.core_passive
    if core is not None:
        effects = ", ".join(f"{e.kind} +{e.value:.0%}" for e in core.effects)
        print(f"\nCore passive {core.name} ({effects}): {core.note}")
        core_passive_active = (
            input("Is it active? [y/N]: ").strip().lower() == "y"
        )
    ability = agent_obj.additional_ability
    if ability is not None:
        print(f"\nAdditional Ability {ability.name}: {ability.note}")
        if ability.max_stacks == 1:
            if input("Active? [y/N]: ").strip().lower() == "y":
                additional_ability_stacks = 1
        else:
            additional_ability_stacks = int(_ask_float(
                f"Active stacks (0-{ability.max_stacks})", 0
            ))

    for level in sorted(agent_obj.mindscapes):
        mindscape = agent_obj.mindscapes[level]
        if mindscape.buff is None:
            continue
        kit_buff = mindscape.buff
        effects = ", ".join(
            f"{e.kind} +{e.value:.0%}/stack" for e in kit_buff.effects
        )
        print(f"\nM{level} {mindscape.name} ({effects}): {mindscape.note}")
        if kit_buff.max_stacks == 1:
            if input("Unlocked and active? [y/N]: ").strip().lower() == "y":
                mindscape_stacks[level] = 1
        else:
            stacks = int(_ask_float(
                f"Active stacks (0-{kit_buff.max_stacks})", 0
            ))
            if stacks:
                mindscape_stacks[level] = stacks

    # Off-field supports: team buffs, signature-engine squad buffs, and
    # squad-facing 4pc sets (all conditionals, asked per support). Applied
    # in every mode; mode-gated effects (Yuzuha's Anomaly/Disorder buffs)
    # only land where they belong.
    squad_sets = {k: s for k, s in disc_sets.items() if s.squad_4pc}
    chosen = {agent_key}
    for slot_no in (1, 2):
        candidates = {f"{a.name} [{a.attribute}]": k
                      for k, a in agents.items() if k not in chosen}
        print(f"\nSupport {slot_no}:")
        picked = _choose("Support", ["None", *candidates])
        if picked == "None":
            break
        support_key = candidates[picked]
        chosen.add(support_key)
        member = agents[support_key]

        buffs: dict[str, int] = {}
        scaling: dict[str, float] = {}
        for team_buff in member.team_buffs:
            for eff in team_buff.effects:
                if eff.scaling is not None and eff.scaling.input not in scaling:
                    scaling[eff.scaling.input] = _ask_float(
                        f"{member.name}'s "
                        f"{eff.scaling.input.replace('_', ' ')}", 0
                    )
            print(f"{team_buff.name}: {team_buff.note}")
            if team_buff.max_stacks == 1:
                if input("Active? [y/N]: ").strip().lower() == "y":
                    buffs[team_buff.name] = 1
            else:
                stacks = int(_ask_float(
                    f"Stacks (0-{team_buff.max_stacks})", 0
                ))
                if stacks:
                    buffs[team_buff.name] = stacks

        engine_buffs: dict[str, int] = {}
        support_rank = None
        support_engine = engines.get(member.default_engine)
        if support_engine is not None and support_engine.squad_buffs:
            support_rank = int(_ask_float(
                f"{support_engine.name} refinement "
                f"(1-{support_engine.max_rank})",
                support_engine.refinement_rank,
            ))
            for squad_buff in support_engine.squad_buffs:
                print(f"{squad_buff.name}: {squad_buff.note}")
                if squad_buff.max_stacks == 1:
                    if input("Active? [y/N]: ").strip().lower() == "y":
                        engine_buffs[squad_buff.name] = 1
                else:
                    stacks = int(_ask_float(
                        f"Stacks (0-{squad_buff.max_stacks})", 0
                    ))
                    if stacks:
                        engine_buffs[squad_buff.name] = stacks

        squad_set = None
        squad_set_stacks = 0
        if squad_sets:
            labels = {s.name: k for k, s in squad_sets.items()}
            picked_set = _choose("Squad-facing 4pc set worn",
                                 ["None", *labels])
            if picked_set != "None":
                squad_set = labels[picked_set]
                bonus = squad_sets[squad_set].squad_4pc
                if bonus.max_stacks == 1:
                    squad_set_stacks = (
                        1 if input("Active? [y/N]: ").strip().lower() == "y"
                        else 0
                    )
                else:
                    squad_set_stacks = int(_ask_float(
                        f"Stacks (0-{bonus.max_stacks})", 0
                    ))

        supports.append(SupportConfig(
            agent_key=support_key, buffs=buffs, scaling_inputs=scaling,
            squad_set=squad_set, squad_set_stacks=squad_set_stacks,
            engine_rank=support_rank, engine_buffs=engine_buffs,
        ))

    if mode == "Direct hit":
        print("\nExternal buffs (press Enter to skip):")
        if buff is not None and buff.bracket == "dmg_bonus":
            tag = " [auto]" if buff.auto else ""
            if buff.max_stacks == 1:
                yn = "Y/n" if buff.auto else "y/N"
                answer = input(f"Is {buff.name}{tag} active? [{yn}]: ").strip().lower()
                active = (answer == "y") or (buff.auto and answer == "")
                engine_buff_stacks = 1.0 if active else 0.0
            else:
                engine_buff_stacks = _ask_float(
                    f"Active {buff.name}{tag} stacks (0-{buff.max_stacks})",
                    buff.max_stacks if buff.auto else 0
                )
        dmg_bonus = _ask_percent("Extra DMG% - EXTERNAL only (season / boss)")
        crit_rate_buff = _ask_percent("Extra CRIT Rate % - EXTERNAL only")
        crit_dmg_buff = _ask_percent("Extra CRIT DMG % - EXTERNAL only")
    else:
        print("\nExternal buffs (press Enter to skip):")
        if buff is not None and buff.bracket == "anomaly_buff":
            # Separate-bracket engine buffs (Joyau Doré) are held constant
            # over the buildup: one stacks value, not per segment. Ramp-up
            # (auto) buffs default to max — the range covers uptime.
            tag = " [auto]" if buff.auto else ""
            engine_buff_stacks = _ask_float(
                f"Active {buff.name}{tag} stacks (0-{buff.max_stacks}, "
                f"constant during buildup)", buff.max_stacks if buff.auto else 0
            )
        anomaly_buff_ext = _ask_percent(
            "Extra Anomaly/Disorder/Vortex DMG buff - EXTERNAL only (season / "
            "boss buffs; kit/engine/set ramp-up buffs are auto-applied)"
        )

        print("\nBuildup segments - the proc snapshots attacker buffs as a")
        print("buildup-weighted average. Describe roughly what % of the")
        print("buildup happened under which buffs; whatever you don't")
        print("assign counts as buff-free buildup.")
        remaining = 100.0
        while remaining > 0.5:
            share = _ask_float(
                f"% of buildup for next segment (Enter = leave the "
                f"remaining {remaining:.0f}% buff-free)", 0
            )
            if share <= 0:
                break
            share = min(share, remaining)
            seg_stacks = 0.0
            if buff is not None and buff.bracket == "dmg_bonus":
                seg_stacks = _ask_float(
                    f"  {buff.name} stacks during this segment "
                    f"(0-{buff.max_stacks})", 0
                )
            seg_dmg = _ask_percent("  Extra DMG% during this segment")
            seg_ap = _ask_float("  Extra flat AP during this segment", 0.0)
            buildup_segments.append(BuildupSegment(
                share=share / 100.0,
                engine_buff_stacks=seg_stacks,
                set_stacks=set_stacks,
                external_dmg_bonuses=[seg_dmg] if seg_dmg else [],
                external_anomaly_proficiency=seg_ap,
            ))
            remaining -= share

    print("\nEnemy-side modifiers (press Enter to skip):")
    res_shred = _ask_percent("Enemy RES shred/ignore (total)")
    dmg_taken = _ask_percent("'DMG taken' debuffs (total)")
    stun_mult = _ask_percent(
        "Stun DMG multiplier shown under the daze bar (Enter = boss default)",
        bosses[boss_name].stun_dmg_multiplier,
    )

    config = CalcConfig(
        agent_key=agent_key,
        engine_key=engine_key,
        engine_rank=engine_rank,
        boss_name=boss_name,
        skill_multiplier=skill,
        skill_tag=skill_tag,
        core_passive_active=core_passive_active,
        additional_ability_stacks=additional_ability_stacks,
        mindscapes=mindscape_stacks,
        scaling_inputs=scaling_inputs,
        supports=supports,
        discs=discs,
        set_stacks=set_stacks,
        engine_squad_buffs=engine_squad_buffs,
        engine_buff_stacks=engine_buff_stacks,
        external_dmg_bonuses=[dmg_bonus] if dmg_bonus else [],
        external_crit_rate=crit_rate_buff,
        external_crit_dmg=crit_dmg_buff,
        external_res_shred=res_shred,
        external_dmg_taken=[dmg_taken] if dmg_taken else [],
        stun_multiplier_override=stun_mult,
        buildup_segments=buildup_segments,
        disorder_replaced=disorder_replaced,
        disorder_elapsed_seconds=disorder_elapsed,
        vortex_infused=vortex_infused,
        anomaly_mult_override=anomaly_mult_override,
        external_anomaly_buff=[anomaly_buff_ext] if anomaly_buff_ext else [],
        external_disorder_mult_add=disorder_mult_add,
    )

    # 7. Output table
    if mode == "Direct hit":
        results = calculate(
            config, consts=consts, disc_data=disc_data, bosses=bosses,
            agents=agents, engines=engines, disc_sets=disc_sets,
        )
        print(f"\n=== Results vs {boss_name} ===")
        print(f"Final ATK: {results.atk_final:,.1f}   "
              f"CRIT: {results.crit_rate:.1%} / {results.crit_dmg:.1%}")
        print(f"Zones: DMG% x{results.dmg_bonus_mult:.3f} | "
              f"DEF x{results.def_mult:.4f} | RES x{results.res_mult:.2f} | "
              f"Taken x{results.dmg_taken_mult:.2f} | "
              f"Stun x{results.stun_mult:.2f}")
        if results.buff_breakdown:
            print("Active buffs:")
            for item in results.buff_breakdown:
                shown = (f"+{item['value']:g}"
                         if item["kind"] in ("flat_atk", "anomaly_proficiency")
                         else f"+{item['value']:.1%}")
                print(f"  {item['source']}: {item['kind']} {shown}")
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
    else:
        r = calculate_anomaly(
            config, consts=consts, disc_data=disc_data, bosses=bosses,
            agents=agents, engines=engines, disc_sets=disc_sets,
            anomaly_data=anomaly_data,
        )
        print(f"\n=== {r.anomaly_name} [{r.element}] vs {boss_name} ===")
        print(f"Final ATK: {r.atk_final:,.1f}   AP: {r.anomaly_proficiency:g} "
              f"(x{r.ap_mult:.3f})   Level mult: x{r.anomaly_level_mult:g}")
        print(f"Burst/proc mult: x{r.anomaly_mult:.3f}")
        print(f"Zones: DMG% x{r.dmg_bonus_mult:.3f} | "
              f"Anomaly buff x{r.anomaly_buff_mult:.3f} | "
              f"DEF x{r.def_mult:.4f} | RES x{r.res_mult:.2f} | "
              f"Taken x{r.dmg_taken_mult:.2f} | Stun x{r.stun_mult:.2f}")
        print("(anomalies cannot crit)")
        if r.buff_breakdown:
            print("Active buffs:")
            for item in r.buff_breakdown:
                shown = (f"+{item['value']:g}"
                         if item["kind"] in ("flat_atk", "anomaly_proficiency")
                         else f"+{item['value']:.1%}")
                print(f"  {item['source']}: {item['kind']} {shown}")
        header = f"{'Scenario':<22}{'Normal':>14}{'Stunned':>14}"
        print("\n" + header)
        print("-" * len(header))
        if r.hits > 1:
            print(f"{'Per tick/proc':<22}{r.per_proc:>14,.1f}"
                  f"{r.per_proc_stunned:>14,.1f}")
            print(f"{f'Full duration (x{r.hits})':<22}{r.full:>14,.1f}"
                  f"{r.full_stunned:>14,.1f}")
        else:
            print(f"{'Burst':<22}{r.per_proc:>14,.1f}"
                  f"{r.per_proc_stunned:>14,.1f}")


if __name__ == "__main__":
    run()
