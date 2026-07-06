"""Local web UI server for direct-hit calculations (Phase A).

Design and scope: see DOCS/ui_plan.md.

Zero-dependency front end: this stdlib HTTP server serves one
self-contained page (``index.html``) whose form builds a
:class:`~zzz_dmg_calc.api.CalcConfig` and calls the same
:func:`~zzz_dmg_calc.api.calculate` the CLI uses. The server is a thin
adapter — all rules/validation stay in the layers below; rejected input
comes back as HTTP 400 carrying the layer's error message, which the
page shows inline instead of crashing.

Endpoints::

    GET    /            the form page
    GET    /data        databases the form needs (agents, engines, bosses,
                        disc tables, sets, user disc inventory, loadouts)
    POST   /calculate   CalcConfig fields as JSON -> CalcResults as JSON
    POST   /optimize    same payload + an ``optimize`` options object ->
                        best build from the saved disc inventory
                        (DOCS/optimizer_plan.md)
    POST   /discs       save one disc to the inventory (deduped) -> its id
    DELETE /discs/{id}  remove an inventory disc (blocked if referenced)
    POST   /loadouts    save equipped discs as a loadout of inventory
                        references (name collision asks before overwrite)

Run either way::

    python run_ui.py                  # root launcher (opens the browser)
    python -m zzz_dmg_calc.ui.server  # from the project root

The server binds to 127.0.0.1 only — it is a local app, not a website.
"""

from __future__ import annotations

import json
import webbrowser
from dataclasses import asdict, dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Timer
from urllib.parse import unquote

try:
    from ..api import CalcConfig, SupportConfig, calculate, calculate_anomaly
    from ..agent import Agent, load_agents
    from ..anomalies import AnomalyData, load_anomalies
    from ..constants import Constants, load_constants
    from ..discs import (
        Disc, DiscData, DiscSet, Loadout, delete_user_disc, load_disc_data,
        load_disc_sets, load_loadouts, load_user_discs, save_loadout,
        save_user_disc, set_bonus_stats, set_tagged_dmg_bonuses,
    )
    from ..enemies import Boss, load_bosses
    from ..engines import Engine, load_engines
    from ..optimizer import OptimizeOptions, optimize, optimize_anomaly
except ImportError:
    # Executed directly as a script (``py server.py``): no parent package,
    # so relative imports fail. Same fallback as main.py.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from zzz_dmg_calc.api import (
        CalcConfig, SupportConfig, calculate, calculate_anomaly,
    )
    from zzz_dmg_calc.agent import Agent, load_agents
    from zzz_dmg_calc.anomalies import AnomalyData, load_anomalies
    from zzz_dmg_calc.constants import Constants, load_constants
    from zzz_dmg_calc.discs import (
        Disc, DiscData, DiscSet, Loadout, delete_user_disc, load_disc_data,
        load_disc_sets, load_loadouts, load_user_discs, save_loadout,
        save_user_disc, set_bonus_stats, set_tagged_dmg_bonuses,
    )
    from zzz_dmg_calc.enemies import Boss, load_bosses
    from zzz_dmg_calc.engines import Engine, load_engines
    from zzz_dmg_calc.optimizer import (
        OptimizeOptions, optimize, optimize_anomaly,
    )


#: The single page served at ``/`` (self-contained: inline CSS/JS).
INDEX_FILE = Path(__file__).parent / "index.html"

DEFAULT_PORT = 8765


@dataclass(frozen=True)
class AppData:
    """All databases, loaded once at startup and shared by every request.

    ``user_discs`` and ``loadouts`` are user data the UI can write; after a
    write the handler swaps in a fresh AppData (see ``_refresh_user_data``).
    """

    consts: Constants
    disc_data: DiscData
    bosses: dict[str, Boss]
    agents: dict[str, Agent]
    engines: dict[str, Engine]
    disc_sets: dict[str, DiscSet]
    anomalies: AnomalyData
    user_discs: dict[str, Disc]
    loadouts: dict[str, Loadout]


def load_app_data() -> AppData:
    """Load every data file; raises the loader's error if one is invalid."""
    disc_data = load_disc_data()
    user_discs = load_user_discs(disc_data)
    return AppData(
        consts=load_constants(),
        disc_data=disc_data,
        bosses=load_bosses(),
        agents=load_agents(),
        engines=load_engines(),
        disc_sets=load_disc_sets(),
        anomalies=load_anomalies(),
        user_discs=user_discs,
        loadouts=load_loadouts(disc_data, user_discs=user_discs),
    )


def _disc_json(disc: Disc) -> dict:
    """One disc in the JSON form the page uses (substats = total rolls)."""
    return {
        "slot": disc.slot,
        "main_stat": disc.main_stat,
        "substats": disc.substats,
        "set": disc.disc_set,
        "element": disc.element,
    }


def user_discs_payload(data: AppData) -> dict:
    """The inventory as id -> disc JSON."""
    return {disc_id: _disc_json(d) for disc_id, d in data.user_discs.items()}


def loadouts_payload(data: AppData) -> dict:
    """Loadouts with their discs fully resolved (references included)."""
    return {
        name: {
            "description": l.description,
            "discs": [_disc_json(d) for d in l.discs],
        }
        for name, l in data.loadouts.items()
    }


def frontend_payload(data: AppData) -> dict:
    """Serialize the databases into the JSON the form page consumes."""
    def kit_buff(buff) -> dict | None:
        if buff is None:
            return None
        return {
            "name": buff.name,
            "max_stacks": buff.max_stacks,
            "effects": [
                {
                    "kind": e.kind,
                    "value": e.value,
                    "skill_tag": e.skill_tag,
                    "modes": list(e.modes) if e.modes else None,
                    "element": e.element,
                    "scaling": (
                        None if e.scaling is None else {
                            "input": e.scaling.input,
                            "base": e.scaling.base,
                            "threshold": e.scaling.threshold,
                            "per_step": e.scaling.per_step,
                            "per_value": e.scaling.per_value,
                            "cap": e.scaling.cap,
                        }
                    ),
                }
                for e in buff.effects
            ],
            "note": buff.note,
            "condition": buff.condition,
        }

    return {
        "agents": {
            key: {
                "name": a.name,
                "attribute": a.attribute,
                "specialty": a.specialty,
                "faction": a.faction,
                "level": a.level,
                "default_engine": a.default_engine,
                "core_passive": kit_buff(a.core_passive),
                "additional_ability": kit_buff(a.additional_ability),
                "team_buffs": [kit_buff(b) for b in a.team_buffs],
                "mindscapes": {
                    str(level): {
                        "name": m.name,
                        "note": m.note,
                        "buff": kit_buff(m.buff),
                    }
                    for level, m in sorted(a.mindscapes.items())
                },
            }
            for key, a in data.agents.items()
        },
        "engines": {
            key: {
                "name": e.name,
                "base_atk": e.base_atk,
                "advanced_stat": e.advanced_stat,
                "refinement_rank": e.refinement_rank,
                "max_rank": e.max_rank,
                "passive_note": e.passive_note,
                "passive_dmg": [
                    {
                        "element": pd.element,
                        "values_by_rank": list(pd.values_by_rank),
                        "note": pd.note,
                    }
                    for pd in e.passive_dmg
                ],
                "passive_ap_by_rank": list(e.passive_ap_by_rank),
                "conditional_buff": (
                    None if e.conditional_buff is None else {
                        "name": e.conditional_buff.name,
                        "element": e.conditional_buff.element,
                        "bracket": e.conditional_buff.bracket,
                        "modes": (
                            list(e.conditional_buff.modes)
                            if e.conditional_buff.modes else None
                        ),
                        "per_stack_by_rank": list(
                            e.conditional_buff.per_stack_by_rank
                        ),
                        "max_stacks": e.conditional_buff.max_stacks,
                        "auto": e.conditional_buff.auto,
                        "note": e.conditional_buff.note,
                    }
                ),
                "squad_buffs": [
                    {
                        "name": b.name,
                        "kind": b.kind,
                        "values_by_rank": list(b.values_by_rank),
                        "max_stacks": b.max_stacks,
                        "auto": b.auto,
                        "note": b.note,
                    }
                    for b in e.squad_buffs
                ],
            }
            for key, e in data.engines.items()
        },
        "bosses": [
            {
                "name": b.name,
                "base_def": b.base_def,
                "res": b.res,
                "stun_dmg_multiplier": b.stun_dmg_multiplier,
            }
            for b in data.bosses.values()
        ],
        "disc": {
            "main_stats": {
                str(slot): table
                for slot, table in data.disc_data.main_stats.items()
            },
            "substat_rolls": data.disc_data.substat_rolls,
            "substats_per_disc": data.disc_data.substats_per_disc,
            "max_rolls_per_substat": data.disc_data.max_rolls_per_substat,
            "max_total_rolls": data.disc_data.max_total_rolls,
        },
        "sets": {
            key: {
                "name": s.name,
                "bonus_2pc": s.bonus_2pc,
                "bonus_4pc": (
                    None if s.bonus_4pc is None else {
                        "stat": s.bonus_4pc.stat,
                        "per_stack": s.bonus_4pc.per_stack,
                        "max_stacks": s.bonus_4pc.max_stacks,
                        "auto": s.bonus_4pc.auto,
                        "note": s.bonus_4pc.note,
                    }
                ),
                "auto_4pc_dmg": s.auto_4pc_dmg,
                "squad_4pc": (
                    None if s.squad_4pc is None else {
                        "value": s.squad_4pc.value,
                        "kind": s.squad_4pc.kind,
                        "max_stacks": s.squad_4pc.max_stacks,
                        "auto": s.squad_4pc.auto,
                        "note": s.squad_4pc.note,
                    }
                ),
                "notes": s.notes,
            }
            for key, s in data.disc_sets.items()
        },
        "anomalies": {
            element: {
                "name": a.name,
                "supported": a.supported,
                "mult": a.mult,
                "hits": a.hits,
                "interval": a.interval,
                "duration": a.duration,
                "debuff_note": a.debuff_note,
            }
            for element, a in data.anomalies.anomalies.items()
        },
        "disorder": {
            element: {"base": r.base, "time_mult": r.time_mult,
                      "window": r.window}
            for element, r in data.anomalies.disorder.items()
        },
        "vortex": {
            element: {"mult": r.mult, "time_mult": r.time_mult,
                      "window": r.window}
            for element, r in data.anomalies.vortex.items()
        },
        "skill_tags": data.consts.skill_tags,
        "user_discs": user_discs_payload(data),
        "loadouts": loadouts_payload(data),
    }


def _parse_disc(entry) -> Disc:
    """A request-body disc entry -> :class:`Disc` (substats = total rolls)."""
    if not isinstance(entry, dict):
        raise ValueError("Each disc must be a JSON object")
    substats_raw = entry.get("substats") or {}
    if not isinstance(substats_raw, dict):
        raise ValueError("Disc 'substats' must be an object of stat -> rolls")
    return Disc(
        slot=int(entry.get("slot", 0)),
        main_stat=str(entry.get("main_stat", "")),
        substats={str(s): int(r) for s, r in substats_raw.items()},
        disc_set=entry.get("set") or None,
        element=entry.get("element") or None,
    )


def _number(body: dict, key: str, default: float = 0.0) -> float:
    """A numeric field from the request body; missing/null -> ``default``."""
    value = body.get(key)
    if value is None or value == "":
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{key}' must be a number, got {value!r}")
    return float(value)


#: Calculation modes the UI may request. ``rupture`` is a declared
#: placeholder: selectable in data terms but rejected with a friendly
#: message until Sheer Force damage is modeled (main plan Phase 5+).
CALC_MODES = ("direct", "anomaly", "disorder", "vortex", "rupture")

#: Share of the buildup the anomaly floor treats as unbuffed (teammate
#: dilution / buff downtime): results show the [floor, optimal] range,
#: assuming the user keeps their entered state up for >= 70% of it.
UNBUFFED_FLOOR_SHARE = 0.30


def _body_mode(body: dict) -> str:
    """The request's calculation mode (default ``direct``).

    Raises:
        ValueError: unknown mode, or the ``rupture`` placeholder.
    """
    mode = str(body.get("mode") or "direct")
    if mode not in CALC_MODES:
        raise ValueError(
            f"Unknown mode '{mode}'; options: {list(CALC_MODES)}"
        )
    if mode == "rupture":
        raise ValueError(
            "Rupture (Sheer Force) damage is not modeled yet — placeholder "
            "for main plan Phase 5+. Pick another mode."
        )
    return mode


def _build_config(body: dict) -> CalcConfig:
    """Build a CalcConfig from a request body (shared by /calculate and
    /optimize).

    The body mirrors :class:`~zzz_dmg_calc.api.CalcConfig` (fractions, not
    human percentages — the page converts). Substats arrive as TOTAL rolls,
    same convention as ``loadouts.json``. Mode-specific fields
    (``disorder_replaced`` / ``vortex_infused`` / ``anomaly_mult_override``) are
    only taken from the body when the mode uses them, so a stale field
    from a previous mode can never leak into the calculation.

    Raises:
        ValueError: malformed body fields (the lower layers validate the
            game rules later, during calculation).
    """
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    mode = _body_mode(body)

    discs = [_parse_disc(entry) for entry in body.get("discs", [])]

    set_stacks_raw = body.get("set_stacks") or {}
    if not isinstance(set_stacks_raw, dict):
        raise ValueError("'set_stacks' must be an object of set key -> stacks")
    set_stacks = {str(k): int(v) for k, v in set_stacks_raw.items() if int(v)}

    engine_squad_raw = body.get("engine_squad_buffs") or {}
    if not isinstance(engine_squad_raw, dict):
        raise ValueError(
            "'engine_squad_buffs' must be an object of buff name -> stacks"
        )
    engine_squad_buffs = {
        str(n): int(s) for n, s in engine_squad_raw.items() if int(s)
    }

    dmg_bonus = _number(body, "external_dmg_bonus")
    dmg_taken = _number(body, "external_dmg_taken")
    stun_raw = body.get("stun_multiplier_override")
    anomaly_buff = _number(body, "external_anomaly_buff")

    engine_rank_raw = body.get("engine_rank")
    mindscapes_raw = body.get("mindscapes") or {}
    if not isinstance(mindscapes_raw, dict):
        raise ValueError("'mindscapes' must be an object of level -> stacks")

    supports_raw = body.get("supports") or []
    if not isinstance(supports_raw, list):
        raise ValueError("'supports' must be a list")
    supports = []
    for entry in supports_raw:
        if not isinstance(entry, dict):
            raise ValueError("Each support must be a JSON object")
        buffs_raw = entry.get("buffs") or {}
        scaling_raw = entry.get("scaling_inputs") or {}
        if not isinstance(buffs_raw, dict) or not isinstance(scaling_raw, dict):
            raise ValueError("Support 'buffs'/'scaling_inputs' must be objects")
        engine_buffs_raw = entry.get("engine_buffs") or {}
        if not isinstance(engine_buffs_raw, dict):
            raise ValueError("Support 'engine_buffs' must be an object")
        rank_raw = entry.get("engine_rank")
        supports.append(SupportConfig(
            agent_key=str(entry.get("agent_key", "")),
            buffs={str(n): int(s) for n, s in buffs_raw.items() if int(s)},
            scaling_inputs={str(n): float(v) for n, v in scaling_raw.items()
                            if isinstance(v, (int, float))},
            squad_set=entry.get("squad_set") or None,
            squad_set_stacks=int(entry.get("squad_set_stacks") or 0),
            engine_key=entry.get("engine_key") or None,
            engine_rank=None if rank_raw in (None, "") else int(rank_raw),
            engine_buffs={str(n): int(s)
                          for n, s in engine_buffs_raw.items() if int(s)},
        ))

    return CalcConfig(
        agent_key=str(body.get("agent_key", "")),
        engine_key=body.get("engine_key") or None,
        engine_rank=(
            None if engine_rank_raw in (None, "") else int(engine_rank_raw)
        ),
        boss_name=str(body.get("boss_name", "")),
        skill_multiplier=_number(body, "skill_multiplier"),
        skill_tag=body.get("skill_tag") or None,
        core_passive_active=bool(body.get("core_passive_active")),
        mindscapes={
            int(level): int(stacks)
            for level, stacks in mindscapes_raw.items() if int(stacks)
        },
        additional_ability_stacks=int(body.get("additional_ability_stacks") or 0),
        scaling_inputs={
            str(n): float(v)
            for n, v in (body.get("scaling_inputs") or {}).items()
            if isinstance(v, (int, float))
        },
        supports=supports,
        discs=discs,
        set_stacks=set_stacks,
        engine_squad_buffs=engine_squad_buffs,
        engine_buff_stacks=_number(body, "engine_buff_stacks"),
        external_dmg_bonuses=[dmg_bonus] if dmg_bonus else [],
        external_crit_rate=_number(body, "external_crit_rate"),
        external_crit_dmg=_number(body, "external_crit_dmg"),
        external_res_shred=_number(body, "external_res_shred"),
        external_dmg_taken=[dmg_taken] if dmg_taken else [],
        stun_multiplier_override=(
            None if stun_raw in (None, "") else _number(body, "stun_multiplier_override")
        ),
        # --- Anomaly-mode inputs (mode-gated; see docstring) ---------------
        external_anomaly_proficiency=_number(
            body, "external_anomaly_proficiency"
        ),
        external_anomaly_buff=[anomaly_buff] if anomaly_buff else [],
        disorder_replaced=(
            str(body.get("disorder_replaced") or "") or None
            if mode == "disorder" else None
        ),
        vortex_infused=(
            str(body.get("vortex_infused") or "") or None
            if mode == "vortex" else None
        ),
        disorder_elapsed_seconds=(
            _number(body, "disorder_elapsed_seconds")
            if mode in ("disorder", "vortex") else 0.0
        ),
        external_disorder_mult_add=(
            _number(body, "external_disorder_mult_add")
            if mode in ("disorder", "vortex") else 0.0
        ),
        polarity_disorder=(
            bool(body.get("polarity_disorder")) if mode == "disorder" else False
        ),
        polarity_special_level=int(body.get("polarity_special_level") or 12),
        anomaly_mult_override=(
            (_number(body, "anomaly_mult_override") or None)
            if mode == "anomaly" else None
        ),
        abloom=bool(body.get("abloom")) if mode == "anomaly" else False,
        abloom_element=(
            (str(body.get("abloom_element") or "") or None)
            if mode == "anomaly" else None
        ),
    )


def run_calculation(data: AppData, body: dict) -> dict:
    """Run one calculation for the request body (see :func:`_build_config`).

    ``body["mode"]`` picks the pipeline: ``direct`` (default) runs
    :func:`calculate`; ``anomaly``/``disorder``/``vortex`` run
    :func:`calculate_anomaly`. The response carries ``mode`` back so the
    page renders the matching results panel.

    Raises:
        ValueError: any invalid input — malformed body fields, or the lower
            layers' CalcError/DiscError/etc. (all ValueError subclasses)
            with their own messages.
    """
    mode = _body_mode(body)
    config = _build_config(body)
    if mode == "direct":
        results = calculate(
            config, consts=data.consts, disc_data=data.disc_data,
            bosses=data.bosses, agents=data.agents, engines=data.engines,
            disc_sets=data.disc_sets,
        )
    else:
        results = calculate_anomaly(
            config, consts=data.consts, disc_data=data.disc_data,
            bosses=data.bosses, agents=data.agents, engines=data.engines,
            disc_sets=data.disc_sets, anomaly_data=data.anomalies,
        )

    payload = asdict(results)
    payload["mode"] = mode
    if mode != "direct":
        # Pessimistic floor: assume only UNBUFFED_FLOOR_SHARE of the
        # buildup lacked the entered buffs (teammate dilution / buffs
        # dropping) — the page shows the floor–optimal range.
        floor = calculate_anomaly(
            replace(config, unbuffed_share=UNBUFFED_FLOOR_SHARE),
            consts=data.consts, disc_data=data.disc_data,
            bosses=data.bosses, agents=data.agents, engines=data.engines,
            disc_sets=data.disc_sets, anomaly_data=data.anomalies,
        )
        payload["floor"] = {
            "unbuffed_share": UNBUFFED_FLOOR_SHARE,
            "per_proc": floor.per_proc,
            "per_proc_stunned": floor.per_proc_stunned,
            "full": floor.full,
            "full_stunned": floor.full_stunned,
            "atk_final": floor.atk_final,
            "anomaly_proficiency": floor.anomaly_proficiency,
        }
    # Echo the applied set bonuses so the page can display them (the
    # calculation already validated the discs, so this cannot raise).
    payload["set_bonuses"] = set_bonus_stats(
        config.discs, data.disc_sets, config.set_stacks
    )
    payload["tag_dmg_bonuses"] = {
        data.disc_sets[key].name: value
        for key, value in set_tagged_dmg_bonuses(
            config.discs, data.disc_sets, config.skill_tag
        ).items()
    }
    return payload


def run_optimization(data: AppData, body: dict) -> dict:
    """Search the saved disc inventory for the best build (plan §3/§7).

    The body is the same payload as ``/calculate`` plus an ``optimize``
    object: ``{objective, set_assumption, locked_slots, top_n, min_stats,
    required_4pc}``. Direct mode runs
    :func:`~zzz_dmg_calc.optimizer.optimize`; anomaly / Disorder / Vortex
    modes run :func:`~zzz_dmg_calc.optimizer.optimize_anomaly` (its
    objectives and default differ — full-duration anomaly damage). The
    response serializes :class:`~zzz_dmg_calc.optimizer.OptimizeResult`;
    each build carries its discs (with inventory ids), the 4-piece stacks
    it was evaluated at (for Apply), and its full verified results table.

    Raises:
        ValueError: invalid body/options, an invalid baseline config, or
            an over-budget search (OptimizeError) — all with user-facing
            messages.
    """
    mode = _body_mode(body)
    config = _build_config(body)
    opt_raw = body.get("optimize") or {}
    if not isinstance(opt_raw, dict):
        raise ValueError("'optimize' must be a JSON object")
    locked_raw = opt_raw.get("locked_slots") or []
    if not isinstance(locked_raw, list):
        raise ValueError("'locked_slots' must be a list of slot numbers")
    min_stats_raw = opt_raw.get("min_stats") or {}
    if not isinstance(min_stats_raw, dict):
        raise ValueError("'min_stats' must be an object of stat -> minimum")
    # Anomaly modes default to full-duration damage; direct to average.
    default_objective = "average" if mode == "direct" else "full"
    options = OptimizeOptions(
        objective=str(opt_raw.get("objective") or default_objective),
        set_assumption=str(opt_raw.get("set_assumption") or "max"),
        locked_slots=frozenset(int(s) for s in locked_raw),
        top_n=int(opt_raw.get("top_n") or 5),
        min_stats={str(stat): float(minimum)
                   for stat, minimum in min_stats_raw.items()},
        required_4pc=opt_raw.get("required_4pc") or None,
    )

    if mode == "direct":
        result = optimize(
            config, options, consts=data.consts, disc_data=data.disc_data,
            bosses=data.bosses, agents=data.agents, engines=data.engines,
            disc_sets=data.disc_sets, user_discs=data.user_discs,
        )
    else:
        result = optimize_anomaly(
            config, options, consts=data.consts, disc_data=data.disc_data,
            bosses=data.bosses, agents=data.agents, engines=data.engines,
            disc_sets=data.disc_sets, anomaly_data=data.anomalies,
            user_discs=data.user_discs,
        )

    def build_option(option) -> dict:
        return {
            "value": option.value,
            "delta": option.delta,
            "discs": [
                dict(_disc_json(disc), id=disc_id)
                for disc, disc_id in zip(option.discs, option.disc_ids)
            ],
            "set_stacks": option.set_stacks,
            "changed_slots": list(option.changed_slots),
            "results": asdict(option.results),
            "set_bonuses": set_bonus_stats(
                list(option.discs), data.disc_sets, option.set_stacks
            ),
        }

    return {
        "objective": result.objective,
        "set_assumption": result.set_assumption,
        "baseline_value": result.baseline_value,
        "baseline_feasible": result.baseline_feasible,
        "already_optimal": result.already_optimal,
        "min_stats": result.min_stats,
        "required_4pc": result.required_4pc,
        "best": build_option(result.best),
        "alternatives": [build_option(o) for o in result.alternatives],
        "combos_evaluated": result.combos_evaluated,
        "candidates_per_slot": {
            str(slot): n for slot, n in result.candidates_per_slot.items()
        },
        "discs_pruned": result.discs_pruned,
    }


class UIRequestHandler(BaseHTTPRequestHandler):
    """Routes the endpoints; ``app_data`` is set once at startup.

    Write endpoints (disc inventory, loadouts) mutate the JSON files
    through the validated writers in ``discs.py``, then swap in a freshly
    loaded AppData so every later request sees the new state.
    """

    server_version = "ZZZDmgUI/0.1"
    app_data: AppData
    write_lock = Lock()

    @classmethod
    def _refresh_user_data(cls) -> None:
        """Reload user-writable data after a write (caller holds the lock)."""
        disc_data = cls.app_data.disc_data
        user_discs = load_user_discs(disc_data)
        cls.app_data = replace(
            cls.app_data,
            user_discs=user_discs,
            loadouts=load_loadouts(disc_data, user_discs=user_discs),
        )

    def _send_json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:   # noqa: N802 (stdlib naming)
        if self.path in ("/", "/index.html"):
            raw = INDEX_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif self.path == "/data":
            self._send_json(frontend_payload(self.app_data))
        else:
            self._send_json({"error": f"Unknown path {self.path}"}, status=404)

    def do_POST(self) -> None:   # noqa: N802 (stdlib naming)
        routes = {
            "/calculate": self._post_calculate,
            "/optimize": self._post_optimize,
            "/discs": self._post_disc,
            "/loadouts": self._post_loadout,
        }
        handler = routes.get(self.path)
        if handler is None:
            self._send_json({"error": f"Unknown path {self.path}"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send_json({"error": "Request body is not valid JSON"}, status=400)
            return
        if not isinstance(body, dict):
            self._send_json({"error": "Request body must be a JSON object"},
                            status=400)
            return
        try:
            handler(body)
        except (ValueError, TypeError, KeyError) as exc:
            # Includes CalcError/DiscError/etc. — the layers' messages are
            # user-facing by design, so pass them through verbatim.
            self._send_json({"error": str(exc)}, status=400)

    def do_DELETE(self) -> None:   # noqa: N802 (stdlib naming)
        if not self.path.startswith("/discs/"):
            self._send_json({"error": f"Unknown path {self.path}"}, status=404)
            return
        disc_id = unquote(self.path[len("/discs/"):])
        try:
            with self.write_lock:
                delete_user_disc(disc_id)
                self._refresh_user_data()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json({"user_discs": user_discs_payload(self.app_data)})

    def _post_calculate(self, body: dict) -> None:
        self._send_json(run_calculation(self.app_data, body))

    def _post_optimize(self, body: dict) -> None:
        self._send_json(run_optimization(self.app_data, body))

    def _post_disc(self, body: dict) -> None:
        """Save one disc to the inventory (validated, deduped)."""
        disc = _parse_disc(body)
        with self.write_lock:
            disc_id, created = save_user_disc(disc, self.app_data.disc_data)
            self._refresh_user_data()
        self._send_json({
            "id": disc_id,
            "created": created,
            "user_discs": user_discs_payload(self.app_data),
        })

    def _post_loadout(self, body: dict) -> None:
        """Save the given discs as a loadout of inventory references.

        Each disc is first saved to the inventory (deduped), then the
        loadout is written referencing the resulting ids. A name collision
        without ``overwrite`` returns 400 with ``"exists": true`` so the
        page can ask the user to confirm.
        """
        name = str(body.get("name", "")).strip()
        overwrite = bool(body.get("overwrite"))
        discs = [_parse_disc(entry) for entry in body.get("discs", [])]
        if not discs:
            raise ValueError("A loadout needs at least one equipped disc")
        with self.write_lock:
            if name in self.app_data.loadouts and not overwrite:
                self._send_json(
                    {"error": f"Loadout '{name}' already exists", "exists": True},
                    status=400,
                )
                return
            disc_ids = [
                save_user_disc(d, self.app_data.disc_data)[0] for d in discs
            ]
            save_loadout(
                name, str(body.get("description", "")), disc_ids,
                self.app_data.disc_data, overwrite=overwrite,
            )
            self._refresh_user_data()
        self._send_json({
            "user_discs": user_discs_payload(self.app_data),
            "loadouts": loadouts_payload(self.app_data),
        })

    def log_message(self, format: str, *args) -> None:
        # Quiet: only calculation posts, not every asset fetch.
        if self.command == "POST":
            print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")


def main(port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    """Load data, start the server on 127.0.0.1, and open the browser."""
    handler = UIRequestHandler
    handler.app_data = load_app_data()

    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"=== ZZZ DMG Optimizer UI ===\nServing on {url}  (Ctrl+C to stop)")
    if open_browser:
        Timer(0.3, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ZZZ DMG Optimizer local web UI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true",
                        help="don't open the browser automatically")
    args = parser.parse_args()
    main(port=args.port, open_browser=not args.no_browser)
