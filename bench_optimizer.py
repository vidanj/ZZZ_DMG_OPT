"""Optimizer performance probe over the REAL user inventory (read-only).

Run after optimizer changes to see wall time / evaluations / width on
scenarios that mirror actual UI use. Not a unit test (timings are
machine-dependent) — the exactness guards live in tests/test_optimizer.py.

    python bench_optimizer.py [--budget N] [--only SUBSTRING]
"""

from __future__ import annotations

import argparse
import time

from zzz_dmg_calc.api import CalcConfig
from zzz_dmg_calc.optimizer import (
    OptimizeError, OptimizeOptions, optimize, optimize_anomaly,
    optimize_sheer,
)
from zzz_dmg_calc.ui.server import load_app_data


def run(label: str, mode: str, config, options, data, user_discs) -> None:
    common = dict(consts=data.consts, disc_data=data.disc_data,
                  bosses=data.bosses, agents=data.agents,
                  engines=data.engines, disc_sets=data.disc_sets,
                  user_discs=user_discs)
    t0 = time.perf_counter()
    try:
        if mode == "anomaly":
            res = optimize_anomaly(config, options, anomaly_data=data.anomalies,
                                   **common)
        elif mode == "rupture":
            res = optimize_sheer(config, options, **common)
        else:
            res = optimize(config, options, **common)
        dt = time.perf_counter() - t0
        flag = " [BUDGET EXHAUSTED]" if res.budget_exhausted else ""
        print(f"{label:34s} {dt:7.2f}s  evals {res.combos_evaluated:>11,}"
              f"  best {res.best.value:>12,.1f}{flag}")
    except OptimizeError as exc:
        dt = time.perf_counter() - t0
        print(f"{label:34s} {dt:7.2f}s  ERROR: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=None,
                        help="combo budget override (default: optimizer's)")
    parser.add_argument("--only", default="",
                        help="run only scenarios whose label contains this")
    args = parser.parse_args()

    data = load_app_data()
    print(f"inventory: {len(data.user_discs)} discs")
    boss = next(iter(data.bosses.values())).name
    budget = {} if args.budget is None else {"combo_budget": args.budget}

    def config(agent_key: str, **extra) -> CalcConfig:
        return CalcConfig(agent_key=agent_key, boss_name=boss,
                          skill_multiplier=2.5, **extra)

    nekomata = config("nekomata")
    scenarios = [
        ("unrestricted", "direct", nekomata, {}),
        ("fast (cap 15)", "direct", nekomata, {"candidate_cap": 15}),
        ("2pc priority", "direct", nekomata, {"sets_only": True}),
        ("min CRIT Rate 50%", "direct", nekomata,
         {"min_stats": {"CRIT Rate": 0.5}}),
        ("required 4pc Woodpecker", "direct", nekomata,
         {"required_4pc": "WOODPECKER_ELECTRO"}),
        ("objective crit", "direct", nekomata, {"objective": "crit"}),
        ("maximize CRIT Rate (5%)", "direct", nekomata,
         {"maximize_stat": "CRIT Rate"}),
        ("anomaly piper (full)", "anomaly", config("piper"),
         {"objective": "full"}),
        ("rupture yixuan", "rupture", config("yixuan"), {}),
    ]
    for label, mode, cfg, extra in scenarios:
        if args.only and args.only not in label:
            continue
        run(label, mode, cfg, OptimizeOptions(top_n=5, **extra, **budget),
            data, data.user_discs)


if __name__ == "__main__":
    main()
