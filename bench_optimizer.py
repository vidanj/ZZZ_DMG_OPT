"""Optimizer performance probe over the REAL user inventory (read-only).

Run after optimizer changes to see wall time / evaluations / width on
scenarios that mirror actual UI use. Not a unit test (timings are
machine-dependent) — the exactness guards live in tests/test_optimizer.py.

    python bench_optimizer.py [--budget N]
"""

from __future__ import annotations

import argparse
import time

from zzz_dmg_calc.optimizer import OptimizeError, OptimizeOptions, optimize
from zzz_dmg_calc.ui.server import _build_config, load_app_data


def run(label: str, config, options, data, user_discs) -> None:
    t0 = time.perf_counter()
    try:
        res = optimize(
            config, options, consts=data.consts, disc_data=data.disc_data,
            bosses=data.bosses, agents=data.agents, engines=data.engines,
            disc_sets=data.disc_sets, user_discs=user_discs,
        )
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
    args = parser.parse_args()

    data = load_app_data()
    print(f"inventory: {len(data.user_discs)} discs")
    boss = next(iter(data.bosses.values())).name
    config = _build_config({"agent_key": "nekomata", "boss_name": boss,
                            "skill_multiplier": 2.5, "discs": []})
    budget = {} if args.budget is None else {"combo_budget": args.budget}

    run("unrestricted", config,
        OptimizeOptions(top_n=5, **budget), data, data.user_discs)
    run("fast (cap 15)", config,
        OptimizeOptions(top_n=5, candidate_cap=15, **budget),
        data, data.user_discs)
    run("2pc priority", config,
        OptimizeOptions(top_n=5, sets_only=True, **budget),
        data, data.user_discs)
    run("min CRIT Rate 50%", config,
        OptimizeOptions(top_n=5, min_stats={"CRIT Rate": 0.5}, **budget),
        data, data.user_discs)


if __name__ == "__main__":
    main()
