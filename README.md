# ZZZ DMG Optimizer

Damage calculator and gear optimizer for **Zenless Zone Zero**: build a
Lv. 60 agent, point it at a boss, and see exactly what a hit — or an
anomaly, Disorder, Vortex, or Sheer-damage proc — deals, then let the
optimizer search your saved discs for a better build.

> **A personal note before anything else.** This is a completely
> personal project, made mostly for my own use — for fun, and honestly,
> out of boredom. It models the things *I* need, in the order *I* run
> into them, and it makes no promises of completeness, correctness for
> your use case, or timely updates. If it happens to be useful to
> someone else: great, genuinely! But temper your expectations
> accordingly — this is a hobby toy, not a maintained community tool.

> Unofficial fan tool, not affiliated with or endorsed by HoYoverse.
> Zenless Zone Zero and all game content are trademarks of
> miHoYo/HoYoverse. Code licensed under the [MIT License](LICENSE).

## Use it online (no install)

The app is published at **<https://vidanj.github.io/ZZZ_DMG_OPT/>** and
runs **entirely in your browser** as Python-on-WebAssembly (Pyodide):
nothing is uploaded anywhere, and your disc inventory and loadouts are
saved in the browser's local storage on your device. The first visit
downloads the runtime (~8 MB, cached afterwards). Use the inventory's
**Export / Import** buttons to back up your data or move it between
browsers — local storage is per-browser and can be wiped by clearing
site data.

## Run it locally

Requires **Python 3.10+** (developed on 3.13), standard library only:

```bash
python run_ui.py                # local server + opens the browser
python run_ui.py --port 9000    # pick another port
python run_ui.py --no-browser   # don't auto-open the browser
```

It serves only on your machine (`127.0.0.1`). Locally your inventory and
loadouts live in plain JSON files
(`zzz_dmg_calc/data/user_discs.json` / `loadouts.json`), which also
makes them trivial to back up or hand-edit. Local mode is noticeably
faster than the browser build for heavy optimizer searches.

(There is also a CLI — `python run.py` — that asks the same questions as
numbered prompts. Same engine, same validation; the UI is the
recommended way to use the tool and the rest of this README describes
it.)

## The main form

Everything lives on one page. Fill it top to bottom; after the first
**Calculate**, results recalculate automatically on any change, and all
inputs are remembered between sessions.

- **Target & Agent** — pick the boss (sets enemy DEF, elemental
  RES/weakness, and default stun multiplier), the agent, and a
  W-Engine with its **refinement rank** (R1–R5). Modeled engine
  passives apply automatically at the chosen rank; anything not modeled
  is shown as a note telling you what to enter manually as an external
  buff. A matchup line shows the resulting RES/DEF/stun numbers.
- **Agent Kit** — the agent's conditionals as real controls: core
  passive toggle, Additional Ability stacks (with a live "condition
  met" hint based on your team), mindscape (M1–M6) toggles/stacks, and
  Potential levels where an agent has them. Only damage-relevant parts
  are modeled; unmodeled ones are listed as reference notes so you can
  add them as external buffs.
- **Team** — up to two **off-field supports**: their team buffs (with
  owner-stat inputs where a buff scales off the support's stats), their
  engine's squad buffs at its own rank, and any squad-facing 4-piece
  set they wear (Swing Jazz, Proto Punk, Astral Voice…).
- **Drive Discs** — the six disc cards (see below).
- **Skill & external buffs** — the skill multiplier (as the in-game
  percent of ATK), its damage type (gates type-conditional bonuses like
  4pc Ultimate DMG), an Aftershock checkbox for hits that also count as
  Aftershocks, and totals for external DMG%, CRIT Rate/DMG, RES shred,
  "DMG taken" debuffs, and the stun multiplier straight off the daze
  bar.

The results panel shows the damage table for the selected mode, the
**Zones** line (each multiplier of the formula, for cross-checking other
calculators), the applied set bonuses, and an approximate **in-game
character panel** for the build (no combat buffs) so you can sanity-check
against your actual stat screen.

## Calculation modes

One dropdown switches the whole pipeline; each mode shows only the
inputs it uses, so nothing stale leaks between modes.

| Mode | What it computes |
|---|---|
| **Direct hit** | The classic crit-table damage of one move: non-crit / crit / average, normal and stunned. |
| **Anomaly proc** | Your attribute anomaly (Assault, Burn, Shatter, Shock, Corruption): per-proc/tick and full-duration totals. Anomalies can't crit. |
| **Disorder burst** | Triggering a second anomaly while one is active: pick the replaced element and its remaining/elapsed time. |
| **Vortex burst** | The Windswept + second-anomaly interaction (wind teams). |
| **Sheer damage (Rupture)** | Sheer Force agents: Sheer damage crits like a direct hit, so it gets the full crit table, plus Sheer-specific inputs (flat Sheer Force, Sheer DMG%). |

Mode-specific caveats worth knowing:

- **Anomaly damage is buildup-weighted**: the proc snapshots the buffs
  that were active *while building the anomaly*, not at proc time. The
  UI assumes the state you entered was up for the whole buildup and
  additionally shows a **pessimistic floor** assuming it was only up for
  ~70% of it — read the result as a floor–optimal range. Enemy-side
  modifiers (RES shred, DMG taken, stun) apply at the moment the proc
  lands and are not averaged.
- **Disorder and Vortex numbers are provisional** pending more in-game
  calibration; the Windswept "+10% DMG taken" bracket likewise.
- Off-field damage (assists and the like) is not modeled yet.

## Discs, inventory, and loadouts

Each of the six disc cards takes a main stat (slots 1–3 are fixed
HP/ATK/DEF; slots 4–6 offer the real main-stat pools at S-rank Lv. 15
values), the disc's **set**, and exactly four substats entered as the
**`+N` upgrade count the game displays** (Enter/0 = base roll — the
initial roll is added automatically). Game rules are enforced: 4
distinct substats, none equal to the main stat, at most +5 on one
substat and 5 upgrades total. 4-piece effects with stacking parts ask
for their active stacks.

- **Save** on a card stores the disc in your **inventory** (exact
  duplicates are detected and reused, never duplicated).
- **Equip… / Browse inventory** opens the inventory modal: filter by
  slot, set, main stat, element, or free text, and equip a disc into
  its slot with one click. Deleting a disc is blocked while any loadout
  still references it.
- **Save loadout** stores the six equipped discs as a named loadout
  belonging to the current agent. Loadouts hold *references* into the
  inventory, so re-saving an improved disc updates every loadout using
  it. **A loadout reserves its discs for its agent**: when you optimize
  a different agent, reserved discs are excluded from the search (you
  can't equip them twice), and the result reports how many were
  excluded. Deleting a loadout frees its discs; the discs themselves
  stay in the inventory.

## The optimizer

After any successful Calculate, the **Optimizer** card appears. It keeps
*everything* you entered — agent, kit state, team, buffs, mode — exactly
as-is and searches your saved inventory (plus what's equipped) for the
disc combination that maximizes the chosen objective. Results come back
as the best build plus up to four alternatives, each with its damage
delta, which slots changed, its full verified results table, and an
**Apply** button that equips it into the form.

**Objectives** follow the calculated mode: direct and Sheer modes offer
average / crit / non-crit damage, each in normal or stunned state;
anomaly modes offer full-duration or per-proc damage, normal or stunned
(default: full duration).

**Options:**

- **Newly formed 4pc set effects** — when a candidate build completes a
  4-piece your current build doesn't have, either *assume it's active*
  at max stacks (default, itemized in the result) or *never assume* and
  count only 2-piece values. With conditional 4-piece effects the
  optimistic assumption can overrate a set — check the itemization.
- **Build around a 4pc set** — force the build to include a chosen
  4-piece. The optimizer doesn't judge synergy; that part is your call.
- **Required slot 4/5/6 main stats** — pin the main-stat types.
- **2pc priority** — every disc must complete a set bonus (no set-less
  or singleton pieces).
- **🔒 per-slot locks** — tick the lock on any disc card to keep that
  slot's current disc untouched.
- **Minimum-stat constraints** — require the final build to keep at
  least a given total of ATK/HP/DEF, CRIT Rate/DMG, PEN (flat or
  Ratio), Impact, Energy Regen, Anomaly Proficiency/Mastery (plus Sheer
  Force in Rupture mode). Note the trade-off: the best build *meeting
  the constraints* may deal less damage than your unconstrained
  current build.
- **Fast search** — only the top 15 discs per slot by standalone value
  are considered (equipped and required-set discs always survive the
  cut). Much faster on big inventories but **approximate**: the true
  optimum may use a disc outside the cut.
- **Bypass search budget** — by default the search stops at a work
  budget and returns the best build found so far (flagged when the
  budget was exhausted). This option runs until the search truly
  finishes: exact, but an unrestricted search over a large inventory
  can take **very** long — prefer adding locks, a required set,
  required mains, or Fast search first.

**Caveats:** an unrestricted search grows combinatorially with your
inventory (hundreds of discs → minutes, or much more with the budget
bypassed), and the in-browser build runs the same search a few times
slower than local Python. Any filter above cuts the space dramatically.
Discs reserved by other agents' loadouts never enter the pool.

## Units cheat-sheet

| Where | How to type it |
|---|---|
| Substat amounts | The **`+N` upgrade count** shown in-game (Enter/0 = base substat); never the stat value |
| Skill multiplier & all buffs | Percent as a plain number: `250` = 250%, `25` = 25% |
| Stun multiplier | The percentage the daze bar displays while stunned, as-is (e.g. `235`) |
| Fractions vs percents | The UI always takes percents; it converts internally |
