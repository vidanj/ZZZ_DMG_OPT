# ZZZ DMG Optimizer

Damage calculator and gear optimizer for **Zenless Zone Zero**: build a
Lv. 60 agent, point it at a boss, and see exactly what a hit — or an
anomaly, Disorder, Vortex, Sheer-damage proc, or Luminize — deals, then
let the optimizer search your saved discs for a better build.

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

## Recent additions (patch 3.1)

- **Remielle Dan**, the first **Lumiflux** agent, with the whole of the
  attribute's machinery: **Attribute Mutation**, the **Refringe**
  multiplier, and a dedicated **Luminize** calculation mode. See
  [Lumiflux agents](#lumiflux-agents-remielle) for how to drive it.
- Her signature **Ode of Resurrected Wings** and the craftable
  **Weeping Gemini** (whose stacking Anomaly Proficiency is a new kind of
  engine buff), plus the **Feathered Fate** drive set.
- The optimizer now works in Luminize mode too.
- Every dropdown (agents, W-Engines, bosses, sets, stats…) is sorted
  alphabetically.

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
  Picking a **Lumiflux** agent adds a **Mutated attribute** dropdown —
  see below.
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
| **Luminize (Remielle)** | Remielle's own damage, fired from stored Voidflares. Its own mode because a Luminize scales off a *teammate's* anomaly — see below. |

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

### Lumiflux agents (Remielle)

Lumiflux is a seventh attribute with no RES of its own and no discs that
buff it, so it works differently in three places.

**1. Attribute Mutation.** Her damage counts as the attribute of the
**next agent in your squad**. Selecting her reveals a **Mutated
attribute** dropdown; that choice drives the boss RES, whether a slot-5
Attribute DMG% disc counts, whether an element-gated 2-piece set bonus
counts (see below), element-gated buffs, and which anomaly her hits
build. The control only exists for Lumiflux agents and is ignored
entirely for everyone else.

**2. Refringe** is her signature multiplier: while an enemy carries
Lumiflux Buildup, the next anomaly a **teammate** inflicts is enhanced by
her *Refringe Coefficient* — `0.02% × her Anomaly Proficiency`, plus a
flat `+10%` with three Anomaly agents and `+20%` at her M2 (these add,
they don't multiply). Use it by putting Remielle in the **Team** card and
ticking the three `Refringe:` buffs, entering her panel AP. It applies in
every anomaly-family mode and to everything derived from the anomaly —
Disorder, Vortex, Abloom — and never to direct hits. It occupies its own
multiplier zone, shown as **Refringe** on the Zones line.

**3. Luminize mode** computes her own damage. The counter-intuitive part:

> **The agent you select is the teammate whose Refringed anomaly the
> Voidflare stored — not Remielle.** A Luminize inherits *their* ATK,
> DMG% and PEN; only the multiplier and the Anomaly DMG bonus are hers.
> She stays in the Team card as a support.

Then pick which of her skills fires it, enter her AP, the number of
stored Voidflares, and tick M4/M1 if you own them. The mode computes the
multiplier (`skill value × (1 + 0.2% × her AP)`, ×1.12 at M4) and, most
importantly, **swaps the Anomaly-DMG bracket over to her** — the
attacker's own anomaly-DMG buffs are excluded automatically and listed as
such in the buff breakdown, because only *general* Anomaly DMG bonuses
reach a Luminize.

Worth knowing about Voidflares: **all stored ones fire at once** (3 stored
= 3 Luminize instances), and only the enhanced Basics consume them — the
Entry Assist and the Ultimate leave them intact, so one set of three can
pay out three times in a rotation. A Voidflare stores the source anomaly's
Effect Strength *and element* but not its type, and only an anomaly
**application** ever creates one (Ablooms and Disorders don't). If your
stored Voidflares came from different teammates, set the count to 1 and
run once per source agent, then add the results.

⚠️ Everything Lumiflux is **provisional**: the attribute shipped with
patch 3.1 and none of it has been calibrated against in-game damage
popups yet.

## Discs, inventory, and loadouts

Each of the six disc cards takes a main stat (slots 1–3 are fixed
HP/ATK/DEF; slots 4–6 offer the real main-stat pools at S-rank Lv. 15
values), the disc's **set**, and exactly four substats entered as the
**`+N` upgrade count the game displays** (Enter/0 = base roll — the
initial roll is added automatically). Game rules are enforced: 4
distinct substats, none equal to the main stat, at most +5 on one
substat and 5 upgrades total. 4-piece effects with stacking parts ask
for their active stacks.

- **Element-gated 2-piece bonuses.** A set whose 2pc is an elemental
  DMG bonus (Fanged Metal = *Physical* DMG +10%, Thunder Metal =
  *Electric*, Polar Metal / Notes From the Chained = *Ice*, Inferno
  Metal = *Fire*, Chaotic Metal / The Sky Ablaze = *Ether*, Wuthering
  Salon = *Wind*, White Water Ballad = *Physical*) only counts when
  your damage is **dealt** as that element — Attribute Mutation
  included, so an electric-mutated Remielle gets Thunder Metal's 10%
  and nothing from Fanged Metal. Off-element, the 2pc is worth 0 and
  the optimizer stops valuing it. (The Sky Ablaze's 4pc CRIT DMG is
  Ether-gated in-game as well; that one is still credited to any
  wearer, so only build it on Ether agents.)
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
- A loadout also records the **W-Engine** selected when you saved it, so
  **Load** restores the whole build — and, more usefully, the app can
  work out that build's character panel. Put that agent in a support
  slot and their kit's "initial ATK / initial Max HP" boxes can be
  filled from the build with **Fill from a saved build** instead of you
  reading the numbers off the game; the boxes stay editable, and typing
  in one takes it back over. Loadouts saved before this have no engine
  on file and offer no values until you save them again.

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
(default: full duration); Luminize mode offers one Voidflare or the whole
stored volley.

In **Luminize mode the optimizer tunes the teammate's discs**, not
Remielle's — which is the right target, since a Luminize inherits that
teammate's ATK/DMG%/PEN, and the same build also raises their own anomaly
damage. Remielle's side (the multiplier, her Anomaly DMG bonus, Refringe)
is constant during the search. The page states this above the controls.

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
  **These are combat totals** — agent base + core + discs + set 2pc,
  *plus* the always-on W-Engine passive, the 4-piece stacks the search
  assumes, and kit/team buffs. The **Final stats** card is the
  out-of-combat panel and excludes the last three, so it reads lower:
  a build satisfying "AP ≥ 400" can show 382 there. Both numbers are
  right; the combat one is what the damage math uses.
- **Maximize stat** — make a stat outrank damage: the search reports the
  build carrying the most of it among those within the *allowed damage
  loss* (default 5%) of the best build found. For a stat the optimized
  hit doesn't use — Remielle's Luminize scales with her Anomaly
  Proficiency, which her direct hits ignore, so a pure damage search
  would strip it. Same stat list as the constraints; the panel reports
  what the stat reached and what it cost against the pure-damage
  optimum. Ties in the stat are broken by damage. Note it runs the
  search **twice** (best damage, then most of the stat), so expect
  roughly double the time.
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
