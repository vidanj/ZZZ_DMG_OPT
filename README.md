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

## Recent additions (patch 3.3)

- **Anby Demara** (A-rank Electric Stun, Cunning Hares) and her signature
  **Demara Battery Mark II** (+15% to +24% Electric DMG, applied
  automatically). Her M2 (+30% Basic Attack: Thunderbolt vs a Stunned
  enemy) and M6 (+45% Basic / Dash Attack while Charge is held) are
  checkboxes. Her first three Basic hits and her Dash Attack deal
  Physical damage in game, but the calculator treats them as Electric.
  ⚠️ From the datamine, not yet checked in game.
- **Yidhari** (Ice Rupture) and her signature **Kraken's Cradle**. She
  runs in Sheer mode like the other Rupture agents (Max HP feeds her
  Sheer Force). Most of her kit asks for her HP to be at or below 50%:
  her core (+100% DMG), her Additional Ability (+30% CRIT DMG) and the
  engine's CRIT Rate. Tick them for hits at half HP or lower. Her Ether
  Veil squad Max HP is the same buff Lucia gives, so tick only one of
  the two. ⚠️ From the datamine, not yet checked in game.
- **Banyue** (Fire Rupture) and his signature **Wrathful Vajra**, plus
  **Pan Yinhu** (Physical Defense, a Rupture support) and his signature
  **Tremor Trigram Vessel**. Banyue runs in Sheer mode like Yixuan and
  Starlight Billy (Max HP feeds his Sheer Force). Pan gives the squad
  Sheer Force from his initial ATK (enter his panel ATK on his Team
  card) and makes the enemy take +20% DMG (+30% at M1). Banyue's M1
  bonus that only reaches six named skills is its own one-hit toggle,
  so ticking M1 never buffs the wrong move. ⚠️ From the datamine, not
  yet checked in game.
- **Severian** (Wind Attack) and his signature **Incarcerating Grip**.
  His Additional Ability turns CRIT Rate into ATK (+8.5 per 1%, up to
  100%) and reads his **total** CRIT Rate in combat — nothing to type:
  the calculator, and the optimizer for every candidate build, work it
  out from the discs, engine and buffs. ⚠️ From live 3.3 data, not yet
  checked in game.
- **Kit, W-Engine and disc-set effects now all count automatically.**
  Effects that used to ask you to type them into the external fields
  are modeled:
  - W-Engines: Cordis Germina's DEF ignore, Sol Exuvia's, Heartstring
    Nocturne's, Chief Sidekick's and Knight's Extolment's RES ignore.
  - Agents: Lycaon's non-Ice vulnerability (now skips Ice damage on its
    own), Yanagi's M1, Phoenix's longer Burn.
  - Disc sets: Chaos Jazz, Polar Metal and Dawn's Bloom (2pc), Notes
    From the Chained, Chaotic Metal and Shadow Harmony at partial
    stacks, and the Branch & Blade Song 4pc.
  - Windowed bonuses (Polar Metal's Freeze window, Chaos Jazz's
    off-field window) are checkboxes on the set card.

  The external fields are now only for Deadly Assault and boss effects.
  **If you typed any of these in before, clear them** or they count
  twice.
- **Optimizer: base-stat constraints** — a minimum can now be checked
  against the build's out-of-combat panel instead of the combat total.
- **Phoenix**, the first agent whose anomalies can **crit**: her Sear
  gives the squad's Anomaly DMG a CRIT Rate (scaling with her initial
  Anomaly Mastery) and CRIT DMG, so Anomaly mode now shows non-crit /
  crit rows and averages the totals. Her Ablooms run through the
  multiplier override, and her signature **Plume of Rebirth** is
  included. Re-checked against the live 3.3 data.
- **Koleda**, with her 3.2 Potential Awakening (squad DMG +35%, squad
  CRIT DMG up to +35%). Supports can now carry the **Potential level
  they own** when it reaches the squad (Koleda, Rina).
- New W-Engines: **Neon Fantasies** (Nangong Yu's signature, now her
  default), **Plume of Rebirth** (Phoenix) and **Knight's Extolment**
  (Sigrid's signature; usable by Ellen or any Ice attacker until Sigrid
  joins the roster).
- **Abloom element**: an Abloom entered through the multiplier override
  can resolve an anomaly other than the agent's own — pick it, and the
  dealt element, boss RES and element gates follow.
- **Fanged Metal**'s 4pc +35% is generic DMG once triggered, so it now
  also counts for Disorder and Vortex bursts dealt as another element.
- **Optimizer**: searches are now exact and fast (about a second over a
  ~650-disc inventory), can be stopped mid-run, can require a 2-piece
  set alongside a 4-piece, and only credit an anomaly-triggered 4-piece
  when the squad has that element. Also new since the last update:
  **Maximize stat**, and **Fill from a saved build** for supports.

Remielle Dan and the Lumiflux attribute (patch 3.1) have their own
section: [Lumiflux agents](#lumiflux-agents-remielle).

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

> **The CLI is deprecated.** There is also a CLI — `python run.py` —
> that asks the same questions as numbered prompts on the same engine.
> It is kept as-is but **frozen**: it gets no new features or fixes, it
> already lacks some of the UI's options (Maximize stat, the 2pc
> requirement, the loadout-lock bypass…), and a future change may break
> it without notice. All development now targets the web UI, which the
> rest of this README describes.

## The main form

Everything lives on one page. Fill it top to bottom; after the first
**Calculate**, results recalculate automatically on any change, and all
inputs are remembered between sessions.

- **Target & Agent** — pick the boss (sets enemy DEF, elemental
  RES/weakness, and default stun multiplier), the agent, and a
  W-Engine with its **refinement rank** (R1–R5). Engine passives apply
  automatically at the chosen rank — the unconditional parts always,
  the conditional ones through one stack control that starts at its
  usual in-rotation value. A matchup line shows the resulting
  RES/DEF/stun numbers.
  Picking a **Lumiflux** agent adds a **Mutated attribute** dropdown —
  see below.
- **Agent Kit** — the agent's conditionals as real controls: core
  passive toggle, Additional Ability stacks (with a live "condition
  met" hint based on your team), mindscape (M1–M6) toggles/stacks, and
  a Potential dropdown (the one level you own, plus its stacks where it
  stacks) where an agent has one. A squad buff that is the same effect
  as one of these slots is shown once, under the slot. **Riders** are
  the parts with their own toggle — a core clause that depends on the
  squad (Severian's +30% DMG with another Wind agent) or a one-hit
  mindscape / Potential bonus. Every damage-relevant part is modeled;
  the notes list only what has no damage term (resources, Daze buildup)
  and motion values, which you type into the skill multiplier.
  Where a buff scales off a stat the agent has *in combat* (Qingyi's
  and Lighter's Impact, Yuzuha's Anomaly Mastery…), you enter the
  **panel** value and their W-Engine / kit combat bonuses are added
  for you.
- **Team** — up to two **off-field supports**: their team buffs (with
  owner-stat inputs where a buff scales off the support's stats), the
  Potential level they own when it reaches the squad (Koleda, Rina), their
  engine's squad buffs at its own rank, and any squad-facing 4-piece
  set they wear (Swing Jazz, Proto Punk, Astral Voice…).
- **Drive Discs** — the six disc cards (see below).
- **Skill & external buffs** — the skill multiplier (as the in-game
  percent of ATK), its damage type (gates type-conditional bonuses like
  4pc Ultimate DMG), an Aftershock checkbox for hits that also count as
  Aftershocks, and totals for external DMG%, CRIT Rate/DMG, RES shred,
  "DMG taken" debuffs, and the stun multiplier straight off the daze
  bar. The external fields are only for buffs from **outside** the
  kit — Deadly Assault effects, boss mechanics: agents, W-Engines and
  disc sets already count on their own, so typing them here would
  count them twice.

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
| **Anomaly proc** | Your attribute anomaly (Assault, Burn, Shatter, Shock, Corruption): per-proc/tick and full-duration totals. Anomalies crit only through an anomaly-crit source (Phoenix's Sear); then a proc also gets non-crit / crit rows and the totals are the average. An Abloom is this mode with a multiplier override, plus the anomaly it resolves on the target. |
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
- **Bypass loadout locks 🔒** (Drive Discs toolbar) lifts that
  reservation for the current agent: reserved discs can be equipped and
  the optimizer searches them too. The loadouts themselves are not
  changed, and the box is unticked again on every reload.
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
- **Team-element gate (automatic)** — a 4-piece that triggers off an
  anomaly is only credited when someone in the squad (your agent or an
  entered support) has that element: Thunder Metal needs Electric
  (Shock), Fanged Metal Physical (Assault), Inferno Metal Fire (Burn),
  Chaotic Metal Ether (Corruption). Otherwise the search values it at 0
  stacks — including on your current build — and the result says which
  sets were not credited.
- **Build around a 4pc set** — force the build to include a chosen
  4-piece. The optimizer doesn't judge synergy; that part is your call.
- **Also require a 2pc set** — force at least 2 pieces of a set whose
  2-piece the damage math can't value (e.g. King of the Summit's Daze for
  a stunner). Together with a 4pc set the build is exactly 4pc + 2pc.
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
  current build. Each row picks what its minimum is checked against:
  - **in combat** — agent base + core + discs + set 2pc, *plus* the
    always-on W-Engine passive, the 4-piece stacks the search assumes,
    and kit/team buffs. The **Final stats** card is the out-of-combat
    panel and excludes the last three, so it reads lower: a build
    satisfying "AP ≥ 400" can show 382 there. Both numbers are right;
    the combat one is what the damage math uses.
  - **base (panel)** — the build's own out-of-combat stats only, exactly
    the **Final stats** card: no kit, team, engine-passive or combat
    buffs. Use it for a requirement the build itself must reach (an
    agent whose kit reads an *initial* stat, an in-game breakpoint).
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
  finishes. Normal searches finish well inside the budget, so it is
  rarely needed.

**Caveats:** the search is exact and normally fast — about a second for
an unrestricted search over a ~650-disc inventory on a desktop — but the
in-browser build runs the same search a few times slower than local
Python. If a search ever reports an exhausted budget, locks, a required
set, required mains or Fast search cut the space dramatically. A running
search can be aborted with **■ Stop search** (its results are
discarded). Discs reserved by other agents' loadouts never enter the
pool unless **Bypass loadout locks** is ticked.

## Units cheat-sheet

| Where | How to type it |
|---|---|
| Substat amounts | The **`+N` upgrade count** shown in-game (Enter/0 = base substat); never the stat value |
| Skill multiplier & all buffs | Percent as a plain number: `250` = 250%, `25` = 25% |
| Stun multiplier | The percentage the daze bar displays while stunned, as-is (e.g. `235`) |
| Fractions vs percents | The UI always takes percents; it converts internally |
