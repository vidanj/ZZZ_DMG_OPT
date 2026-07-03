# ZZZ DMG Optimizer — How to Run & Enter Data

Calculates **direct-hit damage** in Zenless Zone Zero for a Lv. 60 agent
against a boss. This guide only covers running the program and entering data
into its prompts.

> Unofficial fan tool, not affiliated with or endorsed by HoYoverse.
> Zenless Zone Zero and all game content are trademarks of
> miHoYo/HoYoverse. Code licensed under the [MIT License](LICENSE).

## 1. Requirements

- **Python 3.10 or newer** (developed on 3.13). No third-party packages —
  standard library only.

## 2. Starting the program

Any of these works:

```bash
# From the project root (canonical)
python -m zzz_dmg_calc.main

# From the project root (launcher)
python run.py

# From inside the zzz_dmg_calc/ folder
python main.py
```

On Windows, `py` can replace `python`.

### Web UI (direct hit)

A local browser form covers the **Direct hit** mode with every input visible
and editable at once — change any field and the result updates instantly:

```bash
python run_ui.py                # starts a local server and opens the browser
python run_ui.py --port 9000    # pick another port
python run_ui.py --no-browser   # don't auto-open the browser
```

It serves only on your machine (`127.0.0.1`), uses the same calculation and
validation as the CLI (invalid input shows the same error messages inline),
and remembers your inputs between sessions. Loadouts from
[zzz_dmg_calc/data/loadouts.json](zzz_dmg_calc/data/loadouts.json) can be
loaded as a starting point. Anomaly proc and Disorder modes are CLI-only for
now.

The web UI mirrors all of this with form controls: an engine **rank
selector**, and an **Agent Kit** card with checkboxes/stack inputs for the
modeled core passive, Additional Ability, and mindscapes (unmodeled ones
are listed for reference).

A **Team** card adds up to two **off-field supports** (e.g. Zhao, Dialyn):
their team buffs (with owner-stat inputs where a buff scales, like Zhao's
Max HP), their signature engine's squad buffs (with its own refinement
rank), and any squad-facing 4-piece set they wear (Swing Jazz, Proto Punk,
Astral Voice). Team-conditional abilities show a live "condition met"
hint — e.g. Nekomata's Catwalk lights up when a Physical/Cunning Hares
teammate is picked. The CLI asks the same things as numbered/y-N prompts.

The UI also manages a **disc inventory**
([zzz_dmg_calc/data/user_discs.json](zzz_dmg_calc/data/user_discs.json)):
each disc card has **Save** (stores the disc, skipping exact duplicates) and
**Equip…** (searches your saved discs for that slot by main stat, set, or
substat and fills the card). **Save loadout** stores the currently equipped
discs as a named loadout — discs go into the inventory and the loadout
references their ids, so re-rolling a saved disc updates every loadout using
it. Deleting a disc that a loadout still references is blocked.

## 3. Answering the prompts, step by step

Every menu is numbered — type the **number** of your choice and press Enter.
If an answer is invalid, the program explains why and asks again; nothing
crashes on a typo.

### Step 1 — Boss

Pick the target from the numbered boss list (e.g. `Dead End Butcher`,
`Miasma Priest`). The boss determines enemy DEF, elemental
weakness/resistance, and the stun multiplier — all loaded automatically.

### Step 2 — Agent

Pick the agent. The roster includes the `DUMMY (Ellen copy) [ice]` test
agent, **Nekomata [physical]** (the first fully-modeled agent: core
passive and damage-relevant mindscapes are asked as toggles), and the
Phase 5 anomaly roster. The attack element is matched against the boss's
RES automatically.

For agents with a modeled kit (Nekomata), the program asks about each
**conditional**: core passive active? (y/N), mindscape stacks, etc. Only
mindscapes that buff stats/DMG or debuff the enemy are modeled — motion
value mindscapes (skill Lv. +2) are noted but you enter the higher
multiplier yourself in Step 5b.

### Step 3 — W-Engine

Pick the W-Engine, then its **refinement rank** (R1–R5, Enter = the data
default) — one engine entry covers every rank. Modeled passive parts are
applied automatically at the chosen rank (e.g. Steel Cushion's Physical
DMG +20–40%, plus a y/N toggle for its back-attack bonus); anything not
modeled is printed as a note to add manually as external buffs in Step 6
(e.g. the DUMMY engine's Ice DMG +25% while active).

### Step 4 — Drive discs (slots 1–6)

If saved loadouts exist, a menu appears first: pick a **loadout** (e.g.
`TEST`) to load a whole disc set instantly, choose **Enter discs manually**,
or **No discs**. Loadouts live in
[zzz_dmg_calc/data/loadouts.json](zzz_dmg_calc/data/loadouts.json) — edit
that file to save your own sets (substats stored as total rolls = in-game
`+N` + 1; every disc is validated on load).

For manual entry, each of the six slots asks:

1. **`Equip this slot? [Y/n]`** — press Enter to equip it, or type `n` to
   leave the slot empty and skip to the next one.
2. **Main stat** — slots 1/2/3 are fixed (HP 2200 / ATK 316 / DEF 184) and
   applied automatically. For slots 4/5/6 you pick only the stat **type**
   from the menu; the S-rank Lv. 15 **value** is filled in for you:

   After the main stat, pick the disc's **set** (or "No set / not modeled").
   Set bonuses are applied automatically: **2-piece bonuses always** (e.g.
   Woodpecker Electro +8% CRIT Rate, Branch & Blade Song +16% CRIT DMG), and
   if you equip a **4-piece** with a modeled effect, the program asks how many
   stacks are active (e.g. Woodpecker Electro: 0-3 stacks of +9% ATK; Enter =
   0, since
   stacks only exist in combat after crits land). Sets live in
   [zzz_dmg_calc/data/disc_sets.json](zzz_dmg_calc/data/disc_sets.json) —
   add new ones there.

   | Slot | Main stat options (value auto-applied) |
   |---|---|
   | 4 | ATK% 30 · HP% 30 · DEF% 48 · CRIT Rate 24 · CRIT DMG 48 · Anomaly Proficiency 92 |
   | 5 | ATK% 30 · HP% 30 · DEF% 48 · Attribute DMG% 30 · PEN Ratio 24 |
   | 6 | ATK% 30 · HP% 30 · DEF% 48 · Anomaly Mastery 30 · Impact% 18 · Energy Regen% 60 |

3. **Substats** — enter exactly **4 different substats** per equipped disc.
   For each one, pick the stat from the menu, then type its **upgrade count
   exactly as the game displays it**: a substat shown as `ATK% +2` → type
   `2`; a substat with no `+N` (still at its base value) → press **Enter or
   type 0**. The program adds the substat's initial roll automatically
   (`+N` = N + 1 total rolls) and multiplies by the fixed S-rank roll value:

   | Substat | Per roll | Substat | Per roll |
   |---|---|---|---|
   | ATK (flat) | 19 | CRIT Rate | 2.4% |
   | ATK% | 3% | CRIT DMG | 4.8% |
   | HP (flat) | 112 | PEN (flat) | 9 |
   | HP% | 3% | DEF (flat) | 15 |
   | DEF% | 4.8% | Anomaly Proficiency | 9 |

   **Validation rules** (the program enforces these and asks you to re-enter
   the disc if broken):
   - exactly 4 distinct substats per disc;
   - a substat cannot be the same stat as the disc's main stat;
   - at most **+5** upgrades on one substat;
   - at most **5 upgrades total** across the disc's 4 substats (a Lv. 15
     disc has 5 upgrade events; if it started with 3 substats, one of them
     was spent adding the 4th).

   (Internally the program counts *rolls* — upgrades + the initial roll —
   so validation messages may say "6 rolls max per substat / 9 total".)

### Step 5 — Calculation mode

Choose **Direct hit**, **Anomaly proc**, or **Disorder**:

- **Direct hit** — the v1 flow: you'll be asked for the skill multiplier
  (next section) and CRIT buffs.
- **Anomaly proc** — computes your agent's attribute anomaly (Assault,
  Burn, Shatter, Shock, Corruption). No skill multiplier or CRIT inputs
  (anomalies can't crit); instead you can add flat **Anomaly Proficiency**
  from conditional buffs. Output shows per-tick/proc and full-duration
  totals. Windswept (Wind) is not yet supported.

  > **Important (game-measured):** anomaly damage is "stored" during
  > buildup — the proc uses a **buildup-weighted average** of attacker
  > buffs (and, in teams, of each contributor's stats) active *while
  > building*, not the state at proc time. The CLI therefore asks for
  > **buildup segments**: "roughly X% of the buildup happened with these
  > buffs" (engine stacks, DMG%, flat AP per segment), repeated until you
  > stop — any unassigned share counts as buff-free buildup. Enemy-side
  > modifiers (RES shred, DMG taken, stun) are asked once: they apply at
  > the moment the proc lands. For clean popup comparisons, keep one
  > constant buff state during the whole buildup (a single 100% segment).
- **Disorder** — pick which anomaly is being **replaced** and how much
  time was left on it (or elapsed, for Assault/Shatter). Damage is dealt
  as the replaced element. ⚠️ Disorder values are provisional pending
  in-game calibration.

### Step 5b — Skill multiplier (Direct hit only)

Type the skill's damage multiplier **as a percent of ATK** — the number the
in-game skill description shows. Example: a hit listed as `250% ATK` →
type `250`.

Then pick the move's **damage type** (Basic Attack, Dash Attack, Dodge
Counter, Special, EX Special, Chain Attack, Ultimate, Assists — or
"Untyped" to skip). This gates skill-type-conditional bonuses: e.g. with a
4-piece Puffer Electro equipped, its **Ultimate DMG +20%** applies
automatically when — and only when — the damage type is Ultimate. The web
UI has the same selector next to the skill multiplier.

> **Addendum — whole-move convention.** The calculator assumes the move
> **hits in its entirety**: enter the skill's TOTAL motion value, and the
> result is the whole move's damage. In-game damage popups appear **per
> hit**, so a multi-hit move shows smaller individual numbers on screen
> that **add up** to the calculator's result. (Verified in-game: a 2-hit
> 199.7% move showed two popups of exactly half the calculated total.)
> Before suspecting a discrepancy, sum the popups of the full move.

### Step 6 — External buffs (all optional)

Six prompts, all answered **in percent** (e.g. `25` = 25%). **Press Enter
to skip any of them** (defaults to 0):

| Prompt | What to enter | Example |
|---|---|---|
| Extra DMG% bonuses (total) | Sum of active DMG% buffs: engine passives, set bonuses, team buffs | DUMMY engine passive active → `25` |
| Extra CRIT Rate from conditional buffs | CRIT Rate from stacks/triggers not on your stat sheet | engine on-hit stacks → `20` |
| Extra CRIT DMG from conditional buffs | Skill-specific CRIT DMG boosts — core passives are the main case | Ellen's core passive on charged hits → `100` |
| Enemy RES shred/ignore (total) | Total RES reduction applied to the boss | `20` |
| 'DMG taken' debuffs (total) | Sum of "enemy takes increased DMG" debuffs | `10` |
| Stun DMG multiplier shown under the daze bar | The percentage the game displays under the boss's daze bar while stunned — it already includes every vulnerability effect, so type it as-is. Enter = the boss's default (150 for most) | daze bar shows 235% → `235` |

### Step 7 — Read the results

The program prints the build summary and a damage table:

```
=== Results vs Dead End Butcher ===
Final ATK: 1,837.6   CRIT: 69.8% / 78.8%
Zones: DMG% x1.300 | DEF x0.4569 | RES x1.20 | Taken x1.00 | Stun x1.50

Scenario            Normal       Stunned
----------------------------------------
Non-crit           3,274.4       4,911.6
Crit               5,854.7       8,782.0
Average            5,075.4       7,613.1
```

- **Non-crit / Crit** — the hit's damage without / with a critical hit.
- **Average** — expected value using your CRIT Rate (capped at 100%).
- **Normal / Stunned** — against the boss in its normal state vs. stunned
  (stun multiplier applied).
- The **Zones** line shows each multiplier of the damage formula so you can
  see where the damage comes from (or cross-check another calculator).

## 4. Units cheat-sheet

| Where | How to type it |
|---|---|
| Menu choices | The option's number (`1`, `2`, …) |
| Yes/no | Enter = yes, `n` = no |
| Substat amounts | The **`+N` upgrade count** shown in-game (Enter/0 = base substat); never the stat value |
| Skill multiplier & all buffs | Percent as a plain number: `250` = 250%, `25` = 25% |
| Any optional prompt | Press Enter to accept the default shown in brackets |
