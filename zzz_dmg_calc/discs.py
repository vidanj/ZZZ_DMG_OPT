"""Drive disc system: main stat tables, substat rolls, and validation.

All game values live in ``data/discs.json`` (see DOCS/sources.md for where
they were verified); this module loads that file, validates it, and validates
*user-described* discs against it.

The scope rules (DOCS/zzz_dmg_calc_plan.md §1/§4) are enforced here:

- Discs are S-rank Lv. 15 only. The user picks a main stat *type* per slot;
  the value always comes from the table — anything else is rejected.
- Substats are entered as **roll counts** and multiplied by the fixed
  per-roll value. A Lv. 15 disc has exactly 4 distinct substats, at most
  6 rolls on any one substat, at most 9 rolls total, and no substat may
  duplicate the disc's main stat.

Usage::

    from zzz_dmg_calc.discs import Disc, load_disc_data, disc_stats

    data = load_disc_data()
    disc = Disc(slot=4, main_stat="CRIT Rate",
                substats={"CRIT DMG": 5, "ATK%": 2, "ATK": 1, "PEN": 1})
    stats = disc_stats(disc, data)   # {"CRIT Rate": 0.24, "CRIT DMG": 0.24, ...}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .enemies import ELEMENTS

#: Default location of the disc data file, relative to this package.
DATA_FILE = Path(__file__).parent / "data" / "discs.json"

#: Default location of the saved disc loadouts (user data, optional).
LOADOUTS_FILE = Path(__file__).parent / "data" / "loadouts.json"

#: Default location of the disc set registry.
SETS_FILE = Path(__file__).parent / "data" / "disc_sets.json"

#: Default location of the user disc inventory (user data, optional).
USER_DISCS_FILE = Path(__file__).parent / "data" / "user_discs.json"

#: Slots whose main stat is fixed (single option in the table).
SLOTS = (1, 2, 3, 4, 5, 6)


class DiscError(ValueError):
    """Raised for invalid disc data files or invalid user-described discs."""


@dataclass(frozen=True)
class DiscData:
    """Validated, immutable view of ``data/discs.json``.

    Attributes:
        main_stats: slot -> {stat name -> Lv. 15 value}.
        substat_rolls: substat name -> value gained per roll.
        substats_per_disc: exact number of distinct substats on a Lv. 15 disc.
        max_rolls_per_substat: maximum rolls any single substat can hold.
        max_total_rolls: maximum rolls summed across the disc's substats.
    """

    main_stats: dict[int, dict[str, float]]
    substat_rolls: dict[str, float]
    substats_per_disc: int
    max_rolls_per_substat: int
    max_total_rolls: int


@dataclass(frozen=True)
class Disc:
    """A user-described S-rank Lv. 15 drive disc.

    Attributes:
        slot: Partition 1-6.
        main_stat: Stat *type* name; its value is looked up in the table.
        substats: substat name -> roll count (each roll counts, including the
            substat's initial appearance).
        disc_set: Key into the set registry (``data/disc_sets.json``), or
            ``None`` for a disc whose set doesn't matter / isn't modeled.
        element: Which element an **Attribute DMG%** main belongs to (the
            physical drop is element-specific, e.g. a Fire DMG% disc). Only
            allowed on Attribute DMG% mains. ``None`` on such a main is
            legacy data: it keeps the original assume-it-matches behavior
            (optimizer_plan.md §11 E1) — set the element and re-save.
    """

    slot: int
    main_stat: str
    substats: dict[str, int] = field(default_factory=dict)
    disc_set: str | None = None
    element: str | None = None


def _require_positive_int(data: dict, key: str) -> int:
    if key not in data:
        raise DiscError(f"discs.json 'substat_limits' is missing '{key}'")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DiscError(f"'{key}' must be a positive integer, got {value!r}")
    return value


def load_disc_data(path: Path = DATA_FILE) -> DiscData:
    """Load and validate the drive disc data file.

    Raises:
        DiscError: if the file is missing, malformed, or fails validation
            (missing slots, non-numeric values, bad limits).
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DiscError(f"Disc data file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise DiscError(f"Disc data file is not valid JSON: {exc}") from None

    mains_raw = raw.get("main_stats")
    if not isinstance(mains_raw, dict):
        raise DiscError("'main_stats' must be an object of slot -> stat table")
    main_stats: dict[int, dict[str, float]] = {}
    for slot_str, table in mains_raw.items():
        try:
            slot = int(slot_str)
        except ValueError:
            raise DiscError(f"Slot key {slot_str!r} is not an integer") from None
        if not isinstance(table, dict) or not table:
            raise DiscError(f"Slot {slot} main stat table must be a non-empty object")
        validated: dict[str, float] = {}
        for stat, value in table.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise DiscError(
                    f"Main stat '{stat}' in slot {slot} must be a positive number, "
                    f"got {value!r}"
                )
            validated[stat] = float(value)
        main_stats[slot] = validated
    missing = [s for s in SLOTS if s not in main_stats]
    if missing:
        raise DiscError(f"'main_stats' is missing slots: {missing}")

    rolls_raw = raw.get("substat_rolls")
    if not isinstance(rolls_raw, dict) or not rolls_raw:
        raise DiscError("'substat_rolls' must be a non-empty object of stat -> value")
    substat_rolls: dict[str, float] = {}
    for stat, value in rolls_raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise DiscError(
                f"Substat roll value for '{stat}' must be a positive number, "
                f"got {value!r}"
            )
        substat_rolls[stat] = float(value)

    limits = raw.get("substat_limits")
    if not isinstance(limits, dict):
        raise DiscError("'substat_limits' must be an object")

    return DiscData(
        main_stats=main_stats,
        substat_rolls=substat_rolls,
        substats_per_disc=_require_positive_int(limits, "substats_per_disc"),
        max_rolls_per_substat=_require_positive_int(limits, "max_rolls_per_substat"),
        max_total_rolls=_require_positive_int(limits, "max_total_rolls"),
    )


def validate_disc(disc: Disc, data: DiscData) -> None:
    """Validate a user-described disc against the data tables.

    Enforces (per plan §4): valid slot, main stat legal for that slot, legal
    substat types, no substat duplicating the main stat, exactly
    ``substats_per_disc`` distinct substats, 1..``max_rolls_per_substat``
    rolls each, and at most ``max_total_rolls`` rolls overall.

    Raises:
        DiscError: describing the first rule the disc violates.
    """
    if disc.slot not in data.main_stats:
        raise DiscError(
            f"Invalid slot {disc.slot}; valid slots: {sorted(data.main_stats)}"
        )

    slot_table = data.main_stats[disc.slot]
    if disc.main_stat not in slot_table:
        raise DiscError(
            f"'{disc.main_stat}' is not a legal main stat for slot {disc.slot}; "
            f"options: {sorted(slot_table)}"
        )

    if disc.element is not None:
        if disc.main_stat != "Attribute DMG%":
            raise DiscError(
                f"'element' only applies to an Attribute DMG% main "
                f"(slot {disc.slot} has '{disc.main_stat}')"
            )
        if disc.element not in ELEMENTS:
            raise DiscError(
                f"Unknown element '{disc.element}'; "
                f"options: {list(ELEMENTS)}"
            )

    if len(disc.substats) != data.substats_per_disc:
        raise DiscError(
            f"A Lv. 15 disc has exactly {data.substats_per_disc} distinct "
            f"substats, got {len(disc.substats)}"
        )

    total_rolls = 0
    for stat, rolls in disc.substats.items():
        if stat not in data.substat_rolls:
            raise DiscError(
                f"'{stat}' is not a legal substat; options: "
                f"{sorted(data.substat_rolls)}"
            )
        if stat == disc.main_stat:
            raise DiscError(
                f"Substat '{stat}' duplicates the disc's main stat"
            )
        if isinstance(rolls, bool) or not isinstance(rolls, int) or rolls < 1:
            raise DiscError(
                f"Roll count for '{stat}' must be an integer >= 1, got {rolls!r}"
            )
        if rolls > data.max_rolls_per_substat:
            raise DiscError(
                f"'{stat}' has {rolls} rolls; maximum per substat is "
                f"{data.max_rolls_per_substat}"
            )
        total_rolls += rolls

    if total_rolls > data.max_total_rolls:
        raise DiscError(
            f"Disc has {total_rolls} total rolls; maximum is {data.max_total_rolls}"
        )


@dataclass(frozen=True)
class SetBonus4pc:
    """A modeled 4-piece effect: a stacking conditional stat bonus.

    Total contribution = ``per_stack × active stacks`` where the active
    stack count (0..``max_stacks``) is a runtime input — 4-piece effects
    are combat-conditional, unlike the always-on 2-piece bonuses.

    ``auto``: the effect ramps up early and holds through a rotation
    (e.g. Wuthering Salon's AP on EX), so the front end defaults it to
    ``max_stacks``. The anomaly range's unbuffed floor covers the ramp /
    uptime, so no manual toggle is needed (adopted 2026-07-04). Still
    overridable. Situational effects leave ``auto`` false.
    """

    stat: str
    per_stack: float
    max_stacks: int
    auto: bool = False
    note: str = ""


@dataclass(frozen=True)
class SetDmgBonus:
    """A skill-type-conditional DMG% bonus granted by a 4-piece set.

    Applies (unconditionally, no stacks) only when the calculated hit's
    skill tag matches ``skill_tag`` — e.g. Puffer Electro's Ultimate DMG
    +20% applies only to hits tagged ``"ultimate"``. Tag keys are defined
    in ``constants.json`` (``skill_tags``) and validated at calculation
    time so a typo here can't silently never-apply.
    """

    skill_tag: str
    value: float
    note: str = ""


#: Effect kinds a squad-facing 4-piece may grant the on-field agent.
SQUAD_4PC_KINDS = ("dmg_bonus", "crit_rate", "crit_dmg", "res_shred",
                   "dmg_taken", "daze_bonus")


@dataclass(frozen=True)
class SquadBonus4pc:
    """A team-facing 4-piece effect granted to the ON-FIELD agent.

    Worn by an off-field support (e.g. Swing Jazz on a stunner); the
    on-field attacker receives ``value × active stacks`` of ``kind``
    (DMG%, CRIT DMG, ...). Active stacks are a runtime input
    (``max_stacks`` 1 = on/off).
    """

    value: float
    kind: str = "dmg_bonus"
    max_stacks: int = 1
    auto: bool = False
    note: str = ""


@dataclass(frozen=True)
class DiscSet:
    """One entry of the disc set registry (``data/disc_sets.json``).

    Attributes:
        bonus_2pc: stat -> value, unconditional, applied automatically when
            2+ pieces are equipped (the in-game stat sheet shows these).
        bonus_4pc: Modeled stacking effect, or ``None`` if the set's 4-piece
            isn't modeled (its ``notes`` say why).
        bonus_4pc_dmg: Skill-type-conditional DMG% bonuses that come with
            the 4-piece (gated by the hit's skill tag, not by stacks).
        auto_4pc_dmg: An always-assumed-on, general DMG% part of the
            4-piece that ramps up early and holds through a rotation
            (e.g. Wuthering Salon's +18% on Windswept trigger). Applied
            automatically (no toggle) to the ON-FIELD wearer via the
            dilutable kit-DMG% path, so the anomaly range's unbuffed floor
            covers the ramp / uptime. 0 = none.
        squad_4pc: Team-facing 4-piece part (worn by a support, buffs the
            on-field agent), or ``None``.
    """

    key: str
    name: str
    bonus_2pc: dict[str, float]
    bonus_4pc: SetBonus4pc | None = None
    bonus_4pc_dmg: tuple[SetDmgBonus, ...] = ()
    auto_4pc_dmg: float = 0.0
    squad_4pc: SquadBonus4pc | None = None
    notes: str = ""


def load_disc_sets(path: Path = SETS_FILE) -> dict[str, DiscSet]:
    """Load and validate the disc set registry.

    Raises:
        DiscError: if the file is missing/malformed or an entry is invalid.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DiscError(f"Disc set registry not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise DiscError(f"Disc set registry is not valid JSON: {exc}") from None

    entries = raw.get("sets")
    if not isinstance(entries, dict) or not entries:
        raise DiscError("'sets' must be a non-empty object of key -> set")

    sets: dict[str, DiscSet] = {}
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            raise DiscError(f"Set '{key}' must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise DiscError(f"Set '{key}' is missing a valid 'name'")

        bonus_2pc = entry.get("bonus_2pc")
        if not isinstance(bonus_2pc, dict) or not bonus_2pc:
            raise DiscError(f"Set '{key}': 'bonus_2pc' must be a non-empty object")
        for stat, value in bonus_2pc.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise DiscError(
                    f"Set '{key}': 2pc bonus '{stat}' must be a positive number"
                )

        bonus_4pc = None
        raw_4pc = entry.get("bonus_4pc")
        if raw_4pc is not None:
            if not isinstance(raw_4pc, dict):
                raise DiscError(f"Set '{key}': 'bonus_4pc' must be an object")
            stat = raw_4pc.get("stat")
            per_stack = raw_4pc.get("per_stack")
            max_stacks = raw_4pc.get("max_stacks")
            if not isinstance(stat, str) or not stat.strip():
                raise DiscError(f"Set '{key}': 4pc bonus needs a 'stat' name")
            if isinstance(per_stack, bool) or not isinstance(per_stack, (int, float)) or per_stack <= 0:
                raise DiscError(f"Set '{key}': 4pc 'per_stack' must be positive")
            if isinstance(max_stacks, bool) or not isinstance(max_stacks, int) or max_stacks < 1:
                raise DiscError(f"Set '{key}': 4pc 'max_stacks' must be an integer >= 1")
            bonus_4pc = SetBonus4pc(
                stat=stat,
                per_stack=float(per_stack),
                max_stacks=max_stacks,
                auto=bool(raw_4pc.get("auto", False)),
                note=str(raw_4pc.get("note", "")),
            )

        dmg_bonuses: list[SetDmgBonus] = []
        raw_dmg = entry.get("bonus_4pc_dmg", [])
        if not isinstance(raw_dmg, list):
            raise DiscError(f"Set '{key}': 'bonus_4pc_dmg' must be a list")
        for item in raw_dmg:
            if not isinstance(item, dict):
                raise DiscError(
                    f"Set '{key}': 'bonus_4pc_dmg' entries must be objects"
                )
            tag = item.get("skill_tag")
            value = item.get("value")
            if not isinstance(tag, str) or not tag.strip():
                raise DiscError(
                    f"Set '{key}': typed DMG bonus needs a 'skill_tag'"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise DiscError(
                    f"Set '{key}': typed DMG bonus 'value' must be positive"
                )
            dmg_bonuses.append(SetDmgBonus(
                skill_tag=tag,
                value=float(value),
                note=str(item.get("note", "")),
            ))

        squad_4pc = None
        raw_squad = entry.get("squad_4pc")
        if raw_squad is not None:
            if not isinstance(raw_squad, dict):
                raise DiscError(f"Set '{key}': 'squad_4pc' must be an object")
            value = raw_squad.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise DiscError(f"Set '{key}': squad 'value' must be positive")
            max_stacks = raw_squad.get("max_stacks", 1)
            if isinstance(max_stacks, bool) or not isinstance(max_stacks, int) or max_stacks < 1:
                raise DiscError(
                    f"Set '{key}': squad 'max_stacks' must be an integer >= 1"
                )
            kind = raw_squad.get("kind", "dmg_bonus")
            if kind not in SQUAD_4PC_KINDS:
                raise DiscError(
                    f"Set '{key}': squad 'kind' must be one of "
                    f"{list(SQUAD_4PC_KINDS)}, got {kind!r}"
                )
            squad_4pc = SquadBonus4pc(
                value=float(value),
                kind=kind,
                max_stacks=max_stacks,
                auto=bool(raw_squad.get("auto", False)),
                note=str(raw_squad.get("note", "")),
            )

        auto_4pc_dmg = entry.get("bonus_4pc_auto_dmg", 0.0)
        if isinstance(auto_4pc_dmg, bool) or not isinstance(auto_4pc_dmg, (int, float)) or auto_4pc_dmg < 0:
            raise DiscError(
                f"Set '{key}': 'bonus_4pc_auto_dmg' must be a non-negative number"
            )

        sets[key] = DiscSet(
            key=key,
            name=name,
            bonus_2pc={s: float(v) for s, v in bonus_2pc.items()},
            bonus_4pc=bonus_4pc,
            bonus_4pc_dmg=tuple(dmg_bonuses),
            auto_4pc_dmg=float(auto_4pc_dmg),
            squad_4pc=squad_4pc,
            notes=str(entry.get("notes", "")),
        )
    return sets


def set_bonus_stats(
    discs: list[Disc] | tuple[Disc, ...],
    sets: dict[str, DiscSet],
    set_stacks: dict[str, int] | None = None,
) -> dict[str, float]:
    """Compute the automatic stat bonuses granted by equipped disc sets.

    Rules (mirroring the game):

    - 2+ pieces of a set -> its 2-piece bonus applies unconditionally.
    - 4+ pieces -> additionally, its modeled 4-piece stacking bonus applies
      at ``set_stacks[key]`` active stacks (default 0 — 4-piece effects are
      combat-conditional and OFF unless the user says they're up).
    - 4 pieces always include the 2-piece bonus as well.

    Args:
        discs: The equipped discs (``disc_set=None`` pieces count nothing).
        sets: Registry from :func:`load_disc_sets`.
        set_stacks: set key -> active stacks for modeled 4pc effects.

    Raises:
        DiscError: unknown set key on a disc; stacks for a set without 4
            equipped pieces (or without a modeled 4pc); stacks out of range.
    """
    set_stacks = set_stacks or {}

    counts: dict[str, int] = {}
    for disc in discs:
        if disc.disc_set is None:
            continue
        if disc.disc_set not in sets:
            raise DiscError(
                f"Disc in slot {disc.slot} names unknown set "
                f"'{disc.disc_set}'; options: {sorted(sets)}"
            )
        counts[disc.disc_set] = counts.get(disc.disc_set, 0) + 1

    for key, stacks in set_stacks.items():
        if key not in sets:
            raise DiscError(f"Stacks given for unknown set '{key}'")
        bonus = sets[key].bonus_4pc
        if counts.get(key, 0) < 4 or bonus is None:
            raise DiscError(
                f"Stacks given for '{key}' but its 4-piece bonus is not "
                f"active (needs 4 equipped pieces and a modeled effect)"
            )
        if isinstance(stacks, bool) or not isinstance(stacks, int) or not 0 <= stacks <= bonus.max_stacks:
            raise DiscError(
                f"'{key}' stacks must be an integer 0..{bonus.max_stacks}, "
                f"got {stacks!r}"
            )

    bonuses: dict[str, float] = {}
    for key, count in counts.items():
        entry = sets[key]
        if count >= 2:
            for stat, value in entry.bonus_2pc.items():
                bonuses[stat] = bonuses.get(stat, 0.0) + value
        if count >= 4 and entry.bonus_4pc is not None:
            stacks = set_stacks.get(key, 0)
            if stacks:
                b = entry.bonus_4pc
                bonuses[b.stat] = bonuses.get(b.stat, 0.0) + b.per_stack * stacks
    return bonuses


def set_tagged_dmg_bonuses(
    discs: list[Disc] | tuple[Disc, ...],
    sets: dict[str, DiscSet],
    skill_tag: str | None,
    valid_tags: set[str] | frozenset[str] | None = None,
) -> dict[str, float]:
    """Skill-type-conditional DMG% granted by equipped 4-piece sets.

    Returns:
        set key -> total DMG% (fraction) for every set with >= 4 equipped
        pieces whose typed bonuses match the hit's ``skill_tag``.
        ``skill_tag=None`` (untyped hit) matches nothing.

    Raises:
        DiscError: unknown set key on a disc, or — when ``valid_tags`` is
            given — a set's typed bonus naming a tag that isn't in it
            (protects against data-file typos that would otherwise
            silently never apply).
    """
    counts: dict[str, int] = {}
    for disc in discs:
        if disc.disc_set is None:
            continue
        if disc.disc_set not in sets:
            raise DiscError(
                f"Disc in slot {disc.slot} names unknown set "
                f"'{disc.disc_set}'; options: {sorted(sets)}"
            )
        counts[disc.disc_set] = counts.get(disc.disc_set, 0) + 1

    bonuses: dict[str, float] = {}
    for key, count in counts.items():
        if count < 4:
            continue
        for bonus in sets[key].bonus_4pc_dmg:
            if valid_tags is not None and bonus.skill_tag not in valid_tags:
                raise DiscError(
                    f"Set '{key}': typed DMG bonus names unknown skill tag "
                    f"'{bonus.skill_tag}'; valid tags: {sorted(valid_tags)}"
                )
            if skill_tag is not None and bonus.skill_tag == skill_tag:
                bonuses[key] = bonuses.get(key, 0.0) + bonus.value
    return bonuses


@dataclass(frozen=True)
class Loadout:
    """A named, pre-validated set of discs from ``data/loadouts.json``."""

    name: str
    description: str
    discs: tuple[Disc, ...]


def load_loadouts(
    disc_data: DiscData, path: Path = LOADOUTS_FILE,
    user_discs: dict[str, Disc] | None = None,
    user_discs_path: Path = USER_DISCS_FILE,
) -> dict[str, Loadout]:
    """Load saved disc loadouts and validate every disc in them.

    Loadouts are user convenience data (recurring test sets). A disc entry
    is either an embedded disc (substats as TOTAL rolls) or a reference
    into the user disc inventory: ``{"disc_id": "d1"}``. A missing file is
    not an error — loadouts are optional — but a malformed file or an
    invalid disc is, so a stale loadout can't silently corrupt a
    calculation.

    Args:
        user_discs: Already-loaded inventory to resolve references against;
            ``None`` loads it from ``user_discs_path``.

    Raises:
        DiscError: on malformed JSON, invalid discs, unknown ``disc_id``
            references, or duplicate slots within a loadout.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise DiscError(f"Loadouts file is not valid JSON: {exc}") from None

    entries = raw.get("loadouts", {})
    if not isinstance(entries, dict):
        raise DiscError("'loadouts' must be an object of name -> loadout")

    if user_discs is None:
        user_discs = load_user_discs(disc_data, user_discs_path)

    loadouts: dict[str, Loadout] = {}
    for name, entry in entries.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("discs"), list):
            raise DiscError(f"Loadout '{name}' must be an object with a 'discs' list")

        discs: list[Disc] = []
        seen_slots: set[int] = set()
        for disc_raw in entry["discs"]:
            if not isinstance(disc_raw, dict):
                raise DiscError(f"Loadout '{name}': disc entries must be objects")
            if "disc_id" in disc_raw:
                disc_id = disc_raw["disc_id"]
                if disc_id not in user_discs:
                    raise DiscError(
                        f"Loadout '{name}': unknown disc_id '{disc_id}' "
                        f"(not in the user disc inventory)"
                    )
                disc = user_discs[disc_id]
            else:
                try:
                    disc = Disc(
                        slot=disc_raw["slot"],
                        main_stat=disc_raw["main_stat"],
                        substats=dict(disc_raw.get("substats", {})),
                        disc_set=disc_raw.get("set"),
                        element=disc_raw.get("element"),
                    )
                except KeyError as exc:
                    raise DiscError(
                        f"Loadout '{name}': disc is missing {exc}"
                    ) from None
                validate_disc(disc, disc_data)   # raises DiscError with details
            if disc.slot in seen_slots:
                raise DiscError(f"Loadout '{name}': duplicate slot {disc.slot}")
            seen_slots.add(disc.slot)
            discs.append(disc)

        loadouts[name] = Loadout(
            name=name,
            description=str(entry.get("description", "")),
            discs=tuple(discs),
        )
    return loadouts


# ---------------------------------------------------------------------------
# User disc inventory (data/user_discs.json) and loadout saving
#
# User data like loadouts: discs the user has entered, stored with substats
# as TOTAL rolls and keyed by a stable generated id ("d1", "d2", ...).
# Saved loadouts reference these ids ({"disc_id": "d1"}), so a disc lives in
# one place and editing it updates every loadout that uses it.
# ---------------------------------------------------------------------------

_USER_DISCS_META = {
    "description": ("User disc inventory: discs saved from the UI, referenced "
                    "by saved loadouts via their id. User data, not game data."),
    "conventions": ("Substats are stored as TOTAL rolls (in-game '+N' + 1). "
                    "Every disc is validated against discs.json on load. "
                    "Optional 'set' names a key in disc_sets.json."),
}


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON via a temp file + replace, so a crash can't corrupt data."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _disc_payload(disc: Disc) -> dict:
    """A disc's JSON form (loadouts/inventory convention: total rolls)."""
    payload: dict = {
        "slot": disc.slot,
        "main_stat": disc.main_stat,
        "substats": dict(disc.substats),
    }
    if disc.disc_set:
        payload["set"] = disc.disc_set
    if disc.element:
        payload["element"] = disc.element
    return payload


def _read_user_discs_raw(path: Path) -> dict:
    """The inventory file as raw JSON; a missing file is an empty inventory."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"_meta": dict(_USER_DISCS_META), "discs": {}}
    except json.JSONDecodeError as exc:
        raise DiscError(f"User discs file is not valid JSON: {exc}") from None
    if not isinstance(raw.get("discs", {}), dict):
        raise DiscError("'discs' must be an object of id -> disc")
    raw.setdefault("discs", {})
    return raw


def load_user_discs(
    disc_data: DiscData, path: Path = USER_DISCS_FILE
) -> dict[str, Disc]:
    """Load the user disc inventory, validating every disc.

    Returns:
        Mapping of disc id -> :class:`Disc`, in file order.

    Raises:
        DiscError: malformed file or any invalid disc (so a corrupt
            inventory can't silently poison a calculation).
    """
    raw = _read_user_discs_raw(path)
    discs: dict[str, Disc] = {}
    for disc_id, entry in raw["discs"].items():
        if not isinstance(entry, dict):
            raise DiscError(f"User disc '{disc_id}' must be an object")
        try:
            disc = Disc(
                slot=entry["slot"],
                main_stat=entry["main_stat"],
                substats=dict(entry.get("substats", {})),
                disc_set=entry.get("set"),
                element=entry.get("element"),
            )
        except KeyError as exc:
            raise DiscError(f"User disc '{disc_id}' is missing {exc}") from None
        validate_disc(disc, disc_data)
        discs[disc_id] = disc
    return discs


def save_user_disc(
    disc: Disc, disc_data: DiscData, path: Path = USER_DISCS_FILE
) -> tuple[str, bool]:
    """Add ``disc`` to the inventory (validated); dedupes identical discs.

    Returns:
        ``(disc_id, created)`` — ``created`` is False when an identical
        disc (same slot/main/substats/set) already existed, in which case
        its id is returned instead of storing a duplicate.

    Raises:
        DiscError: invalid disc, or a malformed/invalid inventory file.
    """
    validate_disc(disc, disc_data)
    raw = _read_user_discs_raw(path)
    existing = load_user_discs(disc_data, path) if raw["discs"] else {}
    for disc_id, other in existing.items():
        if other == disc:
            return disc_id, False

    numbers = [int(k[1:]) for k in raw["discs"]
               if k.startswith("d") and k[1:].isdigit()]
    disc_id = f"d{max(numbers, default=0) + 1}"
    raw["discs"][disc_id] = _disc_payload(disc)
    _atomic_write_json(path, raw)
    return disc_id, True


def delete_user_disc(
    disc_id: str, path: Path = USER_DISCS_FILE,
    loadouts_path: Path = LOADOUTS_FILE,
) -> None:
    """Remove a disc from the inventory.

    Raises:
        DiscError: unknown id, or the disc is referenced by a saved
            loadout (delete/overwrite the loadout first).
    """
    raw = _read_user_discs_raw(path)
    if disc_id not in raw["discs"]:
        raise DiscError(f"Unknown user disc id '{disc_id}'")

    try:
        loadouts_raw = json.loads(loadouts_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        loadouts_raw = {}
    except json.JSONDecodeError as exc:
        raise DiscError(f"Loadouts file is not valid JSON: {exc}") from None
    for name, entry in (loadouts_raw.get("loadouts") or {}).items():
        discs_list = entry.get("discs") if isinstance(entry, dict) else None
        for disc_raw in discs_list or []:
            if isinstance(disc_raw, dict) and disc_raw.get("disc_id") == disc_id:
                raise DiscError(
                    f"Disc '{disc_id}' is used by loadout '{name}'; "
                    f"delete or overwrite that loadout first"
                )

    del raw["discs"][disc_id]
    _atomic_write_json(path, raw)


def save_loadout(
    name: str, description: str, disc_ids: list[str], disc_data: DiscData,
    *, overwrite: bool = False, path: Path = LOADOUTS_FILE,
    user_discs_path: Path = USER_DISCS_FILE,
) -> None:
    """Save a loadout as references into the user disc inventory.

    Raises:
        DiscError: empty name, unknown disc ids, duplicate slots, a name
            collision without ``overwrite``, or a malformed loadouts file.
    """
    if not isinstance(name, str) or not name.strip():
        raise DiscError("Loadout name must be a non-empty string")
    name = name.strip()

    inventory = load_user_discs(disc_data, user_discs_path)
    seen_slots: set[int] = set()
    for disc_id in disc_ids:
        if disc_id not in inventory:
            raise DiscError(f"Unknown user disc id '{disc_id}'")
        slot = inventory[disc_id].slot
        if slot in seen_slots:
            raise DiscError(f"Duplicate slot {slot} in loadout '{name}'")
        seen_slots.add(slot)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raw = {"loadouts": {}}
    except json.JSONDecodeError as exc:
        raise DiscError(f"Loadouts file is not valid JSON: {exc}") from None
    if not isinstance(raw.get("loadouts", {}), dict):
        raise DiscError("'loadouts' must be an object of name -> loadout")
    raw.setdefault("loadouts", {})

    if name in raw["loadouts"] and not overwrite:
        raise DiscError(f"Loadout '{name}' already exists")

    raw["loadouts"][name] = {
        "description": str(description or ""),
        "discs": [{"disc_id": disc_id} for disc_id in disc_ids],
    }
    _atomic_write_json(path, raw)


def disc_stats(disc: Disc, data: DiscData) -> dict[str, float]:
    """Validate ``disc`` and return its total stat contributions.

    The main stat value comes from the slot table; each substat contributes
    ``roll count × per-roll value``. If the same stat name appears as both a
    percent main and a flat substat they are different names ("ATK%" vs
    "ATK"), so no collision is possible after validation.

    Returns:
        stat name -> total value (fractions for percent stats).

    Raises:
        DiscError: if the disc fails :func:`validate_disc`.
    """
    validate_disc(disc, data)

    stats: dict[str, float] = {
        disc.main_stat: data.main_stats[disc.slot][disc.main_stat]
    }
    for stat, rolls in disc.substats.items():
        stats[stat] = stats.get(stat, 0.0) + rolls * data.substat_rolls[stat]
    return stats
