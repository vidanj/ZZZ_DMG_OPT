"""Attribute Anomaly database loader (anomaly + Disorder + Vortex rules).

Loads and validates ``data/anomalies.json``: per-element anomaly base
multipliers/proc structure, Disorder conversion rules, and Vortex conversion
rules (the Wind variant of Disorder). Values were cross-checked against the
zenless-optimizer datamine (2026-07-03); in-game calibration status is
tracked in DOCS/sources.md Phase 5.

Model notes (see DOCS/anomaly_plan.md):

- An anomaly entry describes damage per hit/tick/proc (``mult × ATK``) and
  its proc structure (``hits`` over ``duration`` at ``interval``).
- ``supported=False`` entries are listed but any attempt to calculate with
  them raises, so unverified numbers can never silently enter a result.
- Disorder deals damage as the REPLACED anomaly's element in one burst,
  with the closed form ``base + time_mult × max(0, window − elapsed)``
  (elapsed = seconds since the replaced anomaly was applied).
- Vortex triggers instead of Disorder while Windswept is active: one burst
  dealt as the INFUSED (non-wind) element, same closed form with per-element
  ``mult``/``time_mult`` and a 30s window. Additive "Disorder DMG
  Multiplier" buffs (Velina's Windbite, Yuzuha M6) add to both bursts'
  base multiplier — the API layer passes them in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .enemies import ELEMENTS

#: Default location of the anomaly data file, relative to this package.
DATA_FILE = Path(__file__).parent / "data" / "anomalies.json"


class AnomalyError(ValueError):
    """Raised for invalid anomaly data or unsupported anomaly requests."""


@dataclass(frozen=True)
class Anomaly:
    """One element's attribute anomaly.

    Attributes:
        element: Attack attribute key ("physical", "fire", ...).
        name: In-game anomaly name (Assault, Burn, ...).
        supported: False while values are pending verification — using an
            unsupported anomaly in a calculation raises AnomalyError.
        mult: Damage per hit/tick/proc as a fraction of ATK (7.13 = 713%).
        hits: Maximum procs over the full duration (1 for one-shot).
        interval: Seconds between procs (None for one-shot anomalies).
        duration: Anomaly duration in seconds.
        debuff_note: Human-readable secondary debuff description.
    """

    element: str
    name: str
    supported: bool
    mult: float | None
    hits: int | None
    interval: float | None
    duration: float | None
    debuff_note: str = ""

    def require_supported(self) -> None:
        """Raise unless this anomaly has verified values."""
        if not self.supported:
            raise AnomalyError(
                f"{self.name} ({self.element}) has no verified damage values "
                f"yet — see data/anomalies.json"
            )


@dataclass(frozen=True)
class DisorderRule:
    """How a replaced anomaly converts into Disorder burst damage.

    Closed form (zenless-optimizer datamine, 2026-07-03):
    ``total_mult = base + time_mult × max(0, window − elapsed_seconds)``
    where ``elapsed_seconds`` is the time since the REPLACED anomaly was
    applied. The wind entry is Polarity Disorder (time_mult 0).
    """

    element: str
    base: float
    time_mult: float
    window: float


@dataclass(frozen=True)
class VortexRule:
    """How the infused element converts into Vortex burst damage.

    Same closed form as :class:`DisorderRule` (``mult`` plays the role of
    ``base``) with a 30s window; damage is dealt as the INFUSED element.
    Wind itself is never an infusion, so it has no rule.
    """

    element: str
    mult: float
    time_mult: float
    window: float


@dataclass(frozen=True)
class AnomalyData:
    """Validated view of ``data/anomalies.json``."""

    anomalies: dict[str, Anomaly]
    disorder: dict[str, DisorderRule]
    vortex: dict[str, VortexRule]


def _rule_number(rule: dict, key: str, where: str,
                 minimum: float = 0.0) -> float:
    """One validated non-negative number of a disorder/vortex rule."""
    value = rule.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
        raise AnomalyError(
            f"{where}: '{key}' must be a number >= {minimum}"
        )
    return float(value)


def load_anomalies(path: Path = DATA_FILE) -> AnomalyData:
    """Load and validate the anomaly database.

    Raises:
        AnomalyError: if the file is missing/malformed, an element is
            missing or unknown, or a supported entry has invalid numbers.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise AnomalyError(f"Anomaly data file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise AnomalyError(f"Anomaly data file is not valid JSON: {exc}") from None

    entries = raw.get("anomalies")
    if not isinstance(entries, dict):
        raise AnomalyError("'anomalies' must be an object of element -> anomaly")
    unknown = sorted(set(entries) - set(ELEMENTS))
    if unknown:
        raise AnomalyError(f"Unknown elements in 'anomalies': {unknown}")
    missing = [e for e in ELEMENTS if e not in entries]
    if missing:
        raise AnomalyError(f"'anomalies' is missing elements: {missing}")

    anomalies: dict[str, Anomaly] = {}
    for element, entry in entries.items():
        if not isinstance(entry, dict):
            raise AnomalyError(f"Anomaly '{element}' must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise AnomalyError(f"Anomaly '{element}' is missing a valid 'name'")
        supported = bool(entry.get("supported", False))

        mult = entry.get("mult")
        hits = entry.get("hits")
        interval = entry.get("interval")
        duration = entry.get("duration")
        if supported:
            if isinstance(mult, bool) or not isinstance(mult, (int, float)) or mult <= 0:
                raise AnomalyError(
                    f"Anomaly '{element}': 'mult' must be a positive number"
                )
            if isinstance(hits, bool) or not isinstance(hits, int) or hits < 1:
                raise AnomalyError(
                    f"Anomaly '{element}': 'hits' must be an integer >= 1"
                )
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
                raise AnomalyError(
                    f"Anomaly '{element}': 'duration' must be a positive number"
                )
            if hits > 1:
                if isinstance(interval, bool) or not isinstance(interval, (int, float)) or interval <= 0:
                    raise AnomalyError(
                        f"Anomaly '{element}': multi-hit anomalies need a "
                        f"positive 'interval'"
                    )
            mult = float(mult)
            duration = float(duration)
            interval = float(interval) if interval is not None else None

        anomalies[element] = Anomaly(
            element=element,
            name=name,
            supported=supported,
            mult=mult if supported else None,
            hits=hits if supported else None,
            interval=interval if supported else None,
            duration=duration if supported else None,
            debuff_note=str(entry.get("debuff_note", "")),
        )

    disorder_raw = raw.get("disorder")
    if not isinstance(disorder_raw, dict):
        raise AnomalyError("'disorder' must be an object of element -> rule")
    disorder: dict[str, DisorderRule] = {}
    for element, rule in disorder_raw.items():
        if element.startswith("_"):
            continue   # commentary keys like "_note"
        if element not in ELEMENTS:
            raise AnomalyError(f"Unknown element in 'disorder': '{element}'")
        if not isinstance(rule, dict):
            raise AnomalyError(f"Disorder rule '{element}' must be an object")
        where = f"Disorder rule '{element}'"
        disorder[element] = DisorderRule(
            element=element,
            base=_rule_number(rule, "base", where),
            time_mult=_rule_number(rule, "time_mult", where),
            window=_rule_number(rule, "window", where),
        )

    vortex_raw = raw.get("vortex")
    if not isinstance(vortex_raw, dict):
        raise AnomalyError("'vortex' must be an object of element -> rule")
    vortex: dict[str, VortexRule] = {}
    for element, rule in vortex_raw.items():
        if element.startswith("_"):
            continue   # commentary keys like "_note"
        if element not in ELEMENTS:
            raise AnomalyError(f"Unknown element in 'vortex': '{element}'")
        if element == "wind":
            raise AnomalyError(
                "'vortex' rules describe the INFUSED element — wind itself "
                "cannot be an infusion"
            )
        if not isinstance(rule, dict):
            raise AnomalyError(f"Vortex rule '{element}' must be an object")
        where = f"Vortex rule '{element}'"
        vortex[element] = VortexRule(
            element=element,
            mult=_rule_number(rule, "mult", where),
            time_mult=_rule_number(rule, "time_mult", where),
            window=_rule_number(rule, "window", where),
        )

    return AnomalyData(anomalies=anomalies, disorder=disorder, vortex=vortex)
