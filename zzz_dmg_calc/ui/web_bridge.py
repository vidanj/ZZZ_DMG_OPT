"""In-browser backend for the static site (Pyodide).

Design and scope: see DOCS/web_hosting_plan.md.

When the page is served as static files (GitHub Pages) there is no HTTP
server: ``worker.js`` boots Pyodide inside a Web Worker, unpacks the
``zzz_dmg_calc`` package into the virtual filesystem, and routes the
page's requests here instead. :func:`handle` mirrors the route table of
:class:`~zzz_dmg_calc.ui.server.UIRequestHandler` — same paths, same
request/response JSON, same ValueError -> 400 error contract — so
``index.html`` works identically over either transport.

All heavy lifting is reused from :mod:`~zzz_dmg_calc.ui.server` (the
pure functions ``load_app_data`` / ``frontend_payload`` /
``run_calculation`` / ``run_optimization``) and the validated writers in
:mod:`~zzz_dmg_calc.discs`. The writers hit ``data/user_discs.json`` /
``data/loadouts.json`` exactly as they do locally — under Pyodide those
paths live in the in-memory filesystem, so :func:`seed` restores them
from the browser's localStorage before anything loads, and
:func:`user_files` hands their raw contents back after every write for
the page to mirror.

Everything crosses the JS boundary as plain ``str`` (JSON), never as
live object proxies. The worker is single-threaded, so unlike the
server there is no write lock.
"""

from __future__ import annotations

import json
from dataclasses import replace
from urllib.parse import unquote

from .server import (
    AppData, frontend_payload, load_app_data, loadouts_payload,
    run_calculation, run_optimization, user_discs_payload, _parse_disc,
)
from ..discs import (
    LOADOUTS_FILE, USER_DISCS_FILE, delete_loadout, delete_user_disc,
    load_loadouts, load_user_discs, save_loadout, save_user_disc,
)

#: The one AppData instance, created by :func:`init` after :func:`seed`.
_data: AppData | None = None


def seed(files_json: str) -> None:
    """Write the user's saved data into the (virtual) filesystem.

    ``files_json`` is ``{"user_discs": <raw file text> | null,
    "loadouts": ...}`` — the localStorage mirror the page keeps (see
    worker.js). Missing/null entries are simply not written; the loaders
    treat absent files as an empty inventory. Must run before
    :func:`init`.
    """
    files = json.loads(files_json) if files_json else {}
    for key, path in (("user_discs", USER_DISCS_FILE),
                      ("loadouts", LOADOUTS_FILE)):
        text = files.get(key)
        if text:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


def init() -> None:
    """Load every database (validating it) — the worker's last boot step."""
    global _data
    _data = load_app_data()


def user_files() -> str:
    """Raw contents of the two user-writable files, as JSON.

    The worker attaches this to every successful write response so the
    page can mirror the files to localStorage (the virtual filesystem
    vanishes when the tab closes).
    """
    payload = {}
    for key, path in (("user_discs", USER_DISCS_FILE),
                      ("loadouts", LOADOUTS_FILE)):
        payload[key] = (path.read_text(encoding="utf-8")
                        if path.exists() else None)
    return json.dumps(payload)


def _refresh_user_data() -> None:
    """Reload user-writable data after a write (mirrors the server's)."""
    global _data
    assert _data is not None
    user_discs = load_user_discs(_data.disc_data)
    _data = replace(
        _data,
        user_discs=user_discs,
        loadouts=load_loadouts(_data.disc_data, user_discs=user_discs,
                               valid_agents=set(_data.agents)),
    )


def _post_disc(body: dict) -> dict:
    """POST /discs — save one disc to the inventory (validated, deduped)."""
    disc = _parse_disc(body)
    disc_id, created = save_user_disc(disc, _data.disc_data)
    _refresh_user_data()
    return {
        "id": disc_id,
        "created": created,
        "user_discs": user_discs_payload(_data),
    }


class _Reply(Exception):
    """A non-200 response that is not a plain error (e.g. exists=true)."""

    def __init__(self, status: int, body: dict) -> None:
        super().__init__(body.get("error", ""))
        self.status = status
        self.body = body


def _post_loadout(body: dict) -> dict:
    """POST /loadouts — mirrors ``UIRequestHandler._post_loadout``."""
    name = str(body.get("name", "")).strip()
    overwrite = bool(body.get("overwrite"))
    agent = str(body.get("agent_key") or "").strip() or None
    if agent is not None and agent not in _data.agents:
        raise ValueError(f"Unknown agent '{agent}'")
    discs = [_parse_disc(entry) for entry in body.get("discs", [])]
    if not discs:
        raise ValueError("A loadout needs at least one equipped disc")
    if name in _data.loadouts and not overwrite:
        raise _Reply(400, {"error": f"Loadout '{name}' already exists",
                           "exists": True})
    disc_ids = [save_user_disc(d, _data.disc_data)[0] for d in discs]
    save_loadout(
        name, str(body.get("description", "")), disc_ids,
        _data.disc_data, agent=agent, overwrite=overwrite,
    )
    _refresh_user_data()
    return {
        "user_discs": user_discs_payload(_data),
        "loadouts": loadouts_payload(_data),
    }


def _delete(path: str) -> dict:
    """DELETE /discs/{id} or /loadouts/{name}."""
    if path.startswith("/discs/"):
        delete_user_disc(unquote(path[len("/discs/"):]))
        _refresh_user_data()
        return {"user_discs": user_discs_payload(_data)}
    if path.startswith("/loadouts/"):
        delete_loadout(unquote(path[len("/loadouts/"):]))
        _refresh_user_data()
        return {"loadouts": loadouts_payload(_data)}
    raise _Reply(404, {"error": f"Unknown path {path}"})


def handle(method: str, path: str, body_text: str | None) -> str:
    """Serve one request; returns JSON ``{"status": int, "body": {...}}``.

    The same contract the HTTP server exposes: 200 with the endpoint's
    payload, 400 with ``{"error": message}`` (message user-facing,
    verbatim from the validating layers), 404 for unknown paths.
    """
    try:
        if method == "GET":
            if path != "/data":
                raise _Reply(404, {"error": f"Unknown path {path}"})
            return json.dumps({"status": 200,
                               "body": frontend_payload(_data)})
        if method == "DELETE":
            return json.dumps({"status": 200, "body": _delete(path)})
        if method != "POST":
            raise _Reply(404, {"error": f"Unsupported method {method}"})
        try:
            body = json.loads(body_text) if body_text else None
        except ValueError:
            raise _Reply(400, {"error": "Request body is not valid JSON"})
        if not isinstance(body, dict):
            raise _Reply(400, {"error": "Request body must be a JSON object"})
        routes = {
            "/calculate": lambda b: run_calculation(_data, b),
            "/optimize": lambda b: run_optimization(_data, b),
            "/discs": _post_disc,
            "/loadouts": _post_loadout,
        }
        handler = routes.get(path)
        if handler is None:
            raise _Reply(404, {"error": f"Unknown path {path}"})
        return json.dumps({"status": 200, "body": handler(body)})
    except _Reply as reply:
        return json.dumps({"status": reply.status, "body": reply.body})
    except (ValueError, TypeError, KeyError) as exc:
        # Includes CalcError/DiscError/etc. — user-facing by design.
        return json.dumps({"status": 400, "body": {"error": str(exc)}})
