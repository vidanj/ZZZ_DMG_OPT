/* Web Worker (ES module): runs the Python damage calculator in the
 * browser. Pyodide >= 314 only supports module workers, so the page
 * creates this with `new Worker("worker.js", { type: "module" })`.
 *
 * Design and scope: see DOCS/web_hosting_plan.md.
 *
 * Used only when index.html is served as a static site (GitHub Pages) —
 * with the local Python server the page talks HTTP and this file is
 * never loaded. The worker boots Pyodide (CPython on WebAssembly),
 * unpacks zzz_dmg_calc.zip into the virtual filesystem, seeds the
 * user's saved inventory/loadouts from the page, and then answers RPC
 * messages by calling zzz_dmg_calc.ui.web_bridge.handle().
 *
 * Protocol (all bodies are JSON strings, never object proxies):
 *   page -> worker : { init: true, files: {user_discs, loadouts} } once,
 *                    then { id, method, path, body }
 *   worker -> page : { stage } progress lines while booting,
 *                    { ready: true } | { bootError } once,
 *                    then { id, status, body, files } per request —
 *                    `files` (raw user-file contents to mirror into
 *                    localStorage) only after a successful write.
 */

// Pyodide is self-hosted next to the site (extracted from the
// pyodide-core release by the Pages workflow): same-origin loads work
// on locked-down networks where cross-origin worker scripts are
// blocked, and the site keeps working if the CDN ever changes.
import { loadPyodide } from "./pyodide/pyodide.mjs";

const PYODIDE_BASE = new URL("pyodide/", import.meta.url).href;

let bridge = null;        // the web_bridge module, once booted
const queue = [];         // requests that arrived while booting
let resolveInit;
const initReceived = new Promise((resolve) => { resolveInit = resolve; });

self.onmessage = (ev) => {
  const msg = ev.data;
  if (msg && msg.init) { resolveInit(msg); return; }
  if (bridge === null) queue.push(msg);
  else handleRequest(msg);
};

function stage(text) { self.postMessage({ stage: text }); }

function handleRequest(msg) {
  let status = 500;
  let body;
  let files = null;
  try {
    const raw = bridge.handle(msg.method, msg.path,
                              msg.body === undefined ? null : msg.body);
    ({ status, body } = JSON.parse(raw));
    if (msg.method !== "GET" && status === 200) {
      files = JSON.parse(bridge.user_files());
    }
  } catch (exc) {
    body = { error: `Calculator crashed: ${exc}` };
  }
  self.postMessage({ id: msg.id, status, body, files });
}

async function boot() {
  stage("Downloading the Python runtime… (about 8 MB, cached after "
        + "the first visit)");
  const pyodide = await loadPyodide({ indexURL: PYODIDE_BASE });

  stage("Loading the damage calculator…");
  const resp = await fetch("zzz_dmg_calc.zip");
  if (!resp.ok) {
    throw new Error(`Could not fetch zzz_dmg_calc.zip (HTTP ${resp.status})`);
  }
  pyodide.unpackArchive(await resp.arrayBuffer(), "zip");

  stage("Restoring your saved data…");
  const init = await initReceived;
  pyodide.runPython("import sys; sys.path.insert(0, '.')");
  const mod = pyodide.pyimport("zzz_dmg_calc.ui.web_bridge");
  mod.seed(JSON.stringify(init.files || {}));
  mod.init();

  bridge = mod;
  self.postMessage({ ready: true });
  for (const msg of queue.splice(0)) handleRequest(msg);
}

boot().catch((exc) => { self.postMessage({ bootError: String(exc) }); });
