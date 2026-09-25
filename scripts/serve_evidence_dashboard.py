#!/usr/bin/env python3
"""Local evidence-dashboard server (replaces dashboard step 5).

Serves a self-contained page and a JSON API that reads the strict evidence
tables.  Opening the page triggers a server sync + rebuild (one-click refresh).

  python scripts/serve_evidence_dashboard.py --port 8770
  # then open http://127.0.0.1:8770/

Endpoints
  GET  /                 dashboard.html
  GET  /api/state        nodes + comparisons + integrity + tree + meta
  POST /refresh          run: SKIP_SYNC=0 bash scripts/refresh_progress_tree.sh
                         (sync from MetaCentrum + rebuild the evidence tree)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PROGRESS = REPORTS / "progress"
STATE_LOCK = threading.Lock()
LAST = {"at": 0.0, "ok": None, "log": "", "running": False}
# Do not re-run the heavy sync+rebuild more often than this (seconds).
MIN_REFRESH_INTERVAL = float(os.environ.get("EVIDENCE_REFRESH_MIN_INTERVAL", "20"))
# Manifest-scoped git sync (same implementation the workbench uses).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import git_sync  # noqa: E402
DASHBOARD_ROOT = Path(os.environ.get("DASHBOARD_PROJECT_ROOT", "/Users/xiaoyuhe/自动发文章dashboard"))
METHOD_MANIFEST = ROOT / "closed_loop_method_manifest.txt"
DASHBOARD_MANIFEST = ROOT / "dashboard_manifest.txt"


def git_target(name: str) -> dict:
    """Return the manifest/remote/branch triple for a sync target.

    Only 'method' and 'dashboard' are valid; anything else raises, so an unknown
    target can never silently sync to the method branch.
    """
    name = str(name).strip().lower()
    if name == "dashboard":
        return dict(root=DASHBOARD_ROOT, manifest=DASHBOARD_MANIFEST,
                    remote=os.environ.get("DASHBOARD_GIT_REMOTE", "dashboard-app"),
                    branch=os.environ.get("DASHBOARD_GIT_BRANCH", "main"))
    if name == "method":
        return dict(root=ROOT, manifest=METHOD_MANIFEST,
                    remote=os.environ.get("GIT_SYNC_REMOTE", "github"),
                    branch=os.environ.get("GIT_SYNC_BRANCH", "current-experiments"))
    raise ValueError(f"unknown git sync target: {name!r} (expected method|dashboard)")


def _query_target(path: str) -> str:
    if "target=dashboard" in path:
        return "dashboard"
    if "target=method" in path:
        return "method"
    return "method"


def read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_text(path: Path):
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def integrity():
    txt = read_text(PROGRESS / "metric_integrity.md")
    d = {}
    for line in txt.splitlines():
        if line.startswith("|") and "|" in line[1:]:
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) == 2 and parts[0] not in ("field", "---"):
                d[parts[0]] = parts[1]
    return {"raw": txt, "fields": d,
            "status": "invalid" if "STATUS = invalid" in txt else ("valid" if "STATUS = valid" in txt else "unknown")}


def read_json(path: Path):
    try:
        import json as _json
        return _json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_jsonl(path: Path):
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                import json as _json
                rows.append(_json.loads(line))
    except Exception:
        pass
    return rows


def read_yaml_dir(path: Path):
    """Read every *.yaml in a directory into {stem: mapping}."""
    out = {}
    try:
        import yaml as _yaml
        for p in sorted(path.glob("*.yaml")):
            try:
                out[p.stem] = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception:
                out[p.stem] = {"_unreadable": p.name}
    except Exception:
        pass
    return out


def read_json_dir(path: Path):
    """Read a JSON registry directory.

    Supports both flat ``<dir>/*.json`` and the cohort-registry layout
    ``<dir>/<cohort_id>/record.json`` (keyed by cohort id).
    """
    import json as _json
    out = {}
    try:
        for p in sorted(path.glob("*.json")):
            try:
                out[p.stem] = _json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                out[p.stem] = {"_unreadable": p.name}
        for sub in sorted(path.iterdir()):
            if sub.is_dir():
                rec = sub / "record.json"
                if rec.exists():
                    try:
                        out[sub.name] = _json.loads(rec.read_text(encoding="utf-8"))
                    except Exception:
                        out[sub.name] = {"_unreadable": rec.name}
    except Exception:
        pass
    return out


def state():
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "integrity": integrity(),
        # Pipeline state machine: registered / submitted / running / synced /
        # gate_failed / gate_passed(=validated) / frozen / holdout_running /
        # holdout_confirmed.  Every transition is decided by a durable receipt,
        # never by a chart.  Paths below match the on-disk layout exactly:
        # reports/cohort_registry/<id>/record.json,
        # reports/cohorts/<id>.yaml, reports/holdout_receipts/<cohort>.json.
        "pipeline": read_json(REPORTS / "pipeline_state.json"),
        "gates": {
            "G0_H1": read_json(REPORTS / "gates" / "G0_H1.json"),
        },
        "sync": read_json(REPORTS / "sync_receipt.json"),
        "cohort_registry": read_json_dir(REPORTS / "cohort_registry"),
        "cohort_manifests": read_yaml_dir(REPORTS / "cohorts"),
        "holdout_receipts": read_json_dir(REPORTS / "holdout_receipts"),
        "frozen_selection": read_yaml_dir(REPORTS / "frozen_selection"),
        "nodes": read_csv(REPORTS / "evidence_nodes.csv"),
        "comparisons": read_csv(REPORTS / "evidence_index.csv"),
        "exclusions": read_csv(REPORTS / "evidence_exclusions.csv"),
        "pairs": read_csv(REPORTS / "evidence_pairs.csv"),
        "tree": read_text(PROGRESS / "progress_tree.md"),
        "last_refresh": LAST,
    }


def _run_refresh():
    """Background worker: sync from MetaCentrum + rebuild the evidence tree."""
    try:
        script = ROOT / "scripts" / "refresh_progress_tree.sh"
        log = ["$ SKIP_SYNC=0 bash scripts/refresh_progress_tree.sh"]
        if not script.exists():
            LAST.update(at=time.time(), ok=False, log="refresh_progress_tree.sh missing", running=False)
            return
        env = dict(os.environ, SKIP_SYNC="0")
        try:
            r = subprocess.run(["bash", str(script)], cwd=str(ROOT), capture_output=True,
                               text=True, timeout=1800, env=env)
            log.append((r.stdout or "")[-4000:])
            log.append((r.stderr or "")[-2000:])
            ok = r.returncode == 0
        except subprocess.TimeoutExpired:
            log.append("[timeout] refresh_progress_tree.sh")
            ok = False
        LAST.update(at=time.time(), ok=ok, log="\n".join(log), running=False)
    finally:
        try:
            STATE_LOCK.release()
        except Exception:
            pass


def start_refresh(force: bool = True):
    """Kick off a refresh in the background and return immediately.

    Running the sync+rebuild inline makes the browser fetch hold a long-lived
    connection and die with BrokenPipeError; instead we start a worker and let
    the page poll /api/state until LAST['running'] is false.
    """
    now = time.time()
    if not force and (now - LAST["at"]) < MIN_REFRESH_INTERVAL:
        return {"started": False, "running": bool(LAST.get("running")), "skipped": True,
                "log": f"skipped (last refresh {now - LAST['at']:.0f}s ago < {MIN_REFRESH_INTERVAL:.0f}s)"}
    if not STATE_LOCK.acquire(blocking=False):
        return {"started": False, "running": True, "log": "refresh already running"}
    LAST["running"] = True
    threading.Thread(target=_run_refresh, daemon=True).start()
    return {"started": True, "running": True, "log": "refresh started"}


HTML = (PROGRESS / "dashboard.html")

# Bilingual bridge injected into the served page so the Dashboard language
# switch (Chinese <-> English) reaches this cross-origin iframe.  The parent
# posts {type:"dashboard-i18n", lang} and the script rewrites only the rendered
# interface text; all data and behaviour stay untouched.  It also honours
# ?lang=en for the standalone "open in new window" case.
I18N_BRIDGE = r'''<script id="dashboard-i18n-bridge">
(function () {
  var DICT = {
    "Evidence Tree · 证明树": "Evidence Tree · Proof tree",
    "实验记录流程图 · Evidence Tree": "Experiment evidence flow · Evidence Tree",
    "frozen 已冻结": "frozen Frozen",
    "gate_failed 门失败": "gate_failed Gate failed",
    "holdout_confirmed holdout确认": "holdout_confirmed Holdout confirmed",
    "holdout_running holdout运行中": "holdout_running Holdout running",
    "registered 已注册": "registered Registered",
    "running 运行中": "running Running",
    "submitted 已提交": "submitted Submitted",
    "synced 已同步": "synced Synced",
    "validated 已验证": "validated Validated",
    "← 唯一实质缺口": "← Sole substantial gap",
    "↻ 一键刷新（同步服务器）": "↻ One-click refresh (sync server)",
    "√ 已证": "√ Proven",
    "≈ 等效": "≈ Equivalent",
    "▶ 进行中": "▶ In progress",
    "◻ 文献假设": "◻ Literature hypothesis",
    "⚠ 待钉": "⚠ To pin down",
    "✗ 已否": "✗ Disproven",
    "上次同步": "Last sync",
    "严格判定 (cohort · k=独立单位)": "Strict verdict (cohort · k=independent unit)",
    "严格证据：仅": "Strict evidence: only",
    "冻结": "frozen",
    "或双击": "or double-click",
    "指标": "Metric",
    "方向 (pooled · 描述性)": "Direction (pooled · descriptive)",
    "无法连接证据服务（端口 8770）。": "Cannot connect to the evidence service (port 8770).",
    "比较明细 · comparisons": "Comparison details · comparisons",
    "状态": "Status",
    "管线状态 · pipeline（状态机）": "Pipeline status · pipeline (state machine)",
    "自动同步：开": "Auto-sync: on",
    "自动同步：关": "Auto-sync: off",
    "自动同步：": "Auto-sync: ",
    "请在终端运行：": "Run in the terminal:",
    "配对覆盖": "Pair coverage",
    "门失败定位 · gate failures（可复现原因）": "Gate failure localization · gate failures (reproducible causes)",
    "（替代 Dashboard 第五步）": "(replaces Dashboard step 5)",
    "；配对 unit = graph_seed×noise_seed / target×seed；判据 CI_lo>0 且 ≥5%。": "; pairing unit = graph_seed×noise_seed / target×seed; criterion CI_lo>0 and >=5%.",
    "无失败门": "no failed gate",
    "· 无失败门": "· no failed gate",
    "open · 硬门未满足": "open · hard gate not met",
    "低于实用阈值": "Below practical threshold",
    "低功效不确定": "Low-power inconclusive",
    "刷新失败：": "Refresh failed: ",
    "同步中…": "Syncing…",
    "否定": "Negative",
    "唯一实质缺口": "Sole substantial gap",
    "实现◐": "Impl ◐",
    "实现✓": "Impl ✓",
    "实现✗": "Impl ✗",
    "已否": "Disproven",
    "已实现 · 未验证": "Implemented · unverified",
    "已证": "Proven",
    "待判定": "To be decided",
    "待核验": "To be verified",
    "待钉": "To pin down",
    "指标无效": "Metric invalid",
    "支持": "Support",
    "文献假设": "Literature hypothesis",
    "未验证": "Unverified",
    "检查器已实现": "Checker implemented",
    "正在同步服务器并重建证据…": "Syncing server and rebuilding evidence…",
    "注册": "Registered",
    "注册✓": "Registered ✓",
    "等效": "Equivalent",
    "等效(±5%)": "Equivalent (±5%)",
    "缺失": "Missing",
    "证据完整": "Evidence complete",
    "证据缺失": "Evidence missing",
    "证据部分": "Partial evidence",
    "超范围": "Out of scope",
    "进行中": "In progress",
    "进行中 · 待配对": "In progress · awaiting pairing",
    "配对无效": "Pairing invalid",
    "文献假设：": "Literature hypothesis: ",
    "比较明细": "Comparison details"
  };
  var PATTERNS = [
    [/^待钉 · 正向 k=(.+)$/, "To pin · forward k=$1", /^To pin · forward k=(.+)$/, "待钉 · 正向 k=$1"],
    [/^待钉 · 反向 k=(.+)$/, "To pin · reverse k=$1", /^To pin · reverse k=(.+)$/, "待钉 · 反向 k=$1"],
    [/^待钉 · 近零 k=(.+)$/, "To pin · near-zero k=$1", /^To pin · near-zero k=(.+)$/, "待钉 · 近零 k=$1"],
    [/^待钉 k=(.+)$/, "To pin down k=$1", /^To pin down k=(.+)$/, "待钉 k=$1"]
  ];
  var REV = {};
  for (var k in DICT) { if (!Object.prototype.hasOwnProperty.call(REV, DICT[k])) REV[DICT[k]] = k; }
  var ATTRS = ["placeholder", "title", "aria-label"];
  var lang = "zh";
  try { if (new URLSearchParams(window.location.search).get("lang") === "en") lang = "en"; } catch (e) {}
  var observer = null;

  function translate(value) {
    if (!value) return value;
    var map = lang === "en" ? DICT : REV;
    if (Object.prototype.hasOwnProperty.call(map, value)) return map[value];
    for (var i = 0; i < PATTERNS.length; i++) {
      var p = PATTERNS[i];
      var re = lang === "en" ? p[0] : p[2];
      if (re.test(value)) return value.replace(re, lang === "en" ? p[1] : p[3]);
    }
    return value;
  }
  function ancestorTag(node, names) {
    var el = node.nodeType === 1 ? node : node.parentNode;
    while (el && el.nodeType === 1) {
      if (names[el.tagName]) return true;
      el = el.parentNode;
    }
    return false;
  }
  function isHard(node) { return ancestorTag(node, { SCRIPT: 1, STYLE: 1 }) || hasSkip(node); }
  function hasSkip(node) {
    var el = node.nodeType === 1 ? node : node.parentNode;
    while (el && el.nodeType === 1) { if (el.hasAttribute && el.hasAttribute("data-i18n-skip")) return true; el = el.parentNode; }
    return false;
  }
  function isRaw(node) { return ancestorTag(node, { PRE: 1, TEXTAREA: 1 }); }
  function walkText(node) {
    if (isHard(node)) return;
    var raw = node.nodeValue || "";
    var core = raw.trim();
    if (!core) return;
    var out;
    if (isRaw(node)) {
      var map = lang === "en" ? DICT : REV;
      if (!Object.prototype.hasOwnProperty.call(map, core)) return;
      out = map[core];
    } else {
      out = translate(core);
    }
    if (out === core) return;
    var idx = raw.indexOf(core);
    node.nodeValue = raw.slice(0, idx) + out + raw.slice(idx + core.length);
  }
  function walkAttrs(el) {
    if (isHard(el) || isRaw(el)) return;
    for (var i = 0; i < ATTRS.length; i++) {
      var name = ATTRS[i];
      var value = el.getAttribute(name);
      if (!value) continue;
      var out = translate(value);
      if (out !== value) el.setAttribute(name, out);
    }
  }
  function walk(root) {
    if (root.nodeType === 3) { walkText(root); return; }
    if (root.nodeType !== 1) return;
    if (root.tagName === "SCRIPT" || root.tagName === "STYLE") return;
    if (root.hasAttribute && root.hasAttribute("data-i18n-skip")) return;
    walkAttrs(root);
    var child = root.firstChild;
    while (child) { var next = child.nextSibling; walk(child); child = next; }
  }
  function apply() {
    document.documentElement.lang = lang === "en" ? "en" : "zh-CN";
    walk(document.body);
    if (lang === "en") {
      if (!observer && window.MutationObserver) {
        observer = new MutationObserver(function (mutations) {
          for (var i = 0; i < mutations.length; i++) {
            var m = mutations[i];
            if (m.type === "childList") {
              m.addedNodes.forEach(function (node) { walk(node); });
            } else if (m.type === "characterData") {
              walkText(m.target);
            } else if (m.type === "attributes") {
              walkAttrs(m.target);
            }
          }
        });
        observer.observe(document.body, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ATTRS });
      }
    } else if (observer) {
      observer.disconnect();
      observer = null;
    }
  }
  window.addEventListener("message", function (event) {
    var data = event.data;
    if (data && data.type === "dashboard-i18n" && (data.lang === "en" || data.lang === "zh")) {
      if (data.lang !== lang) { lang = data.lang; apply(); }
    }
  });
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", apply);
  } else {
    apply();
  }
})();
</script>'''


def render_html() -> bytes:
    """Serve dashboard.html with the bilingual bridge injected before </body>."""
    try:
        text = HTML.read_text(encoding="utf-8")
    except OSError:
        return b"dashboard.html not built"
    if "dashboard-i18n-bridge" not in text:
        idx = text.lower().rfind("</body>")
        if idx == -1:
            text += I18N_BRIDGE
        else:
            text = text[:idx] + I18N_BRIDGE + text[idx:]
    return text.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="text/html; charset=utf-8"):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # the browser navigated away / timed out; not an error we must log
            pass

    def do_GET(self):
        if self.path in ("/", "/index.html") or self.path.startswith("/?"):
            if HTML.exists():
                self._send(200, render_html())
            else:
                self._send(404, b"dashboard.html not built", "text/plain")
        elif self.path.startswith("/api/state"):
            self._send(200, json.dumps(state(), ensure_ascii=False).encode(), "application/json")
        elif self.path.startswith("/api/git/preview"):
            try:
                info = git_sync.preview(**git_target(_query_target(self.path)))
                self._send(200, json.dumps(info, ensure_ascii=False).encode(), "application/json")
            except Exception as exc:  # pragma: no cover - defensive
                self._send(500, json.dumps({"error": str(exc)}).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path.startswith("/refresh"):
            force = "force=0" not in self.path
            self._send(200, json.dumps(start_refresh(force=force), ensure_ascii=False).encode(), "application/json")
            return
        if self.path.startswith("/api/git/sync"):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
            except ValueError:
                payload = {}
            target = str(payload.get("target") or _query_target(self.path))
            message = payload.get("message") or None
            try:
                code, info = git_sync.sync(**git_target(target), message=message,
                                           receipt_dir=REPORTS / "audit" / "git_sync",
                                           target=target)
                self._send(200 if code == 0 else 502,
                           json.dumps(info, ensure_ascii=False).encode(), "application/json")
            except Exception as exc:  # pragma: no cover - defensive
                self._send(500, json.dumps({"error": str(exc)}).encode(), "application/json")
            return
        self._send(404, b"not found", "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"evidence dashboard: http://127.0.0.1:{args.port}/  (Ctrl-C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
