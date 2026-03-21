"""File-based live dashboard server for experiment monitoring.

Reads experiment data from the .anneal/ directory structure on disk.
The runner writes experiments.jsonl and .anneal-status files; the dashboard
polls those files and pushes updates to connected browsers via SSE.

No coupling to the runner process — the dashboard can start before, during,
or after experiment runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import textwrap
import tomllib
from pathlib import Path
from typing import Any

from aiohttp import web
from aiohttp.client_exceptions import ClientConnectionResetError

logger = logging.getLogger(__name__)

MAX_SCORE_HISTORY = 50
POLL_INTERVAL_SECONDS = 2.0


# ---------------------------------------------------------------------------
# File-based state reader
# ---------------------------------------------------------------------------


class AnnealStateReader:
    """Reads experiment state from the .anneal/ directory structure.

    Expected layout:
        <root>/
            config.toml              # target registry
            targets/<id>/
                experiments.jsonl    # one ExperimentRecord per line
            worktrees/<id>/
                .anneal-status       # runner state snapshot
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._record_cursors: dict[str, int] = {}

    @property
    def root(self) -> Path:
        return self._root

    def discover_targets(self) -> dict[str, dict[str, Any]]:
        """Read config.toml and return target metadata keyed by target id."""
        config_path = self._root / "config.toml"
        if not config_path.exists():
            return {}
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        targets_raw = data.get("targets", {})
        result: dict[str, dict[str, Any]] = {}
        for tid, tdata in targets_raw.items():
            result[tid] = {
                "target_id": tid,
                "eval_mode": tdata.get("eval_mode", "unknown"),
                "baseline": tdata.get("baseline_score", 0.0),
                "baseline_raw_scores": tdata.get("baseline_raw_scores"),
                "git_branch": tdata.get("git_branch", ""),
                "artifact_paths": tdata.get("artifact_paths", []),
                "knowledge_path": tdata.get("knowledge_path", ""),
                "worktree_path": tdata.get("worktree_path", ""),
            }
        return result

    def read_experiments(self, target_id: str) -> list[dict[str, Any]]:
        """Read all experiment records for a target from experiments.jsonl."""
        jsonl_path = self._root / "targets" / target_id / "experiments.jsonl"
        if not jsonl_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
        return records

    def read_new_experiments(self, target_id: str) -> list[dict[str, Any]]:
        """Read only experiments added since the last call for this target."""
        all_records = self.read_experiments(target_id)
        cursor = self._record_cursors.get(target_id, 0)
        new_records = all_records[cursor:]
        self._record_cursors[target_id] = len(all_records)
        return new_records

    def read_status(self, target_id: str) -> dict[str, Any] | None:
        """Read .anneal-status from the target's worktree directory."""
        status_path = self._root / "worktrees" / target_id / ".anneal-status"
        if not status_path.exists():
            return None
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def build_snapshot(self) -> dict[str, Any]:
        """Build a complete status snapshot from all file sources."""
        targets_meta = self.discover_targets()
        targets: dict[str, dict[str, Any]] = {}
        score_history: dict[str, list[float]] = {}

        for tid, meta in targets_meta.items():
            records = self.read_experiments(tid)
            # Update cursor to current position
            self._record_cursors[tid] = len(records)

            status = self.read_status(tid) or {}

            experiment_count = len(records)
            scores = [r["score"] for r in records if "score" in r]
            last_record = records[-1] if records else None

            # Per-target outcome distribution
            outcome_counts: dict[str, int] = {}
            for r in records:
                oc = r.get("outcome", "UNKNOWN")
                outcome_counts[oc] = outcome_counts.get(oc, 0) + 1
            kept_count = outcome_counts.get("KEPT", 0)
            kept_rate = kept_count / experiment_count if experiment_count > 0 else 0.0
            durations = [r["duration_seconds"] for r in records if "duration_seconds" in r]
            avg_duration = sum(durations) / len(durations) if durations else 0.0
            best_score = max(scores) if scores else meta.get("baseline", 0.0)

            target_info: dict[str, Any] = {
                "target_id": tid,
                "baseline": meta.get("baseline", 0.0),
                "eval_mode": meta.get("eval_mode", "unknown"),
                "git_branch": meta.get("git_branch", ""),
                "experiment_count": experiment_count,
                "score": last_record["score"] if last_record else meta.get("baseline", 0.0),
                "outcome": last_record.get("outcome") if last_record else None,
                "hypothesis": last_record.get("hypothesis") if last_record else None,
                "cost": last_record.get("cost_usd", 0.0) if last_record else 0.0,
                "total_cost": sum(r.get("cost_usd", 0.0) for r in records),
                "state": status.get("state", "UNKNOWN"),
                "outcome_counts": outcome_counts,
                "kept_rate": kept_rate,
                "avg_duration": avg_duration,
                "best_score": best_score,
            }
            targets[tid] = target_info
            score_history[tid] = scores[-MAX_SCORE_HISTORY:]

        return {
            "targets": targets,
            "score_history": score_history,
        }


# ---------------------------------------------------------------------------
# SSE publisher backed by file polling
# ---------------------------------------------------------------------------


class FilePollingBus:
    """Polls .anneal/ files and publishes new experiments as SSE events."""

    def __init__(self, reader: AnnealStateReader) -> None:
        self._reader = reader
        self._subscribers: list[asyncio.Queue[tuple[str, dict[str, Any]]]] = []
        self._polling = False

    def publish(self, event_type: str, data: dict[str, Any]) -> None:
        for queue in self._subscribers:
            try:
                queue.put_nowait((event_type, data))
            except asyncio.QueueFull:
                logger.warning("Subscriber queue full, dropping event")

    async def subscribe(self) -> asyncio.Queue[tuple[str, dict[str, Any]]]:
        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(maxsize=256)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[tuple[str, dict[str, Any]]]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    async def poll_loop(self) -> None:
        """Continuously poll for new experiment records and publish as events."""
        self._polling = True
        # Initialize cursors with existing data
        self._reader.build_snapshot()

        while self._polling:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            targets_meta = self._reader.discover_targets()

            for tid in targets_meta:
                new_records = self._reader.read_new_experiments(tid)
                for record in new_records:
                    self.publish("experiment_complete", {
                        "target_id": tid,
                        "score": record.get("score", 0.0),
                        "baseline": record.get("baseline_score", 0.0),
                        "outcome": record.get("outcome", "UNKNOWN"),
                        "hypothesis": (record.get("hypothesis") or "")[:200],
                        "cost": record.get("cost_usd", 0.0),
                        "duration": record.get("duration_seconds", 0.0),
                    })

    def stop(self) -> None:
        self._polling = False


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = textwrap.dedent("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Anneal Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;padding:24px}
h1{font-size:1.4rem;margin-bottom:16px;color:#58a6ff}
.status-bar{display:flex;gap:16px;margin-bottom:20px;font-size:0.85rem;color:#8b949e}
table{width:100%;border-collapse:collapse;margin-bottom:24px}
th{text-align:left;padding:8px 12px;border-bottom:2px solid #21262d;color:#8b949e;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em}
td{padding:8px 12px;border-bottom:1px solid #21262d;font-size:0.9rem}
tr:hover{background:#161b22}
.outcome-KEPT{color:#3fb950}
.outcome-DISCARDED{color:#f85149}
.outcome-BLOCKED{color:#d29922}
.outcome-KILLED{color:#f85149;font-weight:600}
.outcome-CRASHED{color:#f85149;font-weight:600}
.state-RUNNING{color:#3fb950}
.state-BLOCKED{color:#d29922}
.state-PAUSED{color:#d29922}
.state-KILLED{color:#f85149}
.state-HALTED{color:#8b949e}
canvas{background:#161b22;border-radius:6px;display:block;margin-bottom:16px}
.chart-section{margin-bottom:24px}
.chart-title{font-size:0.9rem;color:#8b949e;margin-bottom:8px}
.hyp{max-width:320px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.no-data{color:#484f58;font-style:italic;padding:24px;text-align:center}
.cost{color:#8b949e;font-size:0.85rem}
.target-cards{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:24px}
.target-card{background:#161b22;border:1px solid #21262d;border-radius:6px;padding:16px;flex:1;min-width:320px;max-width:480px}
.target-card h3{font-size:0.9rem;color:#58a6ff;margin-bottom:12px}
.outcome-bar{height:16px;border-radius:3px;overflow:hidden;display:flex;margin-bottom:8px;background:#21262d}
.outcome-bar div{height:100%;transition:width 0.3s}
.card-stats{display:grid;grid-template-columns:1fr 1fr;gap:6px 16px;font-size:0.82rem}
.card-stats dt{color:#8b949e}
.card-stats dd{color:#c9d1d9;text-align:right}
.card-stats .kept-rate{color:#3fb950;font-weight:600}
.card-stats .best-score{color:#58a6ff;font-weight:600}
.outcome-legend{display:flex;gap:10px;font-size:0.75rem;color:#8b949e;margin-bottom:8px}
.outcome-legend span::before{content:'';display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:4px;vertical-align:middle}
.legend-kept::before{background:#3fb950 !important}
.legend-discarded::before{background:#f85149 !important}
.legend-blocked::before{background:#d29922 !important}
.legend-crashed::before{background:#f85149 !important}
</style>
</head>
<body>
<h1>Anneal Dashboard</h1>
<div class="status-bar">
  <span id="conn-status">Connecting...</span>
  <span id="event-count">Events: 0</span>
  <span id="last-update">Last update: --</span>
</div>
<div class="chart-section">
  <div class="chart-title">Score Trajectory (last 50 per target)</div>
  <canvas id="chart" width="900" height="260"></canvas>
</div>
<div id="target-cards" class="target-cards"></div>
<table>
<thead>
<tr><th>Target</th><th>Score</th><th>Baseline</th><th>Experiments</th><th>Cost</th><th>State</th><th>Last Outcome</th><th>Last Hypothesis</th></tr>
</thead>
<tbody id="targets"><tr><td colspan="8" class="no-data">Loading...</td></tr></tbody>
</table>
<script>
(function(){
  const targets = {};
  const scoreHistory = {};
  let eventCount = 0;
  const MAX_HISTORY = 50;
  const COLORS = ['#58a6ff','#3fb950','#d29922','#f778ba','#bc8cff','#79c0ff','#56d364','#e3b341'];

  function colorFor(idx){ return COLORS[idx % COLORS.length]; }

  const connEl = document.getElementById('conn-status');
  const evtEl = document.getElementById('event-count');
  const updEl = document.getElementById('last-update');

  // Load initial snapshot from files
  fetch('/api/status').then(function(r){ return r.json(); }).then(function(status){
    const st = status.targets || {};
    const sh = status.score_history || {};
    Object.keys(st).forEach(function(tid){
      targets[tid] = st[tid];
    });
    Object.keys(sh).forEach(function(tid){
      scoreHistory[tid] = sh[tid].slice(-MAX_HISTORY);
    });
    render();
  });

  // SSE for live updates from file polling
  const es = new EventSource('/events');
  es.onopen = function(){ connEl.textContent = 'Connected'; connEl.style.color = '#3fb950'; };
  es.onerror = function(){ connEl.textContent = 'Reconnecting...'; connEl.style.color = '#d29922'; };

  es.addEventListener('experiment_complete', function(e){
    const d = JSON.parse(e.data);
    const tid = d.target_id;
    if(!targets[tid]) targets[tid] = {};
    const t = targets[tid];
    t.target_id = tid;
    t.score = d.score;
    t.baseline = d.baseline;
    t.outcome = d.outcome;
    t.hypothesis = d.hypothesis;
    t.cost = d.cost;
    t.experiment_count = (t.experiment_count || 0) + 1;
    t.total_cost = (t.total_cost || 0) + (d.cost || 0);
    if(!scoreHistory[tid]) scoreHistory[tid] = [];
    scoreHistory[tid].push(d.score);
    if(scoreHistory[tid].length > MAX_HISTORY) scoreHistory[tid].shift();
    eventCount++; render();
  });

  es.addEventListener('status_refresh', function(e){
    const status = JSON.parse(e.data);
    const st = status.targets || {};
    const sh = status.score_history || {};
    Object.keys(st).forEach(function(tid){
      targets[tid] = st[tid];
    });
    Object.keys(sh).forEach(function(tid){
      scoreHistory[tid] = sh[tid].slice(-MAX_HISTORY);
    });
    render();
  });

  function render(){
    updEl.textContent = 'Last update: ' + new Date().toLocaleTimeString();
    evtEl.textContent = 'Events: ' + eventCount;
    renderCards();
    renderTable();
    renderChart();
  }

  function renderCards(){
    var container = document.getElementById('target-cards');
    var ids = Object.keys(targets).sort();
    if(ids.length === 0){ container.innerHTML = ''; return; }
    var OCOLOR = {KEPT:'#3fb950',DISCARDED:'#f85149',BLOCKED:'#d29922',CRASHED:'#f85149'};
    var html = '';
    ids.forEach(function(tid){
      var t = targets[tid];
      var oc = t.outcome_counts || {};
      var total = t.experiment_count || 0;
      var barHtml = '';
      ['KEPT','DISCARDED','BLOCKED','CRASHED'].forEach(function(o){
        var c = oc[o] || 0;
        if(c > 0 && total > 0){
          var pct = (c / total * 100).toFixed(1);
          barHtml += '<div style="width:' + pct + '%;background:' + OCOLOR[o] + '" title="' + o + ': ' + c + ' (' + pct + '%)"></div>';
        }
      });
      // Handle unknown outcomes
      var known = (oc['KEPT']||0) + (oc['DISCARDED']||0) + (oc['BLOCKED']||0) + (oc['CRASHED']||0);
      var other = total - known;
      if(other > 0 && total > 0){
        barHtml += '<div style="width:' + (other/total*100).toFixed(1) + '%;background:#484f58" title="OTHER: ' + other + '"></div>';
      }
      var keptRate = t.kept_rate != null ? (t.kept_rate * 100).toFixed(1) + '%' : '--';
      var bestScore = t.best_score != null ? Number(t.best_score).toFixed(4) : '--';
      var baseline = t.baseline != null ? Number(t.baseline).toFixed(4) : '--';
      var avgDur = t.avg_duration != null ? Number(t.avg_duration).toFixed(1) + 's' : '--';
      var totalCost = t.total_cost != null ? '$' + Number(t.total_cost).toFixed(4) : '--';
      html += '<div class="target-card">'
        + '<h3>' + esc(tid) + '</h3>'
        + '<div class="outcome-legend">'
        + '<span class="legend-kept">Kept</span>'
        + '<span class="legend-discarded">Discarded</span>'
        + '<span class="legend-blocked">Blocked</span>'
        + '<span class="legend-crashed">Crashed</span>'
        + '</div>'
        + '<div class="outcome-bar">' + barHtml + '</div>'
        + '<dl class="card-stats">'
        + '<dt>Kept rate</dt><dd class="kept-rate">' + keptRate + '</dd>'
        + '<dt>Best score</dt><dd class="best-score">' + bestScore + '</dd>'
        + '<dt>Baseline</dt><dd>' + baseline + '</dd>'
        + '<dt>Total cost</dt><dd>' + totalCost + '</dd>'
        + '<dt>Avg duration</dt><dd>' + avgDur + '</dd>'
        + '<dt>Experiments</dt><dd>' + total + '</dd>'
        + '</dl></div>';
    });
    container.innerHTML = html;
  }

  function renderTable(){
    const tbody = document.getElementById('targets');
    const ids = Object.keys(targets).sort();
    if(ids.length === 0){ tbody.innerHTML = '<tr><td colspan="8" class="no-data">No targets found</td></tr>'; return; }
    let html = '';
    ids.forEach(function(tid){
      const t = targets[tid];
      const stateClass = t.state ? 'state-' + t.state : '';
      const outcomeClass = t.outcome ? 'outcome-' + t.outcome : '';
      html += '<tr>'
        + '<td>' + esc(tid) + '</td>'
        + '<td>' + fmt(t.score) + '</td>'
        + '<td>' + fmt(t.baseline) + '</td>'
        + '<td>' + (t.experiment_count || 0) + '</td>'
        + '<td class="cost">$' + (t.total_cost || 0).toFixed(4) + '</td>'
        + '<td class="' + stateClass + '">' + esc(t.state || '--') + '</td>'
        + '<td class="' + outcomeClass + '">' + esc(t.outcome || '--') + '</td>'
        + '<td class="hyp" title="' + esc(t.hypothesis || '') + '">' + esc(t.hypothesis || '--') + '</td>'
        + '</tr>';
    });
    tbody.innerHTML = html;
  }

  function renderChart(){
    const canvas = document.getElementById('chart');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = {t:20, r:20, b:30, l:50};
    ctx.clearRect(0, 0, W, H);
    const ids = Object.keys(scoreHistory).sort();
    if(ids.length === 0) return;

    let allMin = Infinity, allMax = -Infinity;
    ids.forEach(function(tid){
      scoreHistory[tid].forEach(function(v){ if(v < allMin) allMin = v; if(v > allMax) allMax = v; });
    });
    if(allMin === allMax){ allMin -= 1; allMax += 1; }
    const range = allMax - allMin;
    allMin -= range * 0.05;
    allMax += range * 0.05;

    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;

    ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
    for(let i = 0; i <= 4; i++){
      const y = pad.t + plotH * (i / 4);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      ctx.fillStyle = '#484f58'; ctx.font = '11px monospace'; ctx.textAlign = 'right';
      const val = allMax - (allMax - allMin) * (i / 4);
      ctx.fillText(val.toFixed(3), pad.l - 6, y + 4);
    }

    ids.forEach(function(tid, idx){
      const pts = scoreHistory[tid];
      if(pts.length < 2) return;
      ctx.strokeStyle = colorFor(idx);
      ctx.lineWidth = 2;
      ctx.beginPath();
      pts.forEach(function(v, i){
        const x = pad.l + (i / (MAX_HISTORY - 1)) * plotW;
        const y = pad.t + plotH * (1 - (v - allMin) / (allMax - allMin));
        if(i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();

      const lastX = pad.l + ((pts.length - 1) / (MAX_HISTORY - 1)) * plotW;
      const lastY = pad.t + plotH * (1 - (pts[pts.length - 1] - allMin) / (allMax - allMin));
      ctx.beginPath(); ctx.arc(lastX, lastY, 3, 0, Math.PI * 2); ctx.fillStyle = colorFor(idx); ctx.fill();
      ctx.fillStyle = colorFor(idx); ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
      ctx.fillText(tid, lastX + 6, lastY + 4);
    });
  }

  function fmt(v){ return v != null ? Number(v).toFixed(4) : '--'; }
  function esc(s){ if(!s) return ''; const d = document.createElement('div'); d.textContent = String(s); return d.innerHTML; }
})();
</script>
</body>
</html>
""")


# ---------------------------------------------------------------------------
# Dashboard Server
# ---------------------------------------------------------------------------


class DashboardServer:
    """HTTP server that reads .anneal/ files and serves a live dashboard."""

    def __init__(
        self,
        anneal_root: Path,
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        self._reader = AnnealStateReader(anneal_root)
        self._bus = FilePollingBus(self._reader)
        self._host = host
        self._port = port
        self._app = web.Application()
        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_get("/events", self._handle_events)
        self._app.router.add_get("/api/status", self._handle_status)
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._poll_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        self._poll_task = asyncio.create_task(self._bus.poll_loop())
        logger.info(
            "Dashboard serving .anneal/ at %s — http://%s:%d",
            self._reader.root, self._host, self._port,
        )

    async def stop(self) -> None:
        self._bus.stop()
        if self._poll_task is not None:
            self._poll_task.cancel()
            self._poll_task = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
            logger.info("Dashboard server stopped")

    async def _handle_index(self, _request: web.Request) -> web.Response:
        return web.Response(text=DASHBOARD_HTML, content_type="text/html")

    async def _handle_events(self, request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        queue = await self._bus.subscribe()
        try:
            while True:
                event_type, data = await queue.get()
                payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                try:
                    await response.write(payload.encode())
                except (
                    ConnectionResetError,
                    ConnectionAbortedError,
                    BrokenPipeError,
                    ClientConnectionResetError,
                ):
                    break
        finally:
            self._bus.unsubscribe(queue)

        return response

    async def _handle_status(self, _request: web.Request) -> web.Response:
        snapshot = self._reader.build_snapshot()
        return web.json_response(snapshot)
