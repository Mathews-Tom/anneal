"""SSE-based live dashboard server for experiment monitoring.

Provides a lightweight HTTP server with Server-Sent Events for real-time
experiment updates. The runner publishes events to a shared EventBus;
connected browsers receive them via EventSource and render live charts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import textwrap
from collections.abc import AsyncIterator
from typing import Any

from aiohttp import web
from aiohttp.client_exceptions import ClientConnectionResetError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------


class EventBus:
    """Publish-subscribe event bus for experiment updates."""

    MAX_SCORE_HISTORY = 20

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[tuple[str, dict[str, Any]]]] = []
        self._lock = asyncio.Lock()
        self._latest_status: dict[str, dict[str, Any]] = {}
        self._score_history: dict[str, list[float]] = {}

    def publish(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish an event to all subscribers."""
        target_id = data.get("target_id")
        if target_id is not None:
            status = self._latest_status.setdefault(str(target_id), {})
            status["target_id"] = target_id
            if event_type == "experiment_complete":
                status["score"] = data.get("score")
                status["baseline"] = data.get("baseline")
                status["outcome"] = data.get("outcome")
                status["hypothesis"] = data.get("hypothesis")
                status["cost"] = data.get("cost")
                status["duration"] = data.get("duration")
                status["experiment_count"] = status.get("experiment_count", 0) + 1
                score = data.get("score")
                if score is not None:
                    history = self._score_history.setdefault(str(target_id), [])
                    history.append(score)
                    if len(history) > self.MAX_SCORE_HISTORY:
                        history.pop(0)
            elif event_type == "state_change":
                status["state"] = data.get("state")
                status["reason"] = data.get("reason")
            elif event_type == "milestone":
                status["kept_count"] = data.get("kept_count")
                status["score"] = data.get("score")

        for queue in self._subscribers:
            try:
                queue.put_nowait((event_type, data))
            except asyncio.QueueFull:
                logger.warning("Subscriber queue full, dropping event: %s", event_type)

    async def subscribe(self) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield events as they arrive. Async generator."""
        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(maxsize=256)
        self._subscribers.append(queue)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self._subscribers.remove(queue)

    def get_status(self) -> dict[str, Any]:
        """Return current status snapshot for all known targets."""
        return {
            "targets": dict(self._latest_status),
            "score_history": dict(self._score_history),
        }


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
  <div class="chart-title">Score Trajectory (last 20 per target)</div>
  <canvas id="chart" width="900" height="260"></canvas>
</div>
<table>
<thead>
<tr><th>Target</th><th>Score</th><th>Baseline</th><th>Experiments</th><th>State</th><th>Last Outcome</th><th>Last Hypothesis</th></tr>
</thead>
<tbody id="targets"><tr><td colspan="7" class="no-data">Waiting for events...</td></tr></tbody>
</table>
<script>
(function(){
  const targets = {};
  const scoreHistory = {};
  let eventCount = 0;
  const MAX_HISTORY = 20;
  const COLORS = ['#58a6ff','#3fb950','#d29922','#f778ba','#bc8cff','#79c0ff','#56d364','#e3b341'];

  function colorFor(idx){ return COLORS[idx % COLORS.length]; }

  const es = new EventSource('/events');
  const connEl = document.getElementById('conn-status');
  const evtEl = document.getElementById('event-count');
  const updEl = document.getElementById('last-update');

  es.onopen = function(){ connEl.textContent = 'Connected'; connEl.style.color = '#3fb950'; };
  es.onerror = function(){ connEl.textContent = 'Disconnected'; connEl.style.color = '#f85149'; };

  fetch('/api/status').then(function(r){ return r.json(); }).then(function(status){
    const st = status.targets || {};
    const sh = status.score_history || {};
    Object.keys(st).forEach(function(tid){
      targets[tid] = st[tid];
    });
    Object.keys(sh).forEach(function(tid){
      scoreHistory[tid] = sh[tid].slice(-MAX_HISTORY);
    });
    if(Object.keys(targets).length > 0) render();
  });

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
    t.experiment_count = (t.experiment_count || 0) + 1;
    if(!scoreHistory[tid]) scoreHistory[tid] = [];
    scoreHistory[tid].push(d.score);
    if(scoreHistory[tid].length > MAX_HISTORY) scoreHistory[tid].shift();
    eventCount++; render();
  });

  es.addEventListener('state_change', function(e){
    const d = JSON.parse(e.data);
    const tid = d.target_id;
    if(!targets[tid]) targets[tid] = {};
    targets[tid].target_id = tid;
    targets[tid].state = d.state;
    targets[tid].reason = d.reason;
    eventCount++; render();
  });

  es.addEventListener('milestone', function(e){
    const d = JSON.parse(e.data);
    const tid = d.target_id;
    if(!targets[tid]) targets[tid] = {};
    targets[tid].target_id = tid;
    targets[tid].kept_count = d.kept_count;
    targets[tid].score = d.score;
    eventCount++; render();
  });

  function render(){
    evtEl.textContent = 'Events: ' + eventCount;
    updEl.textContent = 'Last update: ' + new Date().toLocaleTimeString();
    renderTable();
    renderChart();
  }

  function renderTable(){
    const tbody = document.getElementById('targets');
    const ids = Object.keys(targets).sort();
    if(ids.length === 0){ tbody.innerHTML = '<tr><td colspan="7" class="no-data">Waiting for events...</td></tr>'; return; }
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

    // grid
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

      // dot at latest
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
    """Lightweight HTTP server with SSE endpoint for live experiment updates."""

    def __init__(
        self,
        event_bus: EventBus,
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        self._event_bus = event_bus
        self._host = host
        self._port = port
        self._app = web.Application()
        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_get("/events", self._handle_events)
        self._app.router.add_get("/api/status", self._handle_status)
        self._app.router.add_post("/api/event", self._handle_post_event)
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    async def start(self) -> None:
        """Start the HTTP server."""
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        logger.info("Dashboard server started at http://%s:%d", self._host, self._port)

    async def stop(self) -> None:
        """Stop the HTTP server."""
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

        async for event_type, data in self._event_bus.subscribe():
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

        return response

    async def _handle_status(self, _request: web.Request) -> web.Response:
        status = self._event_bus.get_status()
        return web.json_response(status)

    async def _handle_post_event(self, request: web.Request) -> web.Response:
        """Receive events from external processes (e.g., anneal run in another terminal)."""
        body = await request.json()
        event_type = body.get("type", "experiment_complete")
        data = body.get("data", {})
        self._event_bus.publish(event_type, data)
        return web.json_response({"ok": True})


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get or create the global event bus."""
    global _event_bus  # noqa: PLW0603
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
