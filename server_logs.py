"""
In-memory log ring + SSE streaming + cache/status reporting.

All pipeline `print()` output is tee-ed into a bounded ring buffer that the
viewer can fetch (one-shot) or subscribe to (SSE). Lets us debug from the
browser without tailing /tmp/lego-server.log.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse


ROOT = Path(__file__).parent
GPT_CACHE_DIR = ROOT / "gpt_cache"
MESH_CACHE_DIR = ROOT / "test_meshes"
OUTPUT_DIR = ROOT / "output"
PHOTOS_DIR = ROOT / "test_photos"


# Bounded ring of {"ts": float, "msg": str, "level": "info"|"warn"|"error"}.
# 500 lines is enough for a full pipeline run; older lines drop off.
LOG_RING: deque[dict] = deque(maxlen=500)
_SUBSCRIBERS: list[asyncio.Queue] = []

# Lightweight run-stats — last completed run gets dropped in here.
LAST_RUN: dict = {"status": "idle"}


class _StdoutTee:
    """Replaces sys.stdout so every print() lands in LOG_RING + real stdout.

    Each newline-terminated chunk becomes one log entry. Errors get auto-
    classified by content sniff (line contains 'error', 'fail', 'traceback').
    """

    def __init__(self, real):
        self.real = real
        self._buf = ""

    def write(self, text: str) -> int:
        try:
            self.real.write(text)
        except Exception:
            pass
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                _publish(line)
        return len(text)

    def flush(self):
        try:
            self.real.flush()
        except Exception:
            pass

    def isatty(self):
        return False


_MAIN_LOOP: asyncio.AbstractEventLoop | None = None


def set_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Record the main asyncio loop so worker threads can post events
    back to subscribers' Queues safely via call_soon_threadsafe."""
    global _MAIN_LOOP
    _MAIN_LOOP = loop


def _publish(msg: str) -> None:
    low = msg.lower()
    if "traceback" in low or "error" in low or "fail" in low:
        level = "error"
    elif "warn" in low or "skip" in low or "retry" in low:
        level = "warn"
    else:
        level = "info"
    entry = {"ts": time.time(), "msg": msg, "level": level}
    LOG_RING.append(entry)
    # asyncio.Queue is NOT thread-safe. When the pipeline runs in a worker
    # thread (via run_in_threadpool), we MUST schedule the put_nowait on
    # the main event loop so the SSE coroutine can wake up promptly.
    if _MAIN_LOOP is not None and _SUBSCRIBERS:
        def _deliver():
            for q in list(_SUBSCRIBERS):
                try:
                    q.put_nowait(entry)
                except asyncio.QueueFull:
                    pass
        try:
            _MAIN_LOOP.call_soon_threadsafe(_deliver)
        except RuntimeError:
            # Loop closed during shutdown
            pass
    else:
        # Called from the main thread before any subscriber connected, or
        # before set_loop() was called — fall back to direct put.
        for q in list(_SUBSCRIBERS):
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                pass


def install() -> None:
    """Idempotent stdout/stderr tee installation."""
    if not isinstance(sys.stdout, _StdoutTee):
        sys.stdout = _StdoutTee(sys.stdout)
    if not isinstance(sys.stderr, _StdoutTee):
        sys.stderr = _StdoutTee(sys.stderr)


def log(msg: str) -> None:
    """Direct log without going through stdout (useful for async handlers)."""
    sys.stdout.write(msg + "\n")


def update_run(**kwargs) -> None:
    LAST_RUN.update(kwargs)
    LAST_RUN["updated_ts"] = time.time()


# ── Verbose pipeline-step logging ──────────────────────────────────────────

import contextlib


@contextlib.contextmanager
def stage(name: str, why: str = "", **extras):
    """Context manager that prints structured start/end markers with timing.

    Use:
        with stage("rembg", "Cut out the subject's background"):
            ...
    """
    extra = "  ".join(f"{k}={v}" for k, v in extras.items())
    head = f"▶ [{name}] {why}" if why else f"▶ [{name}]"
    if extra:
        head += f"   ({extra})"
    print(head)
    t0 = time.time()
    LAST_RUN["stage"] = name
    LAST_RUN["stage_start"] = t0
    try:
        yield
    except Exception as e:
        elapsed = time.time() - t0
        print(f"✗ [{name}] FAILED after {elapsed:.2f}s — {type(e).__name__}: {e}")
        raise
    else:
        elapsed = time.time() - t0
        print(f"✓ [{name}] done in {elapsed:.2f}s")


def step(msg: str) -> None:
    """One-line progress note inside a stage. Prefix with '·' to make it
    visually subordinate to the stage's start/end markers."""
    print(f"  · {msg}")


# ── FastAPI router ─────────────────────────────────────────────────────────

router = APIRouter()


@router.get("/api/logs")
def api_logs(limit: int = 200, level: str | None = None):
    items = list(LOG_RING)[-limit:]
    if level:
        items = [e for e in items if e["level"] == level]
    return {"count": len(items), "logs": items}


@router.delete("/api/logs")
def api_logs_clear():
    LOG_RING.clear()
    return {"cleared": True}


@router.get("/api/logs/stream")
async def api_logs_stream():
    """SSE: emit new log entries as they arrive. One subscriber per request."""
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    _SUBSCRIBERS.append(q)

    async def gen() -> Iterator[bytes]:
        # First flush the existing ring so the client sees recent context
        for entry in list(LOG_RING)[-50:]:
            yield f"data: {json.dumps(entry)}\n\n".encode()
        try:
            while True:
                try:
                    entry = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(entry)}\n\n".encode()
                except asyncio.TimeoutError:
                    # Heartbeat — keeps the connection alive through proxies
                    yield b": keepalive\n\n"
        finally:
            if q in _SUBSCRIBERS:
                _SUBSCRIBERS.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@router.get("/api/status")
def api_status():
    return {"run": LAST_RUN, "log_count": len(LOG_RING)}


def _dir_summary(d: Path) -> dict:
    if not d.exists():
        return {"exists": False, "count": 0, "size_mb": 0.0}
    files = [p for p in d.iterdir() if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    return {
        "exists": True,
        "count": len(files),
        "size_mb": round(total / (1024 * 1024), 2),
        "latest": sorted([p.name for p in files])[-5:],
    }


@router.get("/api/cache")
def api_cache():
    return {
        "gpt_cache": _dir_summary(GPT_CACHE_DIR),
        "meshes":    _dir_summary(MESH_CACHE_DIR),
        "outputs":   _dir_summary(OUTPUT_DIR),
        "photos":    _dir_summary(PHOTOS_DIR),
    }


def _cache_table() -> dict[str, Path]:
    return {
        "gpt": GPT_CACHE_DIR,
        "meshes": MESH_CACHE_DIR,
        "outputs": OUTPUT_DIR,
        "photos": PHOTOS_DIR,
    }


def _clear_cache_dir(d: Path) -> int:
    removed = 0
    if d.exists():
        for p in d.iterdir():
            if p.is_file():
                try:
                    p.unlink()
                    removed += 1
                except Exception:
                    pass
    return removed


@router.delete("/api/cache")
def api_cache_clear_all():
    removed = {
        kind: _clear_cache_dir(path)
        for kind, path in _cache_table().items()
    }
    return {"cleared": "all", "removed": removed, "total_removed": sum(removed.values())}


@router.delete("/api/cache/{kind}")
def api_cache_clear(kind: str):
    table = _cache_table()
    if kind not in table:
        raise HTTPException(400, f"unknown cache kind: {kind}")
    removed = _clear_cache_dir(table[kind])
    return {"cleared": kind, "removed": removed}
