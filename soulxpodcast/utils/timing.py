"""Lightweight timing and profiling utilities.

Provides a Timer context manager and a global TimingCollector that records
named durations. Designed to be low-dependency and safe for quick profiling
from the CLI.
"""
from __future__ import annotations

import time
import threading
from typing import Dict


class TimingCollector:
    def __init__(self):
        self.enabled = False
        self.lock = threading.Lock()
        self.data: Dict[str, list[float]] = {}

    def record(self, name: str, secs: float):
        if not self.enabled:
            return
        with self.lock:
            if name not in self.data:
                self.data[name] = []
            self.data[name].append(secs)

    def reset(self):
        with self.lock:
            self.data = {}

    def report(self, out=None):
        import sys
        if out is None:
            out = sys.stdout
        with self.lock:
            total = 0.0
            out.write("[timings]\n")
            for k, v in sorted(self.data.items()):
                s = sum(v)
                total += s
                cnt = len(v)
                avg = s / cnt if cnt else 0.0
                out.write(f"{k}: total={_format_seconds_compact(s)} count={cnt} avg={_format_seconds_compact(avg)} ({s:.3f}s/{avg:.3f}s)\n")
            out.write(f"total: {_format_seconds_compact(total)} ({total:.3f}s)\n")


TIMING_COLLECTOR = TimingCollector()


class Timer:
    def __init__(self, name: str, collector: TimingCollector = TIMING_COLLECTOR):
        self.name = name
        self.collector = collector

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        t1 = time.perf_counter()
        elapsed = t1 - self.t0
        try:
            self.collector.record(self.name, elapsed)
        except Exception:
            # timing should never raise
            pass


def _format_seconds_compact(seconds: float, precision: int = 6) -> str:
    """Return a compact human-readable duration string that omits larger
    zero-valued time units.

    Examples:
      20.760638 -> "20.760638s"
      0.123 -> "123ms"
      61.5 -> "1m 1.500000s"
    """
    if seconds is None:
        return "0s"
    # Work with floats; produce days/hours/minutes and remaining seconds
    secs = float(seconds)
    days = int(secs // 86400)
    secs -= days * 86400
    hours = int(secs // 3600)
    secs -= hours * 3600
    minutes = int(secs // 60)
    secs -= minutes * 60

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")

    # secs may be fractional
    if secs >= 1 or not parts:
        # show seconds with requested precision, trim trailing zeros
        sec_str = f"{secs:.{precision}f}".rstrip("0").rstrip(".")
        parts.append(f"{sec_str}s")
    else:
        # less than 1 second and we already showed larger units -> show ms
        ms = int(round(secs * 1000))
        parts.append(f"{ms}ms")

    return " ".join(parts)
