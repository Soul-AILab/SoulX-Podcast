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
                out.write(f"{k}: total={s:.3f}s count={cnt} avg={avg:.3f}s\n")
            out.write(f"total: {total:.3f}s\n")


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
