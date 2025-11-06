"""Prompt feature caching utilities.

Provides a simple in-memory cache for prompt WAV-derived features (waveform,
log-mel, flow mel, speaker embedding) keyed by absolute file path. Supports
optional serialization to disk.

This cache is used by the Dataset to avoid recomputing audio features across
multiple inferences.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import hashlib
from pathlib import Path
import json
import time
import logging


class PromptFeatureCache:
    """A small cache for prompt audio features.

    Stored value schema per-path:
        {
            "audio_16k": torch.Tensor,    # 1D float tensor at 16k
            "log_mel": torch.Tensor,      # [num_mels, T]
            "mel": torch.Tensor,          # [T, num_mels] (flow mel)
            "mel_len": int,
            "spk_emb": list or torch.Tensor,
        }
    """

    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}

    def _key(self, path: str) -> str:
        try:
            return str(Path(path).resolve())
        except Exception:
            return os.path.abspath(path)

    def get(self, path: str) -> Optional[Dict[str, Any]]:
        k = self._key(path)
        if k in self.cache:
            return self.cache.get(k)
        # try to load a persisted entry (validates mtime)
        entry = self.load_entry_if_exists(path)
        return entry

    def add(self, path: str, features: Dict[str, Any]):
        self.cache[self._key(path)] = features

    def clear(self):
        self.cache.clear()

    def save(self, dst: str):
        """Serialize cache to disk using torch.save.

        Note: this will try to convert torch tensors to CPU tensors to ensure
        portability.
        """
        serial = {}
        for k, v in self.cache.items():
            serial[k] = {}
            for fk, fv in v.items():
                if isinstance(fv, torch.Tensor):
                    serial[k][fk] = fv.cpu()
                else:
                    serial[k][fk] = fv
        torch.save(serial, dst)

    def load(self, src: str):
        obj = torch.load(src, map_location="cpu")
        self.cache.update(obj)

    # Single-entry persistence helpers
    def _entry_filename(self, path: str, cache_dir: Optional[str] = None) -> str:
        """Return a filename for persisting the entry.

        We prefer to use the WAV's basename so files are easy to match, but if
        a collision with a different absolute path occurs we append a short
        hash suffix to disambiguate.
        """
        key = self._key(path)
        p = Path(path)
        stem = p.stem
        if cache_dir is None:
            cache_dir = str(Path.cwd() / ".prompt_cache")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # candidate using stem
        cand = Path(cache_dir) / (stem + ".pt")
        if not cand.exists():
            return str(cand)

        # if exists, check if same source path is already stored there
        try:
            obj = torch.load(str(cand), map_location="cpu")
            meta = obj.get("_meta") if isinstance(obj, dict) else None
            if meta and meta.get("src_path") == key:
                return str(cand)
        except Exception:
            # if we can't read it, we'll generate a hashed filename
            pass

        # disambiguate using short hash of absolute path
        short = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
        return str(Path(cache_dir) / (f"{stem}_{short}.pt"))

    def persist_entry(self, path: str, cache_dir: Optional[str] = None):
        """Persist a single cache entry to disk. No-op if entry missing."""
        k = self._key(path)
        if k not in self.cache:
            return
        serial = {}
        v = self.cache[k]
        for fk, fv in v.items():
            if isinstance(fv, torch.Tensor):
                serial[fk] = fv.cpu()
            else:
                serial[fk] = fv

        # include metadata for validation (src path and mtime)
        try:
            mtime = Path(path).stat().st_mtime
        except Exception:
            mtime = time.time()
        serial["_meta"] = {"src_path": k, "mtime": mtime}

        fn = self._entry_filename(path, cache_dir)
        torch.save(serial, fn)
        logging.getLogger(__name__).debug("Persisted prompt entry %s -> %s", path, fn)

    def load_entry_if_exists(self, path: str, cache_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Try to find a persisted entry for `path` and validate by mtime.

        Looks for files named `<basename>.pt` and `<basename>_*.pt` in the cache
        directory. If a matching entry is found whose stored `src_path` and
        `mtime` match the live file, the entry is loaded into memory and
        returned. Otherwise None is returned.
        """
        p = Path(path)
        stem = p.stem
        if cache_dir is None:
            cache_dir = str(Path.cwd() / ".prompt_cache")
        d = Path(cache_dir)
        if not d.exists():
            return None

        candidates = list(d.glob(f"{stem}*.pt"))
        if not candidates:
            return None

        for fn in candidates:
            try:
                obj = torch.load(str(fn), map_location="cpu")
                meta = obj.get("_meta") if isinstance(obj, dict) else None
                if not meta:
                    continue
                if meta.get("src_path") != self._key(path):
                    continue
                # check mtime
                try:
                    live_mtime = p.stat().st_mtime
                except Exception:
                    live_mtime = None
                if live_mtime is None or abs(meta.get("mtime", 0) - (live_mtime or 0)) < 1e-6:
                    # good match
                    # remove meta before storing
                    obj_no_meta = {k: v for k, v in obj.items() if k != "_meta"}
                    self.add(path, obj_no_meta)
                    return obj_no_meta
            except Exception:
                continue

        return None


# Module-level singleton used by dataloader and CLI
PROMPT_FEATURE_CACHE = PromptFeatureCache()
