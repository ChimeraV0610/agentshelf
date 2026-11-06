"""Project-wide configuration and path helpers."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    path = path or ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


CFG = load_config()


_MOCK_KEYS = {"results", "samples", "cache", "usage_log"}


def path(key: str) -> Path:
    """Absolute path for a key in CFG['paths']; parent dirs are created lazily.

    With AGENTSHELF_MOCK=1 the LLM-dependent outputs go to *_mock locations so
    offline plumbing runs never mix with real results.
    """
    rel = Path(CFG["paths"][key])
    if os.environ.get("AGENTSHELF_MOCK") == "1" and key in _MOCK_KEYS:
        if rel.suffix:
            rel = rel.with_name(rel.stem + "_mock" + rel.suffix)
        else:
            rel = rel.with_name(rel.name + "_mock")
    p = ROOT / rel
    (p.parent if p.suffix else p).mkdir(parents=True, exist_ok=True)
    return p
