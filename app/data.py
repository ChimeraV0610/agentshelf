"""Cached loaders for the Streamlit app. Reads results/ (real) or results_mock/ (offline)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def results_dir() -> tuple[Path, bool]:
    """(dir, is_mock). Prefer real results; fall back to mock outputs."""
    forced = os.environ.get("AGENTSHELF_MOCK") == "1"
    real, mock = ROOT / "results", ROOT / "results_mock"
    if not forced and (real / "agent_eval.json").exists():
        return real, False
    if (mock / "agent_eval.json").exists():
        return mock, True
    return real, False


@st.cache_data(show_spinner=False)
def load_json(name: str):
    d, _ = results_dir()
    f = d / name
    return json.loads(f.read_text()) if f.exists() else None


@st.cache_data(show_spinner=False)
def load_parquet(rel: str) -> pd.DataFrame | None:
    d, _ = results_dir()
    f = d / rel
    return pd.read_parquet(f) if f.exists() else None


@st.cache_data(show_spinner=False)
def load_csv(rel: str) -> pd.DataFrame | None:
    d, _ = results_dir()
    f = d / rel
    return pd.read_csv(f) if f.exists() else None


@st.cache_data(show_spinner=False)
def load_diagnostic(pid: str) -> dict | None:
    d, _ = results_dir()
    f = d / "diagnostics" / f"{pid}.json"
    return json.loads(f.read_text()) if f.exists() else None


def banner():
    d, is_mock = results_dir()
    if is_mock:
        st.warning("Showing OFFLINE MOCK results (random agent). Set API keys and run `make agents ...` for real numbers.")


def pct(x, digits=0):
    try:
        return f"{100*float(x):.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"
