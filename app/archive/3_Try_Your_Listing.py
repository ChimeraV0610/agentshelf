"""Page 4 - live mode: paste your own listing, run the agent on it, rewrite, re-run."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import CFG  # noqa: E402
from src.live import (available_models, fields_from_rewrite, guess_department, load_resources,  # noqa: E402
                      rewrite_live, run_live, suggest_queries, user_fields)
from src.llm.client import get_client  # noqa: E402

st.set_page_config(page_title="Try your listing · AgentShelf", page_icon="🧪", layout="wide")
st.title("Try your own listing")
st.caption("Paste a listing. We rank it among 1.2M real Amazon products, let an AI shopping agent choose between it and real "
           "competitors on real shopper queries, then rewrite it faithfully and run again.")


@st.cache_resource(show_spinner="Loading indices (1.2M products) and query embeddings…")
def resources():
    return load_resources()


@st.cache_resource
def client():
    return get_client()


models = available_models()
if os.environ.get("AGENTSHELF_MOCK") == "1":
    models = ["mock"] + models
if not models:
    st.error("No model available. Start Ollama and pull a model (`ollama pull qwen2.5:7b`) or put an API key in .env.")
    st.stop()
with st.sidebar:
    model = st.selectbox("Agent model", models, help="Local Ollama models run fully offline.")
    rewriter = st.selectbox("Rewriter / audit model", models, index=0)
    n_cands = st.slider("Competitors per query", 4, 19, 9, help="Fewer = faster with local models")
    n_queries = st.slider("Suggested queries", 2, 8, 4)

with st.form("listing"):
    c1, c2 = st.columns([3, 1])
    title = c1.text_input("Title *", placeholder="e.g. Insulated Stainless Steel Water Bottle 32 oz with Straw Lid")
    brand = c2.text_input("Brand", "")
    bullets = st.text_area("Bullet points (one per line)", height=120)
    d1, d2 = st.columns([3, 1])
    description = d1.text_area("Description (optional)", height=80)
    color = d2.text_input("Color", "")
    typed = st.text_area("Queries shoppers would type (one per line, optional; leave empty to auto-suggest from real Amazon searches)", height=80)
    submitted = st.form_submit_button("Diagnose", type="primary")

if submitted:
    if not title.strip():
        st.error("Title is required.")
        st.stop()
    res = resources()
    f = user_fields(title, brand, color, bullets, description)
    dept = guess_department(f)
    qs = [{"query": q.strip(), "query_id": None} for q in typed.splitlines() if q.strip()]
    if not qs:
        with st.spinner("Finding real shopper queries that match this listing…"):
            qs = suggest_queries(res, f, k=n_queries)
    st.session_state.update({"fields": f, "queries": qs, "dept": dept, "live_before": None, "rewrite": None, "live_after": None})

if st.session_state.get("fields"):
    f, qs, dept = st.session_state["fields"], st.session_state["queries"], st.session_state["dept"]
    res = resources()
    st.markdown(f"**Department guess:** {dept} · **Queries used:** " + " · ".join(f"“{q['query']}”" for q in qs))

    def _render(results: list[dict], label: str):
        df = pd.DataFrame(results)
        picked = int(df["picked"].sum())
        top10 = int((df["rank_hybrid"].fillna(999) <= 10).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{label}: agent picked your listing", f"{picked} / {len(df)}")
        c2.metric("Found in top-10 of 1.2M products", f"{top10} / {len(df)}")
        c3.metric("Avg seconds per query", f"{df['seconds'].mean():.0f}")
        show = df[["query", "source", "rank_hybrid", "agent_label", "picked", "reason_code", "missing_attributes", "picked_instead"]].copy()
        show["rank_hybrid"] = show["rank_hybrid"].map(lambda v: ">100" if v is None or v != v else f"{int(v)}")
        show["missing_attributes"] = show["missing_attributes"].map(lambda m: ", ".join(m) if isinstance(m, list) else "")
        st.dataframe(show, hide_index=True, use_container_width=True)
        errs = df[df["error"].notna()]
        if len(errs):
            st.warning(f"{len(errs)} query call(s) failed: {errs['error'].unique().tolist()}")

    if st.session_state.get("live_before") is None:
        prog = st.progress(0.0, text="Running the agent…")
        log = st.empty()

        def cb(i, n, r):
            prog.progress(i / n, text=f"{i}/{n} · “{r['query']}” → {'PICKED' if r['picked'] else 'not picked'} ({r['seconds']}s)")

        st.session_state["live_before"] = run_live(res, client(), model if model != "mock" else "mock", f, qs, n_cands=n_cands, progress=cb)
        prog.empty()
    st.subheader("Before")
    _render(st.session_state["live_before"], "Before")

    st.subheader("Rewrite")
    if st.button("Rewrite faithfully and run again", type="primary"):
        with st.spinner("Rewriting and auditing…"):
            rw = rewrite_live(client(), f, dept, res.dept_profiles, rewriter if rewriter != "mock" else "mock", rewriter if rewriter != "mock" else "mock")
        st.session_state["rewrite"] = rw
        f2 = fields_from_rewrite(f, rw)
        prog = st.progress(0.0, text="Re-running the agent with the rewritten listing…")

        def cb2(i, n, r):
            prog.progress(i / n, text=f"{i}/{n} · “{r['query']}” → {'PICKED' if r['picked'] else 'not picked'} ({r['seconds']}s)")

        st.session_state["live_after"] = run_live(res, client(), model if model != "mock" else "mock", f2, qs, n_cands=n_cands, progress=cb2)
        prog.empty()
    rw = st.session_state.get("rewrite")
    if rw:
        l, r = st.columns(2)
        with l:
            st.markdown("**Original**")
            st.markdown(f"**{f['title']}**")
            for b in f["bullets_list"][:5]:
                st.markdown(f"- {b}")
        with r:
            st.markdown("**Rewritten** · audit " + ("PASSED ✅" if rw["passed"] else "FAILED ❌") + f" · rounds {rw['rounds']}")
            st.markdown(f"**{rw['title']}**")
            uns = set(rw["llm"].get("unsupported", []))
            for b in rw["bullets"]:
                st.markdown(f"- {'🔴' if b in uns else '🟢'} {b}")
            if rw["rule"]["issues"]:
                st.caption("Rule layer: " + "; ".join(rw["rule"]["issues"]))
            if rw.get("attributes_not_available"):
                st.info("Shoppers in this department ask for, but your listing does not state: " + ", ".join(rw["attributes_not_available"]))
        if st.session_state.get("live_after"):
            st.subheader("After")
            _render(st.session_state["live_after"], "After")
