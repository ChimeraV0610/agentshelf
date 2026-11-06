"""Evidence page: why text matters (ESCI analysis) and what the batch pipeline found, if it was run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Evidence · AgentShelf", page_icon="📊", layout="wide")
st.title("Evidence from 600k human relevance judgements")

f = ROOT / "results" / "relevance_analysis.json"
if not f.exists():
    st.info("Run `python scripts/12_relevance_analysis.py` first.")
    st.stop()
ra = json.loads(f.read_text())
st.subheader("Relevance (getting onto the shelf) is about using the shopper's words")
df = pd.DataFrame([{"query-term coverage": k, "share judged Exact": v["exact_rate"], "n": v["n"]} for k, v in ra["exact_rate_by_query_term_coverage"].items()])
fig = px.bar(df, x="query-term coverage", y="share judged Exact", text=df["share judged Exact"].map(lambda v: f"{v:.0%}"))
fig.update_layout(yaxis_tickformat=".0%", height=350, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)
coef = pd.DataFrame(sorted(ra["logit_coefficients_standardized"].items(), key=lambda x: -abs(x[1])), columns=["feature", "standardized logit coefficient"])
c1, c2 = st.columns([1, 1])
c1.dataframe(coef, hide_index=True, use_container_width=True)
c2.markdown(f"**{ra['n_pairs']:,} US query-product pairs** from Amazon ESCI, human-labelled. "
            "Covering the shopper's own words is the only feature that moves the Exact label; specs, attributes and fluff barely matter for relevance.")
c2.info(ra["narrative"])
st.caption("Which product an agent BUYS among relevant ones has no public labels. That is what the live simulation on the main page measures.")

ev = ROOT / "results" / "agent_eval.json"
if ev.exists():
    st.subheader("Batch agent evaluation (1,000 real queries)")
    e = json.loads(ev.read_text())
    for m, r in e["models"].items():
        wb = r["wrong_buy"]
        st.metric(f"{m}: agent picked a non-Exact product", f"{100*wb['wrong_buy_rate']:.0f}% (CI {100*wb['ci_low']:.0f}–{100*wb['ci_high']:.0f}%)")
