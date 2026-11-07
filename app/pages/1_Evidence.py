"""Evidence page: why text matters (ESCI analysis) and what the batch pipeline found, if it was run."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Evidence · AgentShelf", page_icon="📊", layout="wide")
st.title("AgentShelf Evidence")

st.header("Live demo results")
live = ROOT / "results" / "live_summary.json"
if not live.exists():
    st.info("Live results will appear here after the six-shelf Ollama evaluation is ready.")
else:
    summary = json.loads(live.read_text())
    shelves = summary["shelves"]
    averages = summary["averages"]
    lift = averages["paired_bootstrap"]["pick_rate"]
    faithfulness = ["Passed" if row["faithfulness_passed"] else "Did not pass" for row in shelves]
    categories = [row["category"] for row in shelves]

    st.caption(f"{summary['model']} · {len(shelves)} shelves · {shelves[0]['n_buyers']} buyer scenarios per shelf")
    pick, confidence, audit = st.columns(3)
    pick.metric("Average pick rate",
                f"{averages['before']['pick_rate']:.1%} → {averages['after']['pick_rate']:.1%}",
                f"{100 * lift['delta']:+.1f} pp")
    confidence.metric("95% paired bootstrap CI",
                      f"{100 * lift['ci_low']:+.1f} to {100 * lift['ci_high']:+.1f} pp")
    audit.metric("Rewrites that passed faithfulness", f"{faithfulness.count('Passed')} / {len(shelves)}")

    fig = go.Figure()
    fig.add_bar(x=categories, y=[row["before"]["pick_rate"] for row in shelves],
                name="Before", marker_color="#64748B")
    fig.add_bar(x=categories, y=[row["after"]["pick_rate"] for row in shelves],
                name="After", marker_color="#0F766E", customdata=faithfulness,
                hovertemplate="%{x}<br>Pick rate: %{y:.1%}<br>Faithfulness: %{customdata}<extra>After</extra>")
    fig.add_scatter(x=categories, y=[row["before"]["fair_share"] for row in shelves],
                    mode="lines+markers", line=dict(color="#C2410C", dash="dash"), name="Fair share")
    fig.update_layout(
        barmode="group", yaxis_title="Pick rate", yaxis_tickformat=".0%",
        legend=dict(orientation="h", y=1.12), margin=dict(l=0, r=0, t=45, b=0),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Fair share is the chance of a pick if all listings are equally likely. "
               "The lift and interval include all tested rewrites.")

st.divider()
st.header("Amazon ESCI relevance evidence")

f = ROOT / "results" / "relevance_analysis.json"
if not f.exists():
    st.info("Run `python scripts/12_relevance_analysis.py` first.")
else:
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
