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

    st.subheader("How often did the AI pick our product?")
    st.caption("Same product, competitors and buyer requests before and after the rewrite.")
    pick, confidence, audit = st.columns(3)
    pick.metric("Average chance of being picked",
                f"{averages['before']['pick_rate']:.0%} → {averages['after']['pick_rate']:.0%}",
                f"{100 * lift['delta']:+.1f} pp", delta_color="off")
    confidence.metric("Estimated change (95% interval)",
                      f"{100 * lift['ci_low']:+.1f} to {100 * lift['ci_high']:+.1f} pp")
    audit.metric("Rewrites that passed the fact check", f"{faithfulness.count('Passed')} / {len(shelves)}")

    fig = go.Figure()
    for phase, color in (("before", "#64748B"), ("after", "#0F766E")):
        rates = [row[phase]["pick_rate"] for row in shelves]
        counts = [f"{round(rate * row['n_buyers'])}/{row['n_buyers']}" for rate, row in zip(rates, shelves)]
        fig.add_bar(
            x=rates, y=categories, orientation="h", name=phase.title(), marker_color=color,
            text=counts, textposition="outside", cliponaxis=False, customdata=faithfulness,
            hovertemplate="%{y}<br>Picked %{text} times (%{x:.0%})"
                          + ("<br>Fact check: %{customdata}" if phase == "after" else "")
                          + f"<extra>{phase.title()}</extra>",
        )

    fair_shares = [row["before"]["fair_share"] for row in shelves]
    if len(set(fair_shares)) == 1:
        fig.add_vline(x=fair_shares[0], line_color="#C2410C", line_dash="dash")
    else:
        fig.add_scatter(x=fair_shares, y=categories, mode="markers+lines",
                        line=dict(color="#C2410C", dash="dash"), name="Equal-chance baseline")
    fig.update_layout(
        barmode="group", height=465, xaxis_title="Share of buyers who picked our product",
        xaxis=dict(tickformat=".0%", range=[0, min(1.09, max(.35, max(
            row[phase]["pick_rate"] for row in shelves for phase in ("before", "after")
        ) + .09))]),
        yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.10),
        margin=dict(l=0, r=25, t=35, b=0),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("The dashed line shows the baseline if every product has an equal chance. "
               "The average and interval include all rewrites. pp = percentage points.")

st.divider()
st.header("Amazon ESCI relevance evidence")

f = ROOT / "results" / "relevance_analysis.json"
if not f.exists():
    st.info("Run `python scripts/12_relevance_analysis.py` first.")
else:
    ra = json.loads(f.read_text())
    st.subheader("Do listings that use more of the shopper’s words match better?")
    df = pd.DataFrame([{"query-term coverage": k, "share judged Exact": v["exact_rate"], "n": v["n"]} for k, v in ra["exact_rate_by_query_term_coverage"].items()])
    fig = px.bar(df, x="query-term coverage", y="share judged Exact",
                 labels={"query-term coverage": "Share of shopper’s search words found in the listing",
                         "share judged Exact": "Share rated an exact match"},
                 text=df["share judged Exact"].map(lambda v: f"{v:.0%}"))
    fig.update_layout(yaxis_tickformat=".0%", height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    coef = pd.DataFrame(sorted(ra["logit_coefficients_standardized"].items(), key=lambda x: -abs(x[1])), columns=["feature", "standardized logit coefficient"])
    st.caption(f"Amazon ESCI · {ra['n_pairs']:,} search–product pairs rated by people.")
    with st.expander("How we measured relevance"):
        st.info(ra["narrative"])
        st.dataframe(coef, hide_index=True, width="stretch")

ev = ROOT / "results" / "agent_eval.json"
if ev.exists():
    st.subheader("Batch agent evaluation (1,000 real queries)")
    e = json.loads(ev.read_text())
    for m, r in e["models"].items():
        wb = r["wrong_buy"]
        st.metric(f"{m}: agent picked a non-Exact product", f"{100*wb['wrong_buy_rate']:.0f}% (CI {100*wb['ci_low']:.0f}–{100*wb['ci_high']:.0f}%)")
