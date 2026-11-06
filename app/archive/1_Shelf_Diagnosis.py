"""Page 2 - one product through the AI agent's eyes."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data import banner, load_diagnostic, load_parquet, pct

st.set_page_config(page_title="Shelf Diagnosis · AgentShelf", page_icon="🔎", layout="wide")
st.title("Shelf Diagnosis: how does the AI shopper see my product?")
banner()

summary = load_parquet("diagnostics_summary.parquet")
if summary is None or summary.empty:
    st.info("No diagnostics yet. Run `make eval-retrieval diagnose`.")
    st.stop()

summary = summary.copy()
summary["label"] = summary["title"].str.slice(0, 80) + "  ·  " + summary["product_id"]
dept = st.sidebar.selectbox("Department", ["All"] + sorted(summary["department"].dropna().unique().tolist()))
view = summary if dept == "All" else summary[summary["department"] == dept]
choice = st.selectbox("Product", view["label"].tolist())
pid = view[view["label"] == choice]["product_id"].iloc[0]
diag = load_diagnostic(pid)
if not diag:
    st.error("diagnostic file missing")
    st.stop()

lst = diag["listing"]
st.subheader(lst["title"])
st.caption(f"ASIN {pid} · brand: {lst['brand'] or 'n/a'} · color: {lst['color'] or 'n/a'} · department: {diag.get('department')}")
with st.expander("Original listing text", expanded=False):
    st.markdown("**Bullets:** " + (lst["bullets"] or "_none_"))
    st.markdown("**Description:** " + (lst["description"] or "_none_"))

r = diag["retrieval"]["hybrid"]
a = diag["agent"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Real shopper queries", diag["n_e_queries"], help="Queries where humans labelled this product Exact")
c2.metric("Found in top-10", pct(r["top10_rate"]), help="Hybrid BM25+dense retrieval over 1.2M US products")
c3.metric("Median rank", f"{r['median_rank']:.0f}" if r["median_rank"] == r["median_rank"] else "n/a", help=">100 = not in top 100")
c4.metric("Agent picked it", pct(a["pick_rate_E"]), help=f"fair share if the agent chose uniformly among Exact candidates: {pct(a['fair_share'])}")
c5.metric("Picked when it should NOT be", pct(a["pick_rate_nonE"]), help="On queries where humans labelled it Substitute/Irrelevant")

st.subheader("This is how real shoppers search for it")
pq = pd.DataFrame(diag["per_query"])
show = pq[["query", "human_label", "rank_bm25", "rank_dense", "rank_hybrid", "agent_label", "picked", "reason_code", "missing_attributes"]].copy() \
    if "agent_label" in pq.columns else pq[["query", "human_label", "rank_bm25", "rank_dense", "rank_hybrid"]].copy()
for c in ["rank_bm25", "rank_dense", "rank_hybrid"]:
    show[c] = show[c].map(lambda v: ">100" if v is None or v != v else f"{int(v)}")
if "missing_attributes" in show.columns:
    show["missing_attributes"] = show["missing_attributes"].map(lambda m: ", ".join(m) if isinstance(m, list) else "")
st.dataframe(show, hide_index=True, use_container_width=True)

l, rr = st.columns(2)
with l:
    st.markdown("**Why the agent passed on it** (Exact queries, not picked)")
    if a["top_reasons"]:
        df = pd.DataFrame(a["top_reasons"], columns=["reason", "count"])
        fig = px.bar(df, x="count", y="reason", orientation="h")
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("_no agent data_")
with rr:
    st.markdown("**Attributes the agent could not confirm**")
    if a["top_missing_attributes"]:
        st.dataframe(pd.DataFrame(a["top_missing_attributes"], columns=["attribute", "mentions"]), hide_index=True, use_container_width=True)
    else:
        st.write("_none recorded_")
    bm, de = diag["retrieval"]["bm25"]["median_rank"], diag["retrieval"]["dense"]["median_rank"]
    if bm == bm and de == de:
        hint = ("keyword search finds it but semantic search does not: the title uses the right words but reads unlike what shoppers mean"
                if bm < de - 10 else
                "semantic search finds it but keyword search does not: shoppers use words that are missing from the listing"
                if de < bm - 10 else "keyword and semantic search agree")
        st.caption(f"BM25 median rank {bm:.0f} vs dense {de:.0f}: {hint}.")
