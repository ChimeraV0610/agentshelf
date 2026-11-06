"""Page 3 - faithful rewrite, before/after, and the anti-gaming controls."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from data import banner, load_csv, load_diagnostic, load_json, load_parquet, pct

st.set_page_config(page_title="Rewrite · AgentShelf", page_icon="✍️", layout="wide")
st.title("Rewrite: make the listing legible to the agent, without lying")
banner()

listings = load_parquet("rewrite/listings.parquet")
ba = load_parquet("rewrite/before_after.parquet")
ev = load_json("rewrite_eval.json")
if listings is None or ba is None:
    st.info("No rewrite results yet. Run `make rewrite rewrite-eval`.")
    st.stop()

if ev:
    st.subheader("Does it work, and is it honest?")
    table = load_csv("rewrite/headline_table.csv")
    if table is not None:
        st.dataframe(table, hide_index=True, use_container_width=True)
    e = ev["variants"].get("rewritten", {}).get("E_queries", {})
    ne = ev["variants"].get("rewritten", {}).get("nonE_queries", {})
    st_ = ev["variants"].get("stuffed", {})
    c1, c2, c3, c4 = st.columns(4)
    if e:
        c1.metric("Agent pick rate on real queries", f"{pct(e['pick_rate_before'])} → {pct(e['pick_rate_after'])}",
                  delta=f"{100*e['pick_delta']['delta']:+.0f} pts (CI {100*e['pick_delta']['ci_low']:+.0f}..{100*e['pick_delta']['ci_high']:+.0f})")
        c2.metric("Found in top-10", f"{pct(e['top10_before'])} → {pct(e['top10_after'])}")
    if ne:
        c3.metric("Picked where it should NOT be", f"{pct(ne['pick_rate_before'])} → {pct(ne['pick_rate_after'])}",
                  delta=f"{100*ne['pick_delta']['delta']:+.0f} pts", delta_color="inverse", help="Negative control: Substitute/Irrelevant queries")
    f = ev["faithfulness"]
    if "rewritten" in f:
        c4.metric("Faithfulness pass", pct(f["rewritten"]["pass_rate"]),
                  help=f"keyword-stuffed control passes {pct(f.get('stuffed', {}).get('pass_rate', float('nan')))} and is flagged as stuffing {pct(f.get('stuffed', {}).get('stuffing_flag_rate', float('nan')))}")
    st.caption("The rewriter never sees any search query for the product; all evaluation queries are held out. "
               "The 'stuffed' row is a deliberately cheating baseline that injects the product's own query terms.")

st.divider()
summary = load_parquet("diagnostics_summary.parquet")
pids = listings["product_id"].unique().tolist()
labels = {}
for pid in pids:
    t = summary[summary["product_id"] == pid]["title"].iloc[0] if summary is not None and (summary["product_id"] == pid).any() else pid
    labels[f"{str(t)[:80]}  ·  {pid}"] = pid
choice = st.selectbox("Product", list(labels))
pid = labels[choice]
variant = st.radio("Variant", ["rewritten", "stuffed (cheating control)"], horizontal=True)
vkey = "rewritten" if variant.startswith("rewritten") else "stuffed"
row = listings[(listings["product_id"] == pid) & (listings["variant"] == vkey)].iloc[0]
diag = load_diagnostic(pid) or {}
orig = diag.get("listing", {})

left, right = st.columns(2)
with left:
    st.markdown("### Before")
    st.markdown(f"**{orig.get('title', '')}**")
    for b in (orig.get("bullets") or "").split(" | ")[:5]:
        if b:
            st.markdown(f"- {b}")
with right:
    st.markdown("### After" + ("  🚫 (keyword-stuffed control)" if vkey == "stuffed" else ""))
    st.markdown(f"**{row['title']}**")
    claims = {c.get("text", ""): c for c in json.loads(row["claims"] or "[]")}
    unsupported = set(json.loads(row["unsupported"] or "[]"))
    for b in json.loads(row["bullets"] or "[]"):
        mark = "🔴" if b in unsupported else ("🟢" if row["llm_pass"] else "⚪")
        st.markdown(f"- {mark} {b}")
    status = "PASSED" if row["passed"] else "FAILED"
    st.markdown(f"**Faithfulness audit: {status}** · rounds: {row['rounds']}")
    issues = json.loads(row["rule_issues"] or "[]")
    if issues:
        st.markdown("Rule layer: " + "; ".join(issues))
    if unsupported:
        st.markdown("Unsupported claims: " + "; ".join(unsupported))
    if row.get("stuffing_flag"):
        st.markdown("Judge flagged this as keyword stuffing.")
    attrs = json.loads(row["attributes_not_available"] or "[]")
    if attrs:
        st.info("Seller to-do, information shoppers ask for that the listing does not state: " + ", ".join(attrs))

st.markdown("### Same real queries, before vs after")
sub = ba[(ba["product_id"] == pid) & ba["variant"].isin(["original", vkey])]
piv = sub.pivot_table(index=["query", "label"], columns="variant", values=["rank_hybrid", "picked"], aggfunc="first")
piv.columns = [f"{a}_{b}" for a, b in piv.columns]
cols = [c for c in ["rank_hybrid_original", f"rank_hybrid_{vkey}", "picked_original", f"picked_{vkey}"] if c in piv.columns]
piv = piv[cols].reset_index()
for c in [c for c in cols if c.startswith("rank")]:
    piv[c] = piv[c].map(lambda v: ">100" if v is None or v != v else f"{int(v)}")
st.dataframe(piv, hide_index=True, use_container_width=True)
st.caption("label E = humans say this product is the exact match for the query; S/I = it is not and should not be picked.")
