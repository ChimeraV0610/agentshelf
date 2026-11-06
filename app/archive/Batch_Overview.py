"""AgentShelf demo: page 1 - how often does the AI shopper buy the wrong thing?"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import banner, load_csv, load_json, pct

st.set_page_config(page_title="AgentShelf", page_icon="🛒", layout="wide")
st.title("AgentShelf · When the customer is a bot")
st.caption("What AI shopping agents do with real Amazon search intents (ESCI, US) and what sellers can do about it.")
banner()

ev = load_json("agent_eval.json")
if not ev:
    st.info("No agent evaluation yet. Run `make tag sample agents agent-eval`.")
    st.stop()

models = list(ev["models"].keys())
model = st.sidebar.selectbox("Agent model", models)
m = ev["models"][model]
wb = m["wrong_buy"]
cl = m["candidate_level"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Wrong-buy rate", pct(wb["wrong_buy_rate"], 1), help="Share of queries where the agent's pick is NOT the human-labelled Exact match")
c1.caption(f"95% CI {pct(wb['ci_low'],1)} – {pct(wb['ci_high'],1)} · n={wb['n_picked']}")
c2.metric("Random-pick baseline", pct(ev["random_pick_wrong_buy_rate"], 1))
c3.metric("Substitute bought as Exact", pct(wb["s_as_e_rate"], 1))
c4.metric("Agent asked instead of buying", pct(m["abstention"]["abstain_rate"], 1),
          help=f"of which an Exact match was available: {pct(m['abstention']['abstained_but_E_available'])}")

left, right = st.columns(2)
with left:
    st.subheader("Wrong-buy rate by department")
    df = pd.DataFrame(m["wrong_buy_by_department"]).sort_values("wrong_buy_rate")
    fig = px.bar(df, x="wrong_buy_rate", y="group", orientation="h", error_x=df["ci_high"] - df["wrong_buy_rate"],
                 error_x_minus=df["wrong_buy_rate"] - df["ci_low"], labels={"wrong_buy_rate": "wrong-buy rate", "group": ""},
                 text=df["n_picked"].map(lambda n: f"n={n}"))
    fig.update_layout(xaxis_tickformat=".0%", height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Human label vs agent label (all candidates)")
    cm = cl["confusion_matrix"]
    z = cm["rows_human_cols_agent"]
    fig = go.Figure(data=go.Heatmap(z=z, x=[f"agent {l}" for l in cm["labels"]], y=[f"human {l}" for l in cm["labels"]],
                                    text=z, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"macro F1 {cl['macro_f1']:.2f} · micro F1 {cl['micro_f1']:.2f} · κ vs human {cl['kappa_vs_human']:.2f} "
               f"(ESCI README baseline: macro F1 {cl['esci_readme_baseline']['macro_f1']}, micro F1 {cl['esci_readme_baseline']['micro_f1']})")

st.subheader("Why Exact products get passed over")
reasons = pd.DataFrame(m["reasons_E_not_picked"])
if len(reasons):
    fig = px.bar(reasons, x="share", y="reason_code", orientation="h", labels={"share": "share of non-picked Exact candidates", "reason_code": ""})
    fig.update_layout(xaxis_tickformat=".0%", height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Brand view: who wins the agent's cart?")
bv = load_csv(f"agent_eval/brand_view_{model}.csv")
if bv is not None and len(bv):
    bv = bv.rename(columns={bv.columns[0]: "brand"}) if "brand" not in bv.columns else bv
    b1, b2 = st.columns(2)
    b1.markdown("**Over-picked brands** (preference index = pick share ÷ candidate share)")
    b1.dataframe(bv.head(8)[["brand", "n_candidates", "n_picks", "preference_index"]].round(2), hide_index=True, use_container_width=True)
    b2.markdown("**Under-picked brands**")
    b2.dataframe(bv.tail(8)[["brand", "n_candidates", "n_picks", "preference_index"]].round(2), hide_index=True, use_container_width=True)
    bfe = m["brand_field_effect"]
    st.caption(f"Exact products WITH a brand field are picked {pct(bfe['pick_rate_E_with_brand'])} of the time vs "
               f"{pct(bfe['pick_rate_E_without_brand'])} without one.")

if ev.get("agreement"):
    k, a = next(iter(ev["agreement"].items()))
    st.info(f"Agent panel consistency ({k}): label κ {a['label_kappa']:.2f}, same final pick in {pct(a['same_pick_rate'])} of queries.")

st.divider()
st.markdown("**Next:** open *Shelf Diagnosis* to see a single product through the agent's eyes, then *Rewrite* for the fix.")
