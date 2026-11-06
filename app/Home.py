"""AgentShelf: four steps, end to end, live. Audit -> simulated AI buyers -> faithful rewrite + retest -> fill gaps + retest."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine.audit import audit_listing  # noqa: E402
from src.engine.fetch import Catalog, fetch_amazon, generate_buyers, parse_asin  # noqa: E402
from src.engine.rewrite import apply_rewrite, rewrite  # noqa: E402
from src.engine.simulate import simulate  # noqa: E402
from src.live import available_models  # noqa: E402
from src.llm.client import get_client  # noqa: E402
from src.rewrite.dept_profile import STATIC_PROFILES  # noqa: E402
from src.shelves import load_buyers, load_shelves  # noqa: E402

st.set_page_config(page_title="AgentShelf", page_icon="🛒", layout="wide")


@st.cache_resource
def _client():
    return get_client()


@st.cache_data
def _data():
    return load_shelves(), load_buyers()


@st.cache_resource(show_spinner="Loading the 1.2M-product catalog…")
def _catalog():
    return Catalog()


def fields_of(p: dict) -> dict:
    return {"title": p["title"], "brand": p.get("brand", ""), "color": p.get("color", ""), "bullets_list": list(p["bullets"]),
            "bullets_full": " | ".join(p["bullets"]), "bullets": " | ".join(p["bullets"])[:500],
            "description_full": p.get("description", ""), "description": (p.get("description") or "")[:600]}


def pct(x):
    return "n/a" if x is None or x != x else f"{100 * x:.0f}%"


shelves, buyers = _data()
models = available_models()
if os.environ.get("AGENTSHELF_MOCK") == "1":
    models = ["mock"] + models
models.append("heuristic (offline, not an LLM)")

# ---------------- sidebar ----------------
with st.sidebar:
    st.title("AgentShelf")
    st.caption("When the customer is a bot, the listing is the store.")
    model_label = st.selectbox("Shopping-agent model", models, help="Local Ollama models run fully offline.")
    model = "heuristic" if model_label.startswith("heuristic") else model_label
    llm_models = [m for m in models if not m.startswith("heuristic")]
    rewriter = st.selectbox("Rewriter + audit model", llm_models or ["heuristic"], index=0)
    category = st.selectbox("Shelf (real Amazon products, same search)", list(shelves))
    n_buyers = st.slider("Simulated buyers", 3, len(buyers[category]), min(4, len(buyers[category])),
                         help="Each buyer is one agent call. A 7B local model needs ~15-25 s per buyer.")
    if st.button("Reset"):
        for k in list(st.session_state):
            del st.session_state[k]
        st.rerun()

shelf = shelves[category]
needs = buyers[category][:n_buyers]
st.title("Will the AI shopper pick your listing?")
st.caption(f"Quick-demo shelf: “{shelf['query']}” ({len(shelf['products'])} real Exact products). Core metric: how often the agent picks YOUR listing "
           "on a shelf of real competitors, vs the fair share of 1/8.")

# ---------------- step 0: whose listing ----------------
st.header("0 · Your listing")
mode = st.radio("", ["Paste an Amazon link or ASIN", "Pick one of the shelf products as mine", "Paste the listing text"],
                horizontal=True, label_visibility="collapsed")
audits0 = [audit_listing(fields_of(p), shelf["buyer_vocab"])["score"] for p in shelf["products"]]
shelf_vocab, shelf_needs, competitors_note = shelf["buyer_vocab"], needs, f"the other {len(shelf['products']) - 1} products on this shelf"
if mode.startswith("Paste an Amazon"):
    with st.form("link"):
        link = st.text_input("Amazon product link or ASIN", placeholder="https://www.amazon.com/dp/B071D58ZY5 or B071D58ZY5")
        go_link = st.form_submit_button("Fetch listing", type="primary")
    if go_link:
        asin = parse_asin(link)
        if not asin:
            st.error("Could not find a 10-character ASIN in that text.")
            st.stop()
        cat = _catalog()
        with st.spinner("Looking up the catalog…"):
            f = cat.lookup(asin)
            source = "local ESCI catalog (1.2M US products)"
        if f is None:
            with st.spinner("Not in the local catalog; trying amazon.com (often blocked)…"):
                f = fetch_amazon(asin)
                source = "live amazon.com"
        if f is None:
            st.error("Amazon blocked the fetch and the ASIN is not in the local catalog. Use 'Paste the listing text'.")
            st.stop()
        with st.spinner("Finding real competitors…"):
            comps, how, queries = cat.competitors(asin, f)
        # known category -> use the hand-written buyers (no model call); otherwise ask the model for buyers
        text = (f["title"] + " " + f["bullets_full"]).lower()
        matched = next((c for c in buyers if all(w in text for w in c.replace("1tb ssd", "ssd").split())), None)
        if matched:
            gen, buyers_note = buyers[matched], f"buyers: prewritten set for '{matched}'"
        else:
            with st.spinner("Writing simulated buyers for this category (one local-model call, ~30 s)…"):
                gen = generate_buyers(_client(), rewriter if rewriter != "heuristic" else model, f, n=6) if model != "heuristic" else []
            buyers_note = "buyers: generated by the model" if gen else f"buyers: fallback set for '{category}'"
        st.session_state["linked"] = {"asin": asin, "fields": f, "source": source, "competitors": comps, "how": how,
                                      "vocab": cat.vocab(queries, f), "buyers": gen or buyers[category], "buyers_note": buyers_note}
    if "linked" not in st.session_state:
        st.info("Paste a link; the listing, 7 real competitors and the simulated buyers are derived automatically.")
        st.stop()
    L = st.session_state["linked"]
    mine, competitors, shelf_vocab = L["fields"], L["competitors"], L["vocab"]
    shelf_needs = L["buyers"][:n_buyers]
    competitors_note = L["how"]
    mine_key = f"link:{L['asin']}"
    st.success(f"ASIN {L['asin']} from {L['source']} · competitors: {L['how']} · {L.get('buyers_note', '')}")
    with st.expander("Competitors on the shelf"):
        for c in competitors:
            st.markdown(f"- {c['title'][:110]}")
elif mode.startswith("Pick"):
    labels = [f"{a:>3}/100 · {p['title'][:90]}" for a, p in zip(audits0, shelf["products"])]
    full = [audit_listing(fields_of(p), shelf["buyer_vocab"]) for p in shelf["products"]]
    real = [i for i, a in enumerate(full) if a["dimensions"]["numeric specs"]["score"] > 0 or a["dimensions"]["key attributes"]["score"] > 0] or list(range(len(audits0)))
    default = min(real, key=lambda i: audits0[i])  # weakest listing that still states at least one real spec/attribute
    choice = st.selectbox("Shelf product (audit score · title)", labels, index=default)
    i = labels.index(choice)
    mine = fields_of(shelf["products"][i])
    competitors = [p for j, p in enumerate(shelf["products"]) if j != i]
    mine_key = f"{category}:{shelf['products'][i]['product_id']}"
else:
    with st.form("own"):
        t = st.text_input("Title *")
        b = st.text_area("Bullet points (one per line)", height=110)
        c1, c2 = st.columns(2)
        br = c1.text_input("Brand")
        d = c2.text_area("Description (optional)", height=70)
        ok = st.form_submit_button("Use this listing")
    if ok and t.strip():
        st.session_state["own"] = {"title": t.strip(), "bullets": [x.strip() for x in b.splitlines() if x.strip()], "brand": br.strip(), "description": d.strip(), "color": ""}
    if "own" not in st.session_state:
        st.info("Enter a title (and ideally bullets) to continue.")
        st.stop()
    mine = fields_of(st.session_state["own"])
    competitors = shelf["products"][:7]
    mine_key = f"{category}:own:{hash(mine['title'] + mine['bullets_full']) & 0xFFFFFF}"
needs = shelf_needs

if st.session_state.get("mine_key") != mine_key:
    for k in ["sim0", "rw", "sim1", "sim2", "confirmed"]:
        st.session_state.pop(k, None)
    st.session_state["mine_key"] = mine_key

# ---------------- step 1: audit ----------------
st.header("1 · Audit: is this listing agent-friendly?")
a0 = audit_listing(mine, shelf_vocab)
c1, c2 = st.columns([1, 3])
c1.metric("Agent-friendliness", f"{a0['score']} / 100")
with c2:
    for name, dmn in a0["dimensions"].items():
        st.progress(dmn["score"] / 20, text=f"{name}: {dmn['score']}/20 · {dmn['detail']}")
if a0["issues"]:
    st.markdown("**Issues**")
    for iss in a0["issues"]:
        st.markdown(f"- {iss}")
with st.expander("Listing text"):
    st.markdown(f"**{mine['title']}**")
    for bl in mine["bullets_list"]:
        st.markdown(f"- {bl}")
    if mine["description_full"]:
        st.caption(mine["description_full"][:800])


# ---------------- helpers ----------------
def run_sim(fields: dict, tag: str, label: str) -> dict:
    prog = st.progress(0.0, text=f"{label}: running {len(needs)} simulated buyers…")

    def cb(k, n, r):
        prog.progress(k / n, text=f"{label}: buyer {k}/{n} → rank {r['rank']}/{r['n']}{' · PICKED' if r['picked'] else ''} ({r['seconds']}s)")

    out = simulate(_client(), model, competitors, fields, needs, tag=f"sim:{mine_key}:{tag}", progress=cb)
    prog.empty()
    return out


def show_sim(s: dict, title: str):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agent picked my listing", pct(s["pick_rate"]), delta=f"fair share {pct(s['fair_share'])}", delta_color="off")
    c2.metric("In the agent's top 3", pct(s["top3_rate"]))
    c3.metric("Average rank", f"{s['avg_rank']:.1f} / {s['n']}" if s["avg_rank"] == s["avg_rank"] else "n/a")
    c4.metric("Seconds", f"{s['seconds']:.0f}")
    df = pd.DataFrame(s["per_buyer"])[["need", "rank", "picked", "reason", "missing_facts", "picked_instead", "omitted", "error"]]
    df["missing_facts"] = df["missing_facts"].map(lambda m: ", ".join(m) if isinstance(m, list) else "")
    st.dataframe(df, hide_index=True, use_container_width=True)
    if s["missing_facts"]:
        st.warning("Facts the buyers wanted but could not find in my listing: " + "; ".join(s["missing_facts"][:10]))
    if s.get("omitted_count"):
        st.caption(f"{s['omitted_count']} buyer(s) left my listing out of their ranking; counted as last place.")
    if s["n_ok"] < s["n_buyers"]:
        st.error(f"{s['n_buyers'] - s['n_ok']} buyer call(s) failed; see the error column.")


def stage_chart(stages: dict, fair: float):
    fig = go.Figure()
    fig.add_bar(x=list(stages), y=[v["pick_rate"] for v in stages.values()], text=[pct(v["pick_rate"]) for v in stages.values()], textposition="outside")
    fig.add_hline(y=fair, line_dash="dot", annotation_text=f"fair share {pct(fair)}")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.05], height=300, margin=dict(l=0, r=0, t=20, b=0), title="Agent pick rate by stage")
    st.plotly_chart(fig, use_container_width=True)


# ---------------- step 2: simulated buyers ----------------
st.header("2 · Simulated AI buyers")
st.caption("Each buyer is a real-sounding request. The agent sees only listing text, in random order, must rank every product and explain each rank.")
with st.expander("The buyers"):
    for nd in needs:
        st.markdown(f"- {nd}")
if st.button("Run simulated buyers", type="primary", disabled="sim0" in st.session_state):
    st.session_state["sim0"] = run_sim(mine, "orig", "Original listing")
if "sim0" in st.session_state:
    show_sim(st.session_state["sim0"], "Original")

# ---------------- step 3: rewrite + retest ----------------
if "sim0" in st.session_state:
    st.header("3 · Faithful rewrite, then the same buyers again")
    st.caption("The rewriter sees only the original listing plus which attribute types this category's shoppers ask for. "
               "It never sees the buyers. Every claim is audited against the original; nothing is invented.")
    if st.button("Rewrite and retest", type="primary", disabled="rw" in st.session_state):
        with st.spinner("Rewriting and auditing…"):
            st.session_state["rw"] = rewrite(_client(), mine, STATIC_PROFILES.get(shelf["department"], ""),
                                             rewriter if rewriter != "heuristic" else model, rewriter if rewriter != "heuristic" else model,
                                             tag=f"rw:{mine_key}")
        st.session_state["mine1"] = apply_rewrite(mine, st.session_state["rw"])
        st.session_state["sim1"] = run_sim(st.session_state["mine1"], "rw", "Rewritten listing")
    if "rw" in st.session_state:
        rw = st.session_state["rw"]
        mine1 = st.session_state["mine1"]
        a1 = audit_listing(mine1, shelf_vocab)
        l, r = st.columns(2)
        with l:
            st.markdown(f"**Before · audit {a0['score']}/100**")
            st.markdown(f"**{mine['title']}**")
            for bl in mine["bullets_list"][:7]:
                st.markdown(f"- {bl}")
        with r:
            st.markdown(f"**After · audit {a1['score']}/100 · faithfulness " + ("PASSED ✅" if rw["passed"] else "FAILED ❌") + f" (rounds {rw['rounds']})**")
            st.markdown(f"**{rw['title']}**")
            uns = set(rw["llm"].get("unsupported", []))
            for bl in rw["bullets"]:
                st.markdown(f"- {'🔴' if bl in uns else '🟢'} {bl}")
            if rw["attributes"]:
                st.dataframe(pd.DataFrame(rw["attributes"])[["name", "value", "evidence"]], hide_index=True, use_container_width=True)
            if rw["rule"]["issues"]:
                st.caption("Rule audit: " + "; ".join(rw["rule"]["issues"]))
        if "sim1" in st.session_state:
            show_sim(st.session_state["sim1"], "Rewritten")

# ---------------- step 4: fill the gaps ----------------
if "sim1" in st.session_state:
    st.header("4 · Fill the gaps the agents asked about, then retest")
    rw = st.session_state["rw"]
    gaps = list(dict.fromkeys([g for g in rw["seller_to_confirm"]] + st.session_state["sim0"]["missing_facts"][:6] + st.session_state["sim1"]["missing_facts"][:6]))
    if not gaps:
        st.info("The agents did not ask for anything the listing lacks.")
    else:
        with st.form("gaps"):
            st.caption("Only the seller knows these. Fill what is true; leave the rest empty.")
            vals = {g: st.text_input(g, key=f"gap:{g}") for g in gaps[:10]}
            go_ = st.form_submit_button("Retest with confirmed facts", type="primary")
        if go_:
            st.session_state["confirmed"] = {k: v for k, v in vals.items() if v.strip()}
            mine2 = apply_rewrite(mine, rw, st.session_state["confirmed"])
            st.session_state["sim2"] = run_sim(mine2, "gaps:" + str(hash(tuple(sorted(st.session_state["confirmed"].items()))) & 0xFFFF), "With confirmed facts")
    if "sim2" in st.session_state:
        show_sim(st.session_state["sim2"], "With confirmed facts")

# ---------------- summary ----------------
if "sim0" in st.session_state:
    st.header("Result")
    stages = {"original": st.session_state["sim0"]}
    if "sim1" in st.session_state:
        stages["rewritten"] = st.session_state["sim1"]
    if "sim2" in st.session_state:
        stages["rewritten + confirmed facts"] = st.session_state["sim2"]
    stage_chart(stages, st.session_state["sim0"]["fair_share"])
    st.caption("Same shelf, same buyers, same agent, random order each time. The only thing that changed is the listing text.")
