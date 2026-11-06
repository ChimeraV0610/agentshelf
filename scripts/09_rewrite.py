"""Step 7: rewrite diagnosed listings (faithful + stuffed control), rescore retrieval, re-run the agent."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.agent.runner import run_agent  # noqa: E402
from src.config import CFG, path  # noqa: E402
from src.data.load import load_processed  # noqa: E402
from src.llm.client import get_client, has_api_key  # noqa: E402
from src.retrieval.bm25_index import BM25Index  # noqa: E402
from src.retrieval.dense_index import DenseIndex  # noqa: E402
from src.rewrite.dept_profile import build_profiles  # noqa: E402
from src.rewrite.faithfulness import llm_check_job, original_text_blob, parse_llm_check, rule_check  # noqa: E402
from src.rewrite.rescore import Rescorer, doc_text_from  # noqa: E402
from src.rewrite.rewriter import full_fields, rewrite_listing  # noqa: E402
from src.rewrite.stuffed_baseline import stuffed_listing  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CFG["models"]["agent_primary"], help="agent used for before/after picks")
    ap.add_argument("--rewriter", default=CFG["models"]["rewriter"])
    ap.add_argument("--judge", default=CFG["models"]["faithfulness_judge"])
    ap.add_argument("--limit", type=int, default=0, help="first N products only")
    ap.add_argument("--skip-agent", action="store_true", help="retrieval rescoring only")
    args = ap.parse_args()
    mock = os.environ.get("AGENTSHELF_MOCK") == "1"
    if not mock and not (has_api_key(args.rewriter) and has_api_key(args.model)):
        sys.exit("API keys missing; set .env or AGENTSHELF_MOCK=1")
    client = get_client()
    seed = CFG["seed"]
    ex = load_processed("examples_us")
    products = load_processed("products_us").set_index("product_id")
    depts = pd.read_parquet(path("samples") / "query_departments.parquet")
    selected = pd.read_parquet(path("samples") / "products_for_diagnosis.parquet")
    links = pd.read_parquet(path("samples") / "diag_links.parquet")
    dq = pd.read_parquet(path("samples") / "diag_queries.parquet")
    dc = pd.read_parquet(path("samples") / "diag_candidates.parquet")
    diag_runs_f = path("results") / "agent_runs" / f"diag_{args.model}.parquet"
    diag_runs = pd.read_parquet(diag_runs_f) if diag_runs_f.exists() else pd.DataFrame()
    if args.limit:
        selected = selected.head(args.limit)
    qtext = ex.drop_duplicates("query_id").set_index("query_id")["query"]

    profiles = build_profiles(ex[ex["split"] == "train"], depts, set(links["query_id"]), args.rewriter, client, seed=seed)
    (path("results") / "rewrite").mkdir(exist_ok=True)
    (path("results") / "rewrite" / "dept_profiles.json").write_text(json.dumps(profiles, indent=1))

    bm25 = BM25Index.load(path("indices") / "bm25")
    dense = DenseIndex.load(path("indices") / "dense")
    rescorer = Rescorer(bm25, dense)

    listing_rows, ba_rows = [], []
    for pid in tqdm(selected["product_id"].tolist(), desc="products"):
        row = products.loc[pid]
        fields = full_fields(row)
        dept = selected.set_index("product_id").loc[pid, "department"]
        my_links = links[links["product_id"] == pid]
        e_queries = [qtext[q] for q in my_links[my_links["target_label"] == "E"]["query_id"]]

        rw = rewrite_listing(client, fields, profiles.get(dept, ""), args.rewriter, args.judge, pid)
        st = stuffed_listing(fields, e_queries, n_bullets=CFG["rewrite"]["n_bullets"])
        st_rule = rule_check(original_text_blob(fields), st["title"], st["bullets"])
        st_llm = parse_llm_check(client.call(**llm_check_job(fields, st, args.judge, tag=f"faith:{pid}:stuffed")))
        variants = {
            "original": {"title": fields["title"], "bullets": fields["bullets_list"][: CFG["rewrite"]["n_bullets"]]},
            "rewritten": {"title": rw["title"], "bullets": rw["bullets"]},
            "stuffed": st,
        }
        listing_rows.append({"product_id": pid, "variant": "rewritten", "title": rw["title"], "bullets": json.dumps(rw["bullets"]),
                             "evidence": json.dumps(rw.get("evidence", [])), "attributes_not_available": json.dumps(rw["attributes_not_available"]),
                             "rounds": rw["rounds"], "passed": rw["passed"], "rule_pass": rw["rule"]["pass"], "rule_issues": json.dumps(rw["rule"]["issues"]),
                             "llm_pass": rw["llm"]["pass"], "unsupported": json.dumps(rw["llm"]["unsupported"]), "claims": json.dumps(rw["llm"]["claims"]),
                             "stuffing_flag": rw["llm"]["stuffing_flag"], "history": json.dumps(rw["history"])})
        listing_rows.append({"product_id": pid, "variant": "stuffed", "title": st["title"], "bullets": json.dumps(st["bullets"]),
                             "evidence": "[]", "attributes_not_available": "[]", "rounds": 1,
                             "passed": st_rule["pass"] and st_llm["pass"], "rule_pass": st_rule["pass"], "rule_issues": json.dumps(st_rule["issues"]),
                             "llm_pass": st_llm["pass"], "unsupported": json.dumps(st_llm["unsupported"]), "claims": json.dumps(st_llm["claims"]),
                             "stuffing_flag": st_llm["stuffing_flag"], "history": "[]"})

        # agent before/after
        qset = dq[dq["query_id"].isin(my_links["query_id"])]
        cset = dc[dc["query_id"].isin(my_links["query_id"])]
        agent_by_variant = {}
        if len(diag_runs):
            agent_by_variant["original"] = diag_runs[(diag_runs["product_id"] == pid) & diag_runs["query_id"].isin(my_links["query_id"])]
        if not args.skip_agent:
            for v in ["rewritten", "stuffed"]:
                override = {pid: {"title": variants[v]["title"], "bullets": " | ".join(variants[v]["bullets"])[:400]}}
                runs = run_agent(args.model, qset, cset, products, tag=f"rw:{v}:{pid}", override_text=override, client=client, verbose=False)
                agent_by_variant[v] = runs[runs["product_id"] == pid]

        for lk in my_links.itertuples(index=False):
            q = qtext[lk.query_id]
            for v, lst in variants.items():
                text = doc_text_from(lst["title"], fields["brand"], fields["color"], lst["bullets"])
                r = rescorer.ranks(pid, q, text)
                a = agent_by_variant.get(v)
                a = a[a["query_id"] == lk.query_id] if a is not None and len(a) else None
                ba_rows.append({"product_id": pid, "query_id": int(lk.query_id), "query": q, "label": lk.target_label, "variant": v,
                                **r, "picked": (bool(a["is_pick"].iloc[0]) if a is not None and len(a) else None),
                                "agent_label": (a["agent_label"].iloc[0] if a is not None and len(a) else None),
                                "reason_code": (a["reason_code"].iloc[0] if a is not None and len(a) else None)})
    listings = pd.DataFrame(listing_rows)
    ba = pd.DataFrame(ba_rows)
    listings.to_parquet(path("results") / "rewrite" / "listings.parquet", index=False)
    ba.to_parquet(path("results") / "rewrite" / "before_after.parquet", index=False)
    print(client.summary())
    print(listings.groupby("variant")[["passed", "rule_pass", "llm_pass"]].mean().round(3))
    print(ba.groupby(["label", "variant"])[["picked"]].mean().round(3))


if __name__ == "__main__":
    main()
