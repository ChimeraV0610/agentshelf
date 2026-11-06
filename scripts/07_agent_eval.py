"""Step 5: Module-1 analysis -> results/agent_eval.json + csv tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.analysis.agent_eval import (abstention, agreement, brand_field_effect, brand_view,  # noqa: E402
                                     candidate_level, query_level, random_pick_baseline, reason_summary,
                                     wrong_buy_summary)
from src.config import CFG, path  # noqa: E402
from src.data.load import load_processed  # noqa: E402


def main():
    d = path("results") / "agent_runs"
    files = sorted(f for f in d.glob("*.parquet") if "_limit" not in f.name)
    if not files:
        sys.exit("no agent runs found; run scripts/06_run_agents.py first")
    queries = pd.read_parquet(path("samples") / "queries_1000.parquet")
    cands = pd.read_parquet(path("samples") / "candidates_1000.parquet")
    products = load_processed("products_us")[["product_id", "product_brand"]]
    out_dir = path("results") / "agent_eval"
    out_dir.mkdir(exist_ok=True)
    report: dict = {"random_pick_wrong_buy_rate": random_pick_baseline(cands), "models": {}}
    runs_by_model = {}
    for f in files:
        runs = pd.read_parquet(f)
        model = runs["model"].iloc[0]
        runs_by_model[model] = runs
        q = query_level(runs, queries)
        q.to_parquet(out_dir / f"query_level_{model}.parquet", index=False)
        wb_all = wrong_buy_summary(q)
        tables = {by: wrong_buy_summary(q, by) for by in ["department", "is_brand_query", "difficulty"]}
        for by, t in tables.items():
            t.to_csv(out_dir / f"wrong_buy_by_{by}_{model}.csv", index=False)
        bv = brand_view(runs, products)
        bv.to_csv(out_dir / f"brand_view_{model}.csv", index=False)
        reasons = reason_summary(runs)
        reasons.to_csv(out_dir / f"reasons_E_not_picked_{model}.csv", index=False)
        report["models"][model] = {
            "candidate_level": candidate_level(runs),
            "wrong_buy": wb_all.iloc[0].to_dict(),
            "wrong_buy_by_department": tables["department"].to_dict("records"),
            "wrong_buy_by_brand_query": tables["is_brand_query"].to_dict("records"),
            "wrong_buy_by_difficulty": tables["difficulty"].to_dict("records"),
            "abstention": abstention(q),
            "brand_field_effect": brand_field_effect(runs, products),
            "brand_top5": bv.head(5).to_dict("records"),
            "brand_bottom5": bv.tail(5).to_dict("records"),
            "reasons_E_not_picked": reasons.to_dict("records"),
        }
        print(f"== {model}: wrong-buy {wb_all.iloc[0]['wrong_buy_rate']:.3f} "
              f"[{wb_all.iloc[0]['ci_low']:.3f},{wb_all.iloc[0]['ci_high']:.3f}] "
              f"macroF1 {report['models'][model]['candidate_level']['macro_f1']:.3f} "
              f"abstain {report['models'][model]['abstention']['abstain_rate']:.3f}")
    models = list(runs_by_model)
    if len(models) >= 2:
        report["agreement"] = {f"{models[0]} vs {models[1]}": agreement(runs_by_model[models[0]], runs_by_model[models[1]])}
    (path("results") / "agent_eval.json").write_text(json.dumps(report, indent=2, default=str))
    print("saved results/agent_eval.json")


if __name__ == "__main__":
    main()
