"""Fast unit tests for the pure-Python pieces (no data, no API)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.agent.judge import parse_judgment, select_for_prompt
from src.eval.ndcg import ndcg, ndcg_from_scores
from src.retrieval.hybrid import rrf, rrf_scores_for_candidates
from src.rewrite.faithfulness import rule_check
from src.rewrite.stuffed_baseline import stuffed_listing


def test_ndcg_perfect_and_worst():
    assert ndcg(["E", "S", "I"]) == 1.0
    assert ndcg(["I", "S", "E"]) < 1.0
    assert ndcg(["I", "I"]) == 0.0
    assert abs(ndcg_from_scores([0.1, 0.9], ["I", "E"]) - 1.0) < 1e-9


def test_rrf():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert [p for p, _ in fused][:2] == ["a", "b"] or [p for p, _ in fused][:2] == ["b", "a"]
    s = rrf_scores_for_candidates([np.array([1.0, 0.0]), np.array([1.0, 0.0])])
    assert s[0] > s[1]


def test_select_for_prompt_keeps_all_E():
    df = pd.DataFrame({"product_id": [f"p{i}" for i in range(40)],
                       "esci_label": ["E"] * 5 + ["S"] * 35, "query_id": 1})
    out = select_for_prompt(df, seed=1, max_n=30)
    assert out["in_prompt"].sum() == 30
    assert out[out["esci_label"] == "E"]["in_prompt"].all()


def test_parse_judgment_maps_ids_and_pick():
    payload = [{"cid": "c01", "product_id": "A"}, {"cid": "c02", "product_id": "B"}]
    res = {"ok": True, "data": {"judgments": [{"id": "c01", "label": "E", "confidence": 0.9, "missing_attributes": [], "reason_code": "best_match"},
                                               {"id": "c02", "label": "S", "confidence": 0.5, "missing_attributes": ["size"], "reason_code": "missing_spec"}],
                                "pick": "c01", "pick_reason": "ok"}}
    rows = {r["product_id"]: r for r in parse_judgment(res, payload)}
    assert rows["A"]["is_pick"] and rows["A"]["agent_label"] == "E"
    assert not rows["B"]["is_pick"] and rows["B"]["missing_attributes"] == ["size"]
    assert not rows["A"]["abstained"]


def test_rule_check_catches_new_numbers_and_stuffing():
    orig = "Stainless steel water bottle 32 oz with straw"
    ok = rule_check(orig, "Water Bottle 32 oz Stainless Steel with Straw", ["32 oz capacity", "stainless steel", "straw lid", "a", "b"])
    assert ok["pass"], ok
    bad = rule_check(orig, "Water Bottle 64 oz best", ["x"] * 5)
    assert not bad["pass"] and any("64" in i for i in bad["issues"]) and any("banned" in i for i in bad["issues"])
    stuffed = rule_check(orig, "bottle", ["water bottle straw water bottle straw water bottle straw"] * 5)
    assert any("repeated" in i for i in stuffed["issues"])


def test_stuffed_listing_adds_query_terms():
    fields = {"title": "Insulated Bottle", "bullets_list": ["keeps cold"]}
    st = stuffed_listing(fields, ["hydro flask 32 oz", "32 oz water bottle"], n_bullets=5)
    assert "32" in st["title"] and "oz" in st["title"] and len(st["bullets"]) == 5


def test_rule_check_allows_repeats_present_in_original():
    orig = "Filters 1252 sq ft with 1 air change. Filters 626 sq ft with 2 air changes. Filters 417 sq ft with 3 air changes."
    ok = rule_check(orig, "Air Purifier", ["Coverage: 1252 sq ft with 1 air change, 626 sq ft with 2 air changes, 417 sq ft with 3 air changes", "a", "b", "c", "d"])
    assert not any("repeated" in i for i in ok["issues"]), ok
