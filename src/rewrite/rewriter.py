"""Faithful listing rewrite with up to N repair rounds driven by the faithfulness checks.

The rewriter sees ONLY the product's own original listing and a department-level
attribute profile. It never sees any search query for this product.
"""
from __future__ import annotations

from src.agent.prompts import render_rewrite_system, render_rewrite_user
from src.agent.schema import REWRITE_SCHEMA
from src.config import CFG
from src.data.load import clean_text, split_bullets
from src.rewrite.faithfulness import llm_check_job, original_text_blob, parse_llm_check, rule_check


def full_fields(row) -> dict:
    return {"title": clean_text(row.get("product_title")), "brand": clean_text(row.get("product_brand")),
            "color": clean_text(row.get("product_color")),
            "bullets_full": " | ".join(split_bullets(row.get("product_bullet_point"))),
            "bullets_list": split_bullets(row.get("product_bullet_point")),
            "description_full": clean_text(row.get("product_description"))}


def rewrite_listing(client, fields: dict, dept_profile: str, model: str, judge_model: str, pid: str,
                    max_rounds: int | None = None) -> dict:
    """Returns {'title','bullets','attributes_not_available','rounds','rule','llm','passed','history'}."""
    max_rounds = max_rounds or CFG["rewrite"]["max_rounds"]
    system = render_rewrite_system(dept_profile)
    user = render_rewrite_user(fields)
    blob = original_text_blob(fields)
    history = []
    best = None
    for rnd in range(1, max_rounds + 1):
        res = client.call(model=model, system=system, user=user, schema=REWRITE_SCHEMA, tag=f"rewrite:{pid}:r{rnd}")
        if not (res.get("ok") and res.get("data")):
            history.append({"round": rnd, "error": res.get("error")})
            continue
        d = res["data"]
        title = str(d.get("title", "")).strip()
        bullets = [str(b.get("text", "")).strip() for b in d.get("bullets", []) if b.get("text")]
        evidence = [str(b.get("evidence", "")) for b in d.get("bullets", [])]
        rule = rule_check(blob, title, bullets)
        llm = parse_llm_check(client.call(**llm_check_job(fields, {"title": title, "bullets": bullets}, judge_model,
                                                          tag=f"faith:{pid}:r{rnd}")))
        passed = rule["pass"] and llm["pass"]
        cand = {"title": title, "bullets": bullets, "evidence": evidence,
                "attributes_not_available": d.get("attributes_not_available", []),
                "rounds": rnd, "rule": rule, "llm": llm, "passed": passed}
        history.append({"round": rnd, "passed": passed, "rule_issues": rule["issues"], "unsupported": llm["unsupported"]})
        best = cand
        if passed:
            break
        problems = rule["issues"] + [f"unsupported claim: {u}" for u in llm["unsupported"]]
        user = (render_rewrite_user(fields) + "\n\nYOUR PREVIOUS ATTEMPT FAILED THE FAITHFULNESS AUDIT.\n"
                f"Previous title: {title}\nPrevious bullets: {' | '.join(bullets)}\nProblems:\n"
                + "\n".join(f"- {p}" for p in problems)
                + "\nRewrite again so that every claim is supported by the original listing.")
    if best is None:
        best = {"title": fields["title"], "bullets": fields["bullets_list"][: CFG["rewrite"]["n_bullets"]], "evidence": [],
                "attributes_not_available": [], "rounds": max_rounds, "rule": {"pass": False, "issues": ["no LLM output"]},
                "llm": {"pass": False, "unsupported": [], "claims": [], "stuffing_flag": None}, "passed": False}
    best["history"] = history
    return best
