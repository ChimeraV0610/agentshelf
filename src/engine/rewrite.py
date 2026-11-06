"""Rewrite per the product spec: brand + category + decisive specs first in the title, 'Name: fact' bullets,
a structured attribute table, and a 'seller to confirm' list. Only facts from the original listing.
Audited by the rule layer + an LLM judge (same code as the batch pipeline)."""
from __future__ import annotations

from src.agent.prompts import FAITHFULNESS_SYSTEM, render_faithfulness_user
from src.agent.schema import FAITHFULNESS_SCHEMA
from src.config import CFG
from src.engine.schemas import REWRITE_SCHEMA
from src.rewrite.faithfulness import original_text_blob, parse_llm_check, rule_check

REWRITE_SYSTEM = """You are rewriting an Amazon listing so that an AI shopping agent, which reads only the text, can find and
compare every decision-relevant fact. You must use ONLY facts stated in the ORIGINAL listing. Never invent
specifications, materials, compatibility, certifications, quantities, warranty terms or use cases.

Produce:
1. title: brand (if known), product category, then the most decisive specs, max {title_max} characters, no filler words.
2. bullets: 5 to 7 items, each a short label and ONE factual statement of at most 20 words
   ("Capacity: 32 oz", "Filter: true HEPA H13"), each with the exact span of the original text that supports it.
3. attributes: a table of at most 10 concrete attributes the original states (name/value/evidence). Never write
   "n/a" rows: an attribute the original does not state goes into seller_to_confirm instead.
If the original states few or no concrete facts, keep the title short and factual and put the missing items
under seller_to_confirm; do not pad.
4. seller_to_confirm: attributes shoppers in this category usually ask for ({dept_profile}) that the original
   listing does NOT state. Do not guess their values.
Return JSON only."""


def render_system(dept_profile: str) -> str:
    return REWRITE_SYSTEM.format(title_max=CFG["rewrite"]["title_max_chars"], dept_profile=dept_profile or "size, weight, material, compatibility, warranty, certifications, package contents")


def render_user(fields: dict) -> str:
    return ("ORIGINAL LISTING\n"
            f"Title: {fields['title']}\nBrand: {fields.get('brand') or 'n/a'}\nColor: {fields.get('color') or 'n/a'}\n"
            f"Bullets: {fields.get('bullets_full') or 'n/a'}\nDescription: {fields.get('description_full') or 'n/a'}")


def _bullet_lines(d: dict) -> list[str]:
    return [f"{b.get('name', '').strip()}: {b.get('fact', '').strip()}".strip(": ") for b in d.get("bullets", []) if b.get("fact")]


def rewrite(client, fields: dict, dept_profile: str, model: str, judge_model: str, tag: str = "rw", max_rounds: int | None = None) -> dict:
    max_rounds = max_rounds or CFG["rewrite"]["max_rounds"]
    system = render_system(dept_profile)
    user = render_user(fields)
    blob = original_text_blob(fields)
    best, history = None, []
    for rnd in range(1, max_rounds + 1):
        res = client.call(model=model, system=system, user=user, schema=REWRITE_SCHEMA, tag=f"{tag}:r{rnd}", max_tokens=1800)
        if not (res.get("ok") and res.get("data")):
            history.append({"round": rnd, "error": res.get("error")})
            continue
        d = res["data"]
        title = str(d.get("title", "")).strip()
        bullets = _bullet_lines(d)
        attrs = [{"name": a.get("name", ""), "value": a.get("value", ""), "evidence": a.get("evidence", "")} for a in d.get("attributes", []) if a.get("value")]
        audit_bullets = bullets + [f"{a['name']}: {a['value']}" for a in attrs]
        rule = rule_check(blob, title, bullets[:5] if len(bullets) >= 5 else bullets + [""] * (5 - len(bullets)))
        rule["issues"] = [i for i in rule["issues"] if not i.startswith("expected")]  # bullet count handled by the prompt (5-7)
        rule["pass"] = not rule["issues"]
        # the rule layer also checks numbers in the attribute table
        extra = rule_check(blob, "", audit_bullets[-5:] if len(audit_bullets) >= 5 else audit_bullets + [""] * (5 - len(audit_bullets)))
        rule["issues"] += [i for i in extra["issues"] if i.startswith("numbers") and i not in rule["issues"]]
        rule["pass"] = not rule["issues"]
        llm_res = client.call(model=judge_model, system=FAITHFULNESS_SYSTEM,
                              user=render_faithfulness_user(fields, {"title": title, "bullets": audit_bullets}),
                              schema=FAITHFULNESS_SCHEMA, tag=f"{tag}:faith:r{rnd}", max_tokens=1500)
        llm = parse_llm_check(llm_res)
        passed = rule["pass"] and llm["pass"]
        best = {"title": title, "bullets": bullets, "attributes": attrs, "seller_to_confirm": [str(s) for s in d.get("seller_to_confirm", [])],
                "evidence": [b.get("evidence", "") for b in d.get("bullets", [])], "rounds": rnd, "rule": rule, "llm": llm, "passed": passed}
        history.append({"round": rnd, "passed": passed, "rule_issues": rule["issues"], "unsupported": llm["unsupported"]})
        if passed:
            break
        problems = rule["issues"] + [f"unsupported claim: {u}" for u in llm["unsupported"]]
        user = (render_user(fields) + "\n\nYOUR PREVIOUS ATTEMPT FAILED THE FAITHFULNESS AUDIT.\n"
                f"Previous title: {title}\nPrevious bullets: {' | '.join(bullets)}\nProblems:\n" + "\n".join(f"- {p}" for p in problems)
                + "\nRewrite again using only facts from the original listing.")
    if best is None:
        best = {"title": fields["title"], "bullets": fields.get("bullets_list", [])[:5], "attributes": [], "seller_to_confirm": [],
                "evidence": [], "rounds": max_rounds, "rule": {"pass": False, "issues": ["no model output"]},
                "llm": {"pass": False, "unsupported": [], "claims": [], "stuffing_flag": None}, "passed": False}
    best["history"] = history
    return best


def apply_rewrite(fields: dict, rw: dict, confirmed: dict[str, str] | None = None) -> dict:
    """New fields dict for the rewritten listing; confirmed = seller-supplied facts {attribute: value}."""
    bullets = list(rw["bullets"])
    attrs = [f"{a['name']}: {a['value']}" for a in rw.get("attributes", [])]
    if confirmed:
        attrs += [f"{k}: {v}" for k, v in confirmed.items() if v and v.strip()]
    desc = " | ".join(attrs) if attrs else fields.get("description_full", "")
    new = dict(fields)
    new.update({"title": rw["title"], "bullets_list": bullets, "bullets_full": " | ".join(bullets),
                "bullets": " | ".join(bullets)[:500], "description_full": desc, "description": desc[:600]})
    return new
