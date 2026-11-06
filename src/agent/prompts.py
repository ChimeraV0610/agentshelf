"""Prompt templates for the shopping agent, department tagger, rewriter and faithfulness judge."""
from __future__ import annotations

from src.config import CFG

JUDGE_SYSTEM = """You are a shopping assistant acting on behalf of a customer who typed a search query
and asked you to pick the product to buy. You can only see the product listings below;
you cannot browse, ask follow-up questions, or see prices, ratings or reviews.

For every candidate, assign exactly one relevance label using these definitions
(they follow Amazon's human annotation guidelines):
- E (Exact): the product satisfies all the specifications stated or clearly implied by the query.
- S (Substitute): the product is somewhat relevant; it fails to satisfy some aspect of the
  query but could serve as a functional substitute.
- C (Complement): the product does not satisfy the query but could be used together with
  the exact product (e.g. a case for the requested phone).
- I (Irrelevant): the product is irrelevant or fails a central aspect of the query.

Then decide what you would add to the cart: the single best candidate id, or null if no
candidate is an E and you would rather ask the customer. Base every judgment strictly on
the listing text; do not rely on outside knowledge about specific products or brands.
For each candidate, list the query attributes the listing fails to confirm (at most 4),
give a confidence between 0 and 1, and pick one reason_code: use best_match only for the
candidate you pick; otherwise choose the main reason it is not the best choice.
Judge every candidate; the judgments array must contain one entry per candidate id.
Respond with JSON only, matching the provided schema."""


def render_judge_user(query: str, candidates: list[dict]) -> str:
    """candidates: list of dicts with keys cid, title, brand, color, bullets, description."""
    lines = [f'Query: "{query}"', "", "Candidates (order is random):"]
    for c in candidates:
        lines.append(f"[{c['cid']}] Title: {c['title']}")
        lines.append(f"      Brand: {c['brand'] or 'n/a'} | Color: {c['color'] or 'n/a'}")
        if c["bullets"]:
            lines.append(f"      Bullets: {c['bullets']}")
        if c["description"]:
            lines.append(f"      Description: {c['description']}")
    return "\n".join(lines)


DEPARTMENT_SYSTEM = """You classify e-commerce search queries into exactly one department from this list:
{departments}

Pick the department a shopper would browse to find what the query asks for. If a query is
ambiguous, choose the most likely department. Return JSON only: one item per query_id."""


def render_department_system() -> str:
    return DEPARTMENT_SYSTEM.format(departments="\n".join(f"- {d}" for d in CFG["departments"]))


def render_department_user(rows: list[tuple[int, str]]) -> str:
    return "\n".join(f"{qid}\t{q}" for qid, q in rows)


REWRITE_SYSTEM = """You are a listing copywriter for an e-commerce seller. Your reader is an AI shopping
agent that decides purely from listing text, so clarity and completeness of factual
attributes matter more than persuasion.

Rewrite the title (max {title_max} characters) and write exactly {n_bullets} bullet points for the
product below. Rules:
1. Every fact must be supported by the ORIGINAL listing. Do not invent specifications,
   compatibility, materials, quantities, certifications or use cases.
2. Put the product type and its most decision-relevant attributes first in the title.
3. No marketing filler ("premium", "best", "#1", "amazing"). No repeated phrases.
4. Shoppers in this department typically specify: {dept_profile}
   If the original listing states any of these, surface them; if it does not, do NOT guess,
   and list them under attributes_not_available instead.
5. For each bullet, include the exact span of the original text that supports it.
Return JSON only."""


def render_rewrite_system(dept_profile: str) -> str:
    return REWRITE_SYSTEM.format(title_max=CFG["rewrite"]["title_max_chars"],
                                 n_bullets=CFG["rewrite"]["n_bullets"], dept_profile=dept_profile)


def render_rewrite_user(fields: dict) -> str:
    return ("ORIGINAL LISTING\n"
            f"Title: {fields['title']}\n"
            f"Brand: {fields['brand'] or 'n/a'}\nColor: {fields['color'] or 'n/a'}\n"
            f"Bullets: {fields['bullets_full'] or 'n/a'}\n"
            f"Description: {fields['description_full'] or 'n/a'}")


FAITHFULNESS_SYSTEM = """You are auditing a rewritten product listing against the original. For each claim in the
rewritten title and bullets, decide whether it is SUPPORTED by the original text,
UNSUPPORTED (contradicts or adds a fact not present), or UNVERIFIABLE (vague, no factual
content). Quote the supporting span when SUPPORTED. Be strict: a more specific claim than
the original (e.g. "stainless steel" when the original says "metal") is UNSUPPORTED.
Set overall_pass to true only if there is no UNSUPPORTED claim. Set stuffing_flag to true if
the rewrite repeats search terms unnaturally or lists attributes as bare keywords.
Return JSON only."""


def render_faithfulness_user(original: dict, rewritten: dict) -> str:
    bullets = "\n".join(f"- {b}" for b in rewritten["bullets"])
    return ("ORIGINAL LISTING\n"
            f"Title: {original['title']}\nBrand: {original['brand'] or 'n/a'}\nColor: {original['color'] or 'n/a'}\n"
            f"Bullets: {original['bullets_full'] or 'n/a'}\nDescription: {original['description_full'] or 'n/a'}\n\n"
            "REWRITTEN LISTING\n"
            f"Title: {rewritten['title']}\nBullets:\n{bullets}")


DEPT_PROFILE_SYSTEM = """You are given a sample of real shopper search queries from one e-commerce department.
Summarise, in at most 150 words, which attribute TYPES shoppers in this department
typically specify (e.g. size, compatible device model, material, pack quantity, intended
user, power source). Describe attribute types, do not list product names or brands.
Return JSON only."""
