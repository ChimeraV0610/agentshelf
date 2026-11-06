"""Two-layer faithfulness check: cheap rules first, then an LLM judge."""
from __future__ import annotations

import re
from collections import Counter

from src.agent.prompts import FAITHFULNESS_SYSTEM, render_faithfulness_user
from src.agent.schema import FAITHFULNESS_SCHEMA
from src.config import CFG

_DIGIT_TOKEN = re.compile(r"[a-z]*\d[a-z0-9.\-/]*")
_WORD = re.compile(r"[a-z0-9]+")


def _digit_tokens(text: str) -> set[str]:
    return set(_DIGIT_TOKEN.findall(text.lower()))


def _ngrams(words: list[str], n: int = 3) -> Counter:
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def rule_check(original_text: str, title: str, bullets: list[str]) -> dict:
    """Returns {'pass': bool, 'issues': [str]}."""
    cfg = CFG["rewrite"]
    issues = []
    rewritten = title + " " + " ".join(bullets)
    orig_tokens = _digit_tokens(original_text)
    new_numbers = [t for t in _digit_tokens(rewritten) if t not in orig_tokens]
    if new_numbers:
        issues.append(f"numbers/model codes not in original: {sorted(new_numbers)[:8]}")
    low = rewritten.lower()
    banned = [w for w in cfg["banned_words"] if w.lower() in low]
    if banned:
        issues.append(f"banned marketing words: {banned}")
    if len(title) > cfg["title_max_chars"]:
        issues.append(f"title too long ({len(title)} > {cfg['title_max_chars']})")
    if len(bullets) != cfg["n_bullets"]:
        issues.append(f"expected {cfg['n_bullets']} bullets, got {len(bullets)}")
    words = _WORD.findall(low)
    repeated = [" ".join(g) for g, c in _ngrams(words).items() if c >= 3]
    if repeated:
        issues.append(f"repeated phrases (keyword stuffing?): {repeated[:5]}")
    return {"pass": not issues, "issues": issues}


def llm_check_job(original: dict, rewritten: dict, model: str, tag: str) -> dict:
    return {"model": model, "system": FAITHFULNESS_SYSTEM,
            "user": render_faithfulness_user(original, rewritten), "schema": FAITHFULNESS_SCHEMA, "tag": tag}


def parse_llm_check(res: dict) -> dict:
    if not (res.get("ok") and res.get("data")):
        return {"pass": False, "unsupported": [], "claims": [], "stuffing_flag": None, "error": res.get("error")}
    d = res["data"]
    claims = d.get("claims", [])
    unsupported = [c.get("text", "") for c in claims if c.get("verdict") == "UNSUPPORTED"]
    return {"pass": bool(d.get("overall_pass")) and not unsupported, "unsupported": unsupported,
            "claims": claims, "stuffing_flag": bool(d.get("stuffing_flag")), "error": None}


def original_text_blob(fields: dict) -> str:
    return " ".join(str(fields.get(k, "")) for k in ["title", "brand", "color", "bullets_full", "description_full"])
