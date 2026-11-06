"""Deliberately 'cheating' rewrite: stuff the product's own E-query terms into the listing.

This is the negative control. It uses the product's queries on purpose, so it is
exactly what a faithful rewriter must NOT do.
"""
from __future__ import annotations

import re
from collections import Counter

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the for with and or of to in on at by from is are be my me your our this that these those it its "
            "new best top cheap free size color colour pack set men women mens womens kids".split())


def stuffed_listing(fields: dict, e_queries: list[str], n_terms: int = 8, n_bullets: int = 5) -> dict:
    title_words = set(_WORD.findall(fields["title"].lower()))
    counts = Counter(w for q in e_queries for w in _WORD.findall(q.lower()) if w not in _STOP and w not in title_words)
    extra = [w for w, _ in counts.most_common(n_terms)]
    title = (fields["title"] + " " + " ".join(extra)).strip()
    bullets = list(fields["bullets_list"][: n_bullets - 1])
    bullets.append("Perfect for: " + ", ".join(dict.fromkeys(e_queries)) + ". " + " ".join(extra) + " " + " ".join(extra))
    while len(bullets) < n_bullets:
        bullets.append("Great choice: " + " ".join(extra))
    return {"title": title, "bullets": bullets}
