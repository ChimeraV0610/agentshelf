"""Department-level 'what shoppers specify' profiles, built from TRAIN-split queries only.

The profile is the only query-derived signal the rewriter ever sees, and it never
includes queries linked to a diagnosed product.
"""
from __future__ import annotations

import pandas as pd

from src.agent.prompts import DEPT_PROFILE_SYSTEM
from src.agent.schema import DEPT_PROFILE_SCHEMA
from src.config import CFG

STATIC_PROFILES = {
    "Electronics": "device or model compatibility, connector type (USB-C, HDMI), capacity or storage size, wireless standard, power or wattage, screen size, pack quantity.",
    "Home & Kitchen": "dimensions and capacity, material, set or pack count, color, compatible appliance or bed size, dishwasher or oven safety, intended room.",
    "Clothing, Shoes & Jewelry": "size and fit, gender or age group, color, material or fabric, style (e.g. high waisted, slip on), occasion, pack count.",
    "Beauty & Personal Care": "skin or hair type, shade or color, volume or size, key ingredient or free-of claims, scent, intended use.",
    "Health & Household": "dosage or strength, count or supply length, form (gummy, capsule), dietary claims, pack size, intended user.",
    "Grocery": "flavor, package size or weight, count, dietary claims (organic, sugar free, gluten free), brand or variety, caffeine.",
    "Toys & Games": "age range, character or franchise, number of pieces or players, battery requirements, size, educational purpose.",
    "Sports & Outdoors": "size or weight capacity, sport or activity, material, gender or age, indoor vs outdoor use, set contents.",
    "Tools, Automotive & Garden": "vehicle make, model and year fit, voltage or power source, size or thread, material, quantity, indoor or outdoor rating.",
    "Office, Books & Media": "format or edition, size (letter, A4), count, color, author or series, compatible printer model.",
    "Pet Supplies": "animal and breed size, age stage, flavor, weight or count, size (S/M/L), material, intended use.",
    "Baby & Other": "age or weight range, size, count, material or safety certification, gender, compatibility with stroller or crib.",
}


def build_profiles(train_examples: pd.DataFrame, depts: pd.DataFrame, exclude_qids: set, model: str,
                   client=None, n_per_dept: int = 150, seed: int = 42) -> dict[str, str]:
    """LLM-written profiles per department; falls back to STATIC_PROFILES when client is None."""
    if client is None:
        return dict(STATIC_PROFILES)
    q = train_examples.drop_duplicates("query_id")[["query_id", "query"]]
    q = q[~q["query_id"].isin(exclude_qids)].merge(depts[["query_id", "department"]], on="query_id", how="inner")
    jobs, order = [], []
    for dept in CFG["departments"]:
        sub = q[q["department"] == dept]
        if sub.empty:
            continue
        sample = sub.sample(n=min(n_per_dept, len(sub)), random_state=seed)["query"].tolist()
        jobs.append({"model": model, "system": DEPT_PROFILE_SYSTEM,
                     "user": f"Department: {dept}\n\nQueries:\n" + "\n".join(f"- {s}" for s in sample),
                     "schema": DEPT_PROFILE_SCHEMA, "tag": "dept_profile"})
        order.append(dept)
    results = client.map(jobs, desc="department profiles")
    out = dict(STATIC_PROFILES)
    for dept, res in zip(order, results):
        if res.get("ok") and res.get("data", {}).get("profile"):
            out[dept] = res["data"]["profile"].strip()
    return out
