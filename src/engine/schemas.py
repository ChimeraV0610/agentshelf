"""JSON schemas for the live engine (simulate + rewrite). Subset of JSON Schema that Ollama/Anthropic/OpenAI accept."""

SIMULATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ranking", "pick", "pick_reason"],
    "properties": {
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "rank", "reason", "missing_facts"],
                "properties": {
                    "id": {"type": "string", "description": "candidate id such as c01"},
                    "rank": {"type": "integer", "description": "1 = best fit for this customer"},
                    "reason": {"type": "string", "description": "one sentence: why this rank, citing listing facts"},
                    "missing_facts": {"type": "array", "items": {"type": "string"},
                                      "description": "facts the customer needs that this listing does not state (at most 2)"},
                },
            },
        },
        "pick": {"type": "string", "description": "candidate id to buy"},
        "pick_reason": {"type": "string"},
    },
}

REWRITE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "bullets", "attributes", "seller_to_confirm"],
    "properties": {
        "title": {"type": "string"},
        "bullets": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False, "required": ["name", "fact", "evidence"],
                      "properties": {"name": {"type": "string"}, "fact": {"type": "string"},
                                     "evidence": {"type": "string", "description": "exact span of the original listing"}}},
        },
        "attributes": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False, "required": ["name", "value", "evidence"],
                      "properties": {"name": {"type": "string"}, "value": {"type": "string"}, "evidence": {"type": "string"}}},
        },
        "seller_to_confirm": {"type": "array", "items": {"type": "string"},
                              "description": "facts agents ask for that the original listing does not state"},
    },
}
