"""JSON schemas for all LLM calls. Kept to the subset of JSON Schema that both
Anthropic structured outputs and OpenAI json_schema accept (no pattern/min/max);
those constraints are enforced in code instead."""

REASON_CODES = [
    "best_match", "missing_spec", "spec_mismatch", "wrong_product_type", "unclear_title",
    "accessory_or_part", "quantity_or_pack_mismatch", "brand_mismatch", "other",
]

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["judgments", "pick", "pick_reason"],
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "label", "confidence", "missing_attributes", "reason_code"],
                "properties": {
                    "id": {"type": "string", "description": "candidate id such as c01"},
                    "label": {"type": "string", "enum": ["E", "S", "C", "I"]},
                    "confidence": {"type": "number", "description": "0 to 1"},
                    "missing_attributes": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Query requirements the listing does not confirm (at most 4), e.g. 'size 10'",
                    },
                    "reason_code": {"type": "string", "enum": REASON_CODES},
                },
            },
        },
        "pick": {"anyOf": [{"type": "string"}, {"type": "null"}],
                 "description": "candidate id to add to cart, or null to ask the customer"},
        "pick_reason": {"type": "string", "description": "one or two sentences"},
    },
}

DEPARTMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query_id", "department"],
                "properties": {
                    "query_id": {"type": "integer"},
                    "department": {"type": "string"},
                },
            },
        }
    },
}

REWRITE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "bullets", "attributes_not_available"],
    "properties": {
        "title": {"type": "string"},
        "bullets": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "evidence"],
                "properties": {
                    "text": {"type": "string"},
                    "evidence": {"type": "string", "description": "exact span of the original listing"},
                },
            },
        },
        "attributes_not_available": {
            "type": "array", "items": {"type": "string"},
            "description": "attributes shoppers in this department ask for that the original listing does not state",
        },
    },
}

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims", "overall_pass", "stuffing_flag"],
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "verdict", "evidence"],
                "properties": {
                    "text": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["SUPPORTED", "UNSUPPORTED", "UNVERIFIABLE"]},
                    "evidence": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
            },
        },
        "overall_pass": {"type": "boolean"},
        "stuffing_flag": {"type": "boolean", "description": "true if the rewrite reads like keyword stuffing"},
    },
}

DEPT_PROFILE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["profile"],
    "properties": {"profile": {"type": "string", "description": "<=150 words"}},
}
