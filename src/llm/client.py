"""Provider-agnostic LLM client with sqlite cache, concurrency limit, usage log and budget cap.

Models starting with "claude" go to the Anthropic SDK (structured outputs via
output_config.format); models starting with "gpt" go to the OpenAI SDK
(response_format json_schema). Every call is cached by a hash of
(model, system, user, schema, tag), so re-running any script is free and the
Streamlit demo never touches the network.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import CFG, ROOT, path

load_dotenv(ROOT / ".env")


class BudgetExceeded(RuntimeError):
    pass


class _Retryable(Exception):
    pass


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    prices = CFG["llm"]["prices"].get(model)
    if not prices:
        return 0.0
    return in_tok / 1e6 * prices[0] + out_tok / 1e6 * prices[1]


class LLMClient:
    def __init__(self, cache_path: Path | None = None, usage_log: Path | None = None,
                 budget_usd: float | None = None, concurrency: int | None = None):
        self.cache_path = cache_path or path("cache")
        self.usage_log = usage_log or path("usage_log")
        self.budget = budget_usd if budget_usd is not None else CFG["llm"]["budget_usd"]
        self.sem = asyncio.Semaphore(concurrency or CFG["llm"]["concurrency"])
        self.db = sqlite3.connect(self.cache_path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT, "
                        "response TEXT, usage TEXT, created REAL)")
        self.db.commit()
        self._anthropic = None
        self._openai = None
        self.spent = self._spent_so_far()
        self.stats = {"cached": 0, "calls": 0, "errors": 0}

    # ---------- bookkeeping ----------
    def _spent_so_far(self) -> float:
        if not self.usage_log.exists():
            return 0.0
        total = 0.0
        with open(self.usage_log) as f:
            for line in f:
                try:
                    total += json.loads(line).get("cost", 0.0)
                except json.JSONDecodeError:
                    pass
        return total

    def _log_usage(self, model: str, tag: str, in_tok: int, out_tok: int, cost: float):
        with open(self.usage_log, "a") as f:
            f.write(json.dumps({"ts": time.time(), "model": model, "tag": tag, "in": in_tok,
                                "out": out_tok, "cost": round(cost, 6)}) + "\n")

    @staticmethod
    def key(model: str, system: str, user: str, schema: dict | None, tag: str = "") -> str:
        payload = json.dumps([model, system, user, schema, tag], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _get(self, key: str):
        row = self.db.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, key: str, model: str, response: dict, usage: dict):
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)",
                        (key, model, json.dumps(response, ensure_ascii=False), json.dumps(usage), time.time()))
        self.db.commit()

    # ---------- providers ----------
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60),
           retry=retry_if_exception_type(_Retryable), reraise=True)
    async def _call_anthropic(self, model, system, user, schema, max_tokens):
        import anthropic

        if self._anthropic is None:
            self._anthropic = anthropic.AsyncAnthropic()
        output_config: dict[str, Any] = {}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        effort = CFG["llm"]["effort"].get(model)
        if effort:
            output_config["effort"] = effort
        kwargs: dict[str, Any] = dict(model=model, max_tokens=max_tokens, system=system,
                                      messages=[{"role": "user", "content": user}])
        if output_config:
            kwargs["output_config"] = output_config
        try:
            resp = await self._anthropic.messages.create(**kwargs)
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            raise _Retryable(str(e)) from e
        usage = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
        if resp.stop_reason == "refusal":
            return {"ok": False, "error": "refusal", "raw": ""}, usage
        text = next((b.text for b in resp.content if b.type == "text"), "")
        if resp.stop_reason == "max_tokens":
            return {"ok": False, "error": "max_tokens", "raw": text}, usage
        return {"ok": True, "raw": text}, usage

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60),
           retry=retry_if_exception_type(_Retryable), reraise=True)
    async def _call_openai(self, model, system, user, schema, max_tokens):
        import openai

        if self._openai is None:
            self._openai = openai.AsyncOpenAI()
        kwargs: dict[str, Any] = dict(model=model, max_tokens=max_tokens, temperature=0,
                                      messages=[{"role": "system", "content": system},
                                                {"role": "user", "content": user}])
        if schema is not None:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "out", "schema": schema, "strict": False}}
        try:
            resp = await self._openai.chat.completions.create(**kwargs)
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as e:
            raise _Retryable(str(e)) from e
        usage = {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens}
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            return {"ok": False, "error": "max_tokens", "raw": choice.message.content or ""}, usage
        return {"ok": True, "raw": choice.message.content or ""}, usage

    # ---------- public API ----------
    async def acall(self, model: str, system: str, user: str, schema: dict | None = None,
                    tag: str = "", max_tokens: int | None = None) -> dict:
        """Returns {'ok', 'data' (parsed JSON or None), 'error', 'cached'}."""
        key = self.key(model, system, user, schema, tag)
        cached = self._get(key)
        if cached is not None:
            self.stats["cached"] += 1
            return {**cached, "cached": True}
        if self.spent >= self.budget:
            raise BudgetExceeded(f"spent ${self.spent:.2f} >= budget ${self.budget:.2f}")
        max_tokens = max_tokens or CFG["llm"]["max_tokens"]
        async with self.sem:
            if model.startswith("claude"):
                out, usage = await self._call_anthropic(model, system, user, schema, max_tokens)
            elif model.startswith("gpt"):
                out, usage = await self._call_openai(model, system, user, schema, max_tokens)
            else:
                raise ValueError(f"unknown provider for model {model}")
        cost = _cost(model, usage["in"], usage["out"])
        self.spent += cost
        self.stats["calls"] += 1
        self._log_usage(model, tag, usage["in"], usage["out"], cost)
        result: dict[str, Any] = {"ok": out["ok"], "error": out.get("error"), "data": None, "raw": out["raw"]}
        if out["ok"]:
            try:
                result["data"] = json.loads(out["raw"])
            except json.JSONDecodeError:
                result["ok"] = False
                result["error"] = "parse_error"
        if not result["ok"]:
            self.stats["errors"] += 1
        # cache even failures (refusal / parse_error) so reruns do not re-pay; delete the row to retry
        self._put(key, model, result, usage)
        return {**result, "cached": False}

    def call(self, *args, **kwargs) -> dict:
        return asyncio.run(self.acall(*args, **kwargs))

    async def amap(self, jobs: list[dict], desc: str = "llm") -> list[dict]:
        """jobs: list of kwargs for acall. Preserves order. Shows a progress bar."""
        from tqdm.asyncio import tqdm_asyncio

        async def one(j):
            try:
                return await self.acall(**j)
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001 - record and continue
                self.stats["errors"] += 1
                return {"ok": False, "error": f"exception: {type(e).__name__}: {e}", "data": None, "cached": False}

        return await tqdm_asyncio.gather(*[one(j) for j in jobs], desc=desc)

    def map(self, jobs: list[dict], desc: str = "llm") -> list[dict]:
        return asyncio.run(self.amap(jobs, desc=desc))

    def summary(self) -> str:
        return (f"calls={self.stats['calls']} cached={self.stats['cached']} errors={self.stats['errors']} "
                f"spent_total=${self.spent:.2f} (budget ${self.budget:.0f})")


def has_api_key(model: str) -> bool:
    if model.startswith("claude"):
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if model.startswith("gpt"):
        return bool(os.environ.get("OPENAI_API_KEY"))
    return False


def get_client(**kwargs):
    """LLMClient, or MockClient when AGENTSHELF_MOCK=1 (offline plumbing test)."""
    if os.environ.get("AGENTSHELF_MOCK") == "1":
        from src.llm.mock import MockClient

        return MockClient()
    return LLMClient(**kwargs)
