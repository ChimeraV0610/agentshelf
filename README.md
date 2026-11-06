# AgentShelf

UCLA Anderson MSBA Hackathon · track "When the customer is a bot"

AgentShelf shows small sellers and brands how AI shopping agents see their products. Using real Amazon
search intents (ESCI, US) and a panel of simulated LLM shopping agents it

1. measures how often an agent buys the wrong product (**wrong-buy rate**) and which brands it favours,
2. diagnoses a single product: can the agent find it, where does it rank, does it get picked, and why not,
3. rewrites the listing with a faithfulness audit, then proves the lift on held-out queries with negative controls.

Full plan (Chinese): [AgentShelf_Hackathon_Plan.md](AgentShelf_Hackathon_Plan.md)

## Live demo (the product)

```bash
make setup && cp .env.example .env
bash scripts/00_download.sh && make data && make index      # data + BM25 index (dense is optional)
ollama serve & ollama pull qwen2.5:7b                        # local agent, fully offline
python scripts/11_build_shelves.py && python scripts/12_relevance_analysis.py
python scripts/engine_cli.py --all --buyers 6 --rewrite      # pre-warm the cache before the pitch
make app                                                     # Streamlit, 4 steps: audit -> buyers -> rewrite -> gaps
```

Paste an Amazon link or ASIN (or pick a shelf product, or paste text). The listing, 7 real competitors
(ESCI Exact products for the same search, else BM25 neighbours among 1.2M products) and the simulated buyers
are derived automatically. Without Ollama or an API key the app falls back to a transparent heuristic agent.

## Setup

```bash
make setup                      # python venv + requirements
cp .env.example .env            # add ANTHROPIC_API_KEY (and/or OPENAI_API_KEY)
bash scripts/00_download.sh     # ESCI parquet files -> data/raw/
```

## Pipeline

Everything below is one-time, cached, and read by the app from `results/` and `cache/`.

| step | command | output |
|---|---|---|
| data | `make data` | `data/processed/{examples_us,products_us,corpus}.parquet`, `reports/data_profile.md` |
| BM25 index | `.venv/bin/python scripts/02_build_indices.py --skip-dense` | `indices/bm25` (dense index is optional and not used by the live flow) |
| retrieval sanity | BM25 nDCG inside each test query's candidates: see `results/retrieval_eval.json` (0.824 vs random 0.747) | |
| shelves | `python scripts/11_build_shelves.py` | `data/shelves.json`: 6 real ESCI shelves, 8 Exact products each |
| evidence | `python scripts/12_relevance_analysis.py` | `results/relevance_analysis.json` (query-term coverage vs human Exact label) |
| pre-warm / multi-shelf lift | `python scripts/engine_cli.py --all --buyers 6 --rewrite` | cached simulations for every shelf; summary via `scripts/13_live_summary.py` |
| agent accuracy (optional) | `make tag sample` then `python scripts/06_run_agents.py --model ollama:qwen2.5:7b --limit 200` and `python scripts/07_agent_eval.py` | `results/agent_eval.json`: wrong-buy rate, macro F1, confusion matrix vs ESCI labels |
| demo | `make app` | Streamlit: Home (4-step live flow) + Evidence |

All LLM calls go through `src/llm/client.py`: Ollama (`ollama:<model>`), or Anthropic / OpenAI if a key is set.
Every call is cached in `cache/llm_cache.sqlite` by (model, prompt, schema, tag); `results/usage.jsonl` tracks spend.
`requirements.lock` pins the exact versions that were tested.

## Data

[amazon-science/esci-data](https://github.com/amazon-science/esci-data), `small_version==1`, `product_locale=='us'`:
29,844 queries, 601,354 judgements, 1,215,854 US products. Labels: E 43% / S 35% / I 17% / C 5%.

## Layout

```
configs/default.yaml   models, paths, sample sizes, gains
src/data               loading + cleaning
src/eval               nDCG (E=1, S=0.1, C=0.01, I=0)
src/retrieval          bm25s index, bge-small + FAISS, RRF; single-doc rescoring for rewrites
src/sampling           department tagging, strata, stratified sampling
src/llm                provider-agnostic client (Anthropic / OpenAI), sqlite cache, budget cap, mock client
src/agent              shopping-agent prompt, JSON schemas, judge, runner
src/diagnose           product selection, shelf diagnosis
src/rewrite            dept profiles (train split only), rewriter, faithfulness (rules + LLM), stuffed control, rescoring
src/analysis           module-1 metrics, before/after statistics
scripts/               numbered pipeline steps
app/                   Streamlit demo (3 pages)
```
