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

| step | command | output |
|---|---|---|
| 0 data | `make data` | `data/processed/{examples_us,products_us,corpus}.parquet`, `reports/data_profile.md` |
| 2 indices | `make index` | `indices/bm25`, `indices/dense` (bge-small-en-v1.5 + FAISS) |
| 2 retrieval eval | `make eval-retrieval` | `results/retrieval_eval.json`, `results/full_corpus_ranks.parquet` |
| 3 tag + sample | `make tag sample` | `data/samples/queries_1000.parquet`, `candidates_1000.parquet` |
| 4 agents | `make agents-sanity` then `make agents` | `results/agent_runs/*.parquet` (every call cached in `cache/llm_cache.sqlite`) |
| 5 agent eval | `make agent-eval` | `results/agent_eval.json` + tables |
| 6 diagnose | `make diagnose` | `results/diagnostics/*.json`, `diagnostics_summary.parquet` |
| 7 rewrite | `make rewrite` | `results/rewrite/{listings,before_after}.parquet` |
| 8 rewrite eval | `make rewrite-eval` | `results/rewrite_eval.json`, `headline_table.csv` |
| demo | `make app` | Streamlit, reads `results/` only |

### Local models with Ollama (no API key)

```bash
brew install ollama && ollama serve &          # once
ollama pull qwen2.5:7b                         # ~4.7 GB; qwen2.5:3b is ~2x faster, weaker
python scripts/06_run_agents.py --model ollama:qwen2.5:7b --limit 20   # measure speed first
```

Any `--model` (or the `models:` block in `configs/default.yaml`) accepts `ollama:<name>`.
The client sets `num_ctx` to 16k (Ollama's default context would silently truncate 20-30 candidates)
and runs local calls with concurrency 2. Budget: a 7B model on an M-series laptop handles roughly
2-3 queries per minute for the listwise judge, so plan the sample size accordingly (e.g. 300 queries
for the local agent, or use it as the second panel member on a subset).

Offline plumbing test without any API key: prefix any command with `AGENTSHELF_MOCK=1`.
Outputs then go to `results_mock/`, `data/samples_mock/` and a separate cache, and the app shows a mock banner.

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
