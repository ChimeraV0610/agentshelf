# AgentShelf

UCLA Anderson MSBA Hackathon · track "When the customer is a bot"

AgentShelf shows small sellers and brands how AI shopping agents see their products. Using real Amazon
search intents (ESCI, US) and a panel of simulated LLM shopping agents it

1. measures how often an agent buys the wrong product (**wrong-buy rate**) and which brands it favours,
2. diagnoses a single product: can the agent find it, where does it rank, does it get picked, and why not,
3. rewrites the listing with a faithfulness audit, then proves the lift on held-out queries with negative controls.

Full plan (Chinese): [AgentShelf_Hackathon_Plan.md](AgentShelf_Hackathon_Plan.md)

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
