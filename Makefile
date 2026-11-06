PY := .venv/bin/python

.PHONY: setup data index eval-retrieval tag sample agents agent-eval diagnose rewrite rewrite-eval app all

setup:
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt

data:
	$(PY) scripts/01_build_corpus.py

index:
	$(PY) scripts/02_build_indices.py

eval-retrieval:
	$(PY) scripts/03_eval_retrieval.py

tag:
	$(PY) scripts/04_tag_departments.py --split test

sample:
	$(PY) scripts/05_sample_queries.py

agents:
	$(PY) scripts/06_run_agents.py --model claude-sonnet-5-5
	$(PY) scripts/06_run_agents.py --model claude-haiku-4-5

agents-sanity:
	$(PY) scripts/06_run_agents.py --model claude-sonnet-5-5 --limit 50

agent-eval:
	$(PY) scripts/07_agent_eval.py

diagnose:
	$(PY) scripts/08_diagnose.py

rewrite:
	$(PY) scripts/09_rewrite.py

rewrite-eval:
	$(PY) scripts/10_rewrite_eval.py

app:
	.venv/bin/streamlit run app/Home.py

all: data index eval-retrieval tag sample agents agent-eval diagnose rewrite rewrite-eval
