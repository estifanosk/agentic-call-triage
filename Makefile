PY := .venv/bin/python
TRIAGE := .venv/bin/triage

.PHONY: help setup test evals demo demo-clean demo-injection pause java clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv and install the package
	uv venv --python 3.13 .venv
	uv pip install --python $(PY) -e ".[dev]"

test:  ## Behavioural tests (deterministic, no model required)
	.venv/bin/pytest tests/test_graph.py -q

evals:  ## Outcome evals against the golden set
	.venv/bin/pytest tests/test_evals.py -q

evals-live:  ## Same evals, scored against the real local model
	TRIAGE_PROVIDER=ollama .venv/bin/pytest tests/test_evals.py -q

java:  ## Start the Java case-management service on :8080
	java java/CaseService.java

demo:  ## Full run on the violation-heavy call, real model
	$(TRIAGE) --provider ollama run --call CALL-10041

demo-clean:  ## Run on the compliant call - should produce no findings
	$(TRIAGE) --provider ollama run --call CALL-10042

demo-injection:  ## Run on the transcript containing a prompt-injection payload
	$(TRIAGE) --provider ollama run --call CALL-10043

pause:  ## Suspend at the human gate so a second process can resume it
	$(TRIAGE) --provider ollama run --call CALL-10041 --thread demo --auto pause

fast:  ## Same trajectory with the stub backend, instant
	$(TRIAGE) --provider stub run --call CALL-10041 --auto approve

clean:
	rm -f triage.db
