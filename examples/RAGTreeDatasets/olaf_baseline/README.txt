OLAF full 3-dataset parallel-5 NO-CEILING patch

Replace:
  <olaf_baseline>\src\openrouter_generator.py
with:
  src\openrouter_generator.py

Add:
  src\all_datasets_budget_runner.py

Add:
  notebooks\02_all_datasets_full_parallel5_no_ceiling.ipynb

Then restart/select:
  OLAF Baseline (.venv)

Execution:
  - DocRED: 998 exact dev documents
  - FinCausal: 967 documents
  - EventStoryLine: 443 documents
  - Total: 2408 documents
  - DOCUMENT_WORKERS = 5
  - Model: openai/gpt-oss-20b
  - Reasoning: minimal
  - NO monetary ceiling
  - NO budget-triggered stop
  - token/cost telemetry is still recorded
  - completed documents are resumed and never intentionally re-run
  - existing paid smoke results are reused where possible

Run directory stays:
  runs\olaf_lite_full_3datasets_v1

So this notebook can also resume any partial execution started with the prior version.
