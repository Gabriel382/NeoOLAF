# OLAF isolated baseline sandbox

This folder is designed to live at:

`NeoOLAF/examples/RAGTreeDatasets/olaf_baseline`

It is fully isolated from the NeoOLAF environment and uses its own `.venv`.

## Why this design

The upstream OLAF repository is now archived/read-only. The exact OLAF source supplied for this experiment is vendored under `vendor/olaf`, so future upstream changes cannot alter the baseline.

The upstream OLAF `OpenAIGenerator` hard-codes `gpt-3.5-turbo` and the OpenAI endpoint. This sandbox does not edit OLAF itself. Instead, `src/openrouter_generator.py` implements OLAF's `LLMGenerator` interface and calls OpenRouter with `openai/gpt-oss-20b`.

The first smoke uses a deliberately cheap **OLAF-lite** configuration:
1. POS candidate extraction
2. LLM concept grouping
3. POS relation-term extraction
4. LLM relation grouping

That is only about **2 LLM calls per document**. Hierarchisation and OWL axiom generation are omitted for the benchmark baseline because they add cost but do not directly help document-level relation scoring.

## Windows setup

From PowerShell:

```powershell
cd C:\Users\galencarmedeiro\NeoOLAF\examples\RAGTreeDatasets\olaf_baseline
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

Then select the Jupyter kernel:

`OLAF Baseline (.venv)`

Create a `.env` from `.env.example` or set:

```powershell
$env:OPENROUTER_API_KEY="..."
$env:OLAF_OPENROUTER_MODEL="openai/gpt-oss-20b"
$env:OLAF_REASONING_EFFORT="minimal"
```

## First experiment

Run `notebooks/00_smoke_one_doc_each.ipynb`.

It:
- finds the RAGTree preprocessed dataset files;
- chooses one positive-gold document from DocRED, FinCausal, and EventStoryLine;
- strips gold before OLAF sees the text;
- runs one document at a time;
- saves raw OLAF concepts/relations and elapsed time under `runs/smoke_one_each`.

This first smoke is **not yet the final benchmark evaluation**. OLAF is fundamentally an ontology-learning framework, while these datasets are document-level relation-extraction benchmarks. We should inspect the native OLAF relation objects first, then freeze one transparent projection/evaluation adapter before spending money on full corpora.


## spaCy model preflight

The smoke notebook now checks `en_core_web_sm` before any OpenRouter call.
If it is missing, it installs the model using the notebook kernel's own
`.venv\Scripts\python.exe`, verifies `spacy.load("en_core_web_sm")`, and only
then allows the paid OLAF smoke to run.
