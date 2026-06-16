# TaxoDrivenKG adaptation for XQuality

Minimal adaptation of TaxoDrivenKG for NeoOLAF preprocessing JSON (`translated_text`) and an OWL ontology.

Original repository and paper:
- https://github.com/Jo-Pan/TaxoDrivenKG

This version keeps the same general ideas:
- token chunking
- few-shot extraction prompt
- entity / relation tuple parsing
- taxonomy-guided candidate injection

Main difference:
- input comes from a NeoOLAF layer00 state JSON file
- taxonomy candidates come from the OWL ontology labels and local names
- supports multiple backends: `vllm`, `openrouter`, `openai`, `ollama`
