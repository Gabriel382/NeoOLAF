# 🧠 NeoOLAF

**A Grounded and Agentic Framework for Ontology Learning and Ontology-Guided Knowledge Graph Construction from Text**

NeoOLAF is a **next-generation semantic construction framework** designed to build both a **refined ontology** and a **grounded knowledge graph** from unstructured text. It extends the philosophy of **OLAF (Ontology Learning Applied Framework)** by unifying:

- **Ontology Learning (OL)**
- **Ontology-Guided Knowledge Graph Population**
- **Document-Level Validation and Reasoning**
- **Semantic RAG-Based Justification**
- **Layer-Centered Agentic Orchestration**

NeoOLAF is particularly designed for **technical and industrial corpora**, where documents contain operational, contextual, and causal knowledge that must be structured into explainable semantic artifacts.

---

## 🔍 Why NeoOLAF?

Classical ontology learning pipelines can structure text into concepts, relations, hierarchies, and axioms, but they usually stop short of building a fully grounded and validated factual graph.

Conversely, knowledge graph construction pipelines can populate graphs from text, but they often assume a fixed ontology and do not evolve the schema in a controlled way.

In realistic industrial settings, this creates three major problems:

1. **Semantic construction from text remains difficult**  
   Technical documents are heterogeneous, implicit, and context-heavy.

2. **Ontology learning alone is not enough**  
   Schema induction does not fully solve factual graph construction.

3. **KG construction alone is not enough**  
   Graph population without controlled ontology evolution remains brittle.

NeoOLAF addresses these problems by introducing a **shared candidate semantic layer** and an **iterative loop** in which ontology refinement and graph construction continuously inform one another.

---

## 🎯 Core Objective

NeoOLAF aims to move from raw industrial documents to:

- a **refined ontology**
- a **grounded knowledge graph**
- and eventually **explainable causal knowledge**

This makes it particularly relevant for:

- ontology-guided retrieval
- root cause analysis
- semantic indexing
- industrial explainability
- knowledge graph construction from technical documents
- downstream RAG and KG-RAG systems

---

## 🧬 Core Principles

NeoOLAF is built around six principles:

### 1. Unified Ontology Learning and KG Construction
Ontology learning and knowledge graph population are treated as part of the same semantic construction loop.

### 2. Document as the Main Semantic Unit
The framework processes each document as a coherent semantic object. Chunking is used only internally when needed.

### 3. Shared Candidate Semantic Layer
Linguistic expressions become candidate semantic objects before being promoted into ontology elements or graph assertions.

### 4. Grounding Through Semantic RAG
Every important semantic transformation can be justified through retrieval over:
- source documents
- seed ontology
- external knowledge resources
- artifacts produced by previous layers

### 5. Document-Level Validation and Reasoning
Each document produces explicit intermediate artifacts that can be validated, reasoned over, and serialized.

### 6. Agentic Semantic Construction
Each semantic layer can be handled by a specialized agent with its own prompt, retrieval space, and validation role.

---

## 🏗️ High-Level Architecture

NeoOLAF follows an iterative, layer-centered pipeline:

```text
Text Corpus
   ↓
Pre-processing
   ↓
Linguistic Expression Extraction
   ↓
Candidate Enrichment
   ↓
Candidate Typing and Resolution
   ↓
Candidate Relation Extraction
   ↓
Candidate Triple Generation
   ↓
Concept / Relation Induction
   ↓
Hierarchisation
   ↓
Axiom Schemata Extraction
   ↓
General Axiom Extraction
   ↓
Validation / Reasoning
   ↓
Inference / Completion
   ↓
Refined Ontology + Grounded Knowledge Graph
````

Unlike classical ontology learning pipelines, NeoOLAF does not stop at ontology induction. It explicitly continues toward graph construction, validation, and grounded completion.

---

## 🧱 Main Semantic Objects

NeoOLAF manipulates explicit semantic artifacts throughout the pipeline:

* **Linguistic expressions**
* **Entity candidates**
* **Relation candidates**
* **Attribute/value candidates**
* **Event/state candidates**
* **Candidate triples**
* **Ontology deltas**
* **Local document ontology**
* **Local document knowledge graph**
* **Global ontology**
* **Global knowledge graph**

These artifacts are designed to remain serializable and inspectable throughout execution.

---

## 🤖 Agentic Layer

NeoOLAF introduces a **layer-centered multi-agent architecture**.

Each semantic layer may be assigned to a dedicated agent that:

* receives the current semantic artifact
* uses a layer-specific prompt
* retrieves supporting evidence
* produces a structured output artifact
* passes the result to the next layer

This differs from chunk-centered worker architectures. In NeoOLAF, the document remains the main semantic unit, while chunking is only an internal processing mechanism.

### Example agent roles

* Extraction Agent
* Enrichment Agent
* Typing Agent
* Relation Agent
* Triple Agent
* Induction Agent
* Hierarchy Agent
* Axiom Agent
* Validation Agent
* Completion Agent

---

## 🧠 Semantic Guidance

NeoOLAF can also receive **user semantic guidance**, which helps steer the modeling process.

Typical guidance includes:

* **Domain focus**
  e.g. industrial maintenance and causal failure chains

* **Abstraction level**
  e.g. treat machine types as concepts and document-specific occurrences as individuals

* **Priority relations**
  e.g. causal, part-of, affects, observed-by, temporal

* **Population policy**
  e.g. promote a candidate to concept only if it is recurrent

* **Event modeling preference**
  e.g. treat failures and shutdowns as events/states rather than simple entities

---

## 📦 Outputs

NeoOLAF produces two primary outputs:

### 1. NeoOLAF Refined Ontology

A refined ontology that includes:

* promoted concepts
* relations
* hierarchies
* axiom schemata
* general axioms
* `rdfs:description` for ontology entities

### 2. NeoOLAF Grounded Knowledge Graph

A knowledge graph containing:

* validated triples
* provenance-aware assertions
* document-level factual structures
* inferred and completed relations

Outputs can be serialized to:

* **TTL / RDF**
* **OWL**
* **JSON**
* **Neo4j-compatible graph formats**

---

## ✅ Validation and Reasoning

NeoOLAF does not rely only on extraction quality.

It includes explicit support for:

* ontology conformance
* contradiction detection
* provenance coverage
* faithfulness
* document-level validation outcomes
* rule- and ontology-based reasoning
* completion of missing information

Intermediate artifacts can be serialized at each major stage, such as:

* candidate graph TTL
* ontology delta TTL
* validated ontology TTL
* inferred graph TTL

---

## 📊 Evaluation Modes

NeoOLAF supports two complementary evaluation regimes.

### With Gold Truth

When benchmark annotations are available, NeoOLAF can be evaluated with:

* entity precision / recall / F1
* relation precision / recall / F1
* ontology conformance metrics

### Without Full Gold Truth

In industrial use cases such as XQuality, NeoOLAF can be evaluated with:

* faithfulness
* ontology alignment
* BLEU
* provenance coverage
* contradiction rate
* validation outcomes

This dual evaluation design is one of the key methodological strengths of the framework.

---

## 🏭 Main Use Case: XQuality

NeoOLAF is currently being developed with a strong focus on the **XQuality textual branch**, where the goal is to structure technical documents into explainable semantic artifacts that support:

* quality issue identification
* underlying cause analysis
* semantic structuring of industrial knowledge
* explainable downstream reasoning

XQuality is therefore both:

* a **real-world industrial use case**
* and a **driving constraint** for NeoOLAF’s architecture

---

## 🌱 Relation to OLAF

NeoOLAF is directly inspired by **OLAF: Ontology Learning Applied Framework**, but extends it in several major ways.

### OLAF provided

* a modular ontology learning pipeline
* term extraction
* enrichment
* concept / relation extraction
* hierarchy induction
* axiom extraction

### NeoOLAF adds

* unified KG population
* candidate triples and graph outputs
* document-level validation and reasoning
* SemanticRAG-based grounding
* explicit agentic orchestration
* dual evaluation with and without gold truth

In this sense, NeoOLAF can be seen as a **grounded and agentic extension of OLAF**.

---

## 🧪 Typical Use Cases

NeoOLAF is relevant for:

* industrial knowledge extraction
* ontology-guided causal analysis
* technical document mining
* explainable root cause analysis
* ontology-guided KG construction
* semantic search
* RAG / KG-RAG pipelines
* domain bootstrapping when no ontology fully exists

---

## 🚧 Project Status

NeoOLAF is an **active research and engineering project**.

### Current status

* [x] conceptual architecture defined
* [x] layered methodology formalized
* [x] grounding strategy identified
* [x] XQuality use case defined
* [x] evaluation logic with and without gold truth formalized
* [ ] first full implementation of the unified loop
* [ ] complete integration of SemanticRAG into all targeted layers
* [ ] full benchmark and XQuality experiments
* [ ] publication-oriented experimental package

---

## 📚 Research Context

NeoOLAF builds on research in:

* ontology learning from text
* ontology-guided knowledge graph construction
* symbolic–neural hybrid systems
* Semantic RAG
* grounding-aware LLM workflows
* agentic semantic construction

It is currently aligned with:

* NeoOLAF article preparation
* XQuality-related semantic construction work
* and the RAGTree benchmark line for semantic-guided RAG evaluation

---

## 🤝 Contributing

Contributions are welcome from:

* ontology engineers
* semantic web researchers
* knowledge graph practitioners
* industrial AI researchers
* NLP / LLM researchers
* agentic systems developers

---

## 📜 License

MIT License.