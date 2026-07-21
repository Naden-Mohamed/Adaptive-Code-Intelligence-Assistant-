# Adaptive Code Intelligence Assistant

An AI-powered programming assistant that automatically routes between **code explanation** and **code generation**, backed by a self-improving Retrieval-Augmented Generation (RAG) pipeline. Built with LangChain, Streamlit, and a vector database (Chroma).

The system doesn't just answer coding questions — it decides *how* to answer them (retrieve vs. generate from scratch), checks its own retrieval quality before trusting it, and feeds every new solution back into its knowledge base so it gets better with use.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Design Decisions](#design-decisions)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Overview

When a user submits a query, an **LLM Task Router** classifies it as either:

1. **Code Explanation** — routed directly to the LLM, bypassing retrieval entirely for lower latency.
2. **Code Generation** — routed through a RAG pipeline that searches a vector database of code examples, documentation, and past solutions.

For generation requests, a second LLM acts as a **Retrieval Evaluator**, judging whether the retrieved context is actually relevant before it's used. If it's relevant, the system generates code grounded in that context. If not, the context is discarded and the LLM generates from its own knowledge instead. Either way, the output includes both the code and a structured explanation.

Newly generated solutions that weren't backed by useful retrieval are automatically chunked, embedded, and inserted back into the vector database — a self-learning loop that reduces future retrieval failures without manual curation.

---

## Architecture

```
User Query
    │
    ▼
LLM Task Router ──────────────┬─────────────────────┐
    │                         │                      
Code Explanation        Code Generation
    │                         │
Direct to LLM          Search Vector DB
    │                         │
    │                  LLM Retrieval Evaluator
    │                    ┌────┴────┐
    │                Relevant   Not Relevant
    │                    │          │
    │              RAG Generation  Generate from LLM
    │                    └────┬────┘
    │                         ▼
    │              Generated Code + Explanation
    │                         │
    │           (if generated without useful retrieval)
    │                         ▼
    │              Chunk → Embed → Add to Vector DB
    │                         │
    └─────────────────────────┴──► Display in Streamlit
                                          │
                                          ▼
                              Optional: Execute Generated Code


## Key Features

- **Intent-based routing** — no manual mode switching; the LLM decides the workflow.
- **Retrieval quality gating** — an LLM-as-judge evaluator prevents irrelevant context from polluting generations.
- **Self-learning knowledge base** — every novel solution expands the vector database automatically, with deduplication to prevent quality drift.
- **Consistent output schema** — both the RAG and non-RAG generation paths return the same structured object (`code`, `explanation`, `language`, `dependencies`), so downstream UI logic doesn't need to know which path was taken.
- **Transparent decision trail** — the UI surfaces which route was taken and whether retrieved context was accepted or rejected.
- **(Bonus) Sandboxed code execution** — run generated code in an isolated environment and view stdout, errors, and execution time directly in the app.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain |
| LLM | Gorq API (configurable) |
| Vector Database | Chroma |
| Embeddings | `BAAI/bge-m3` (configurable) |
| UI | Streamlit |
| Sandbox Execution | Isolated subprocess |
| Seed Dataset | [OpenAI HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) |

---

---

## Getting Started

### Prerequisites

- Python 3.10+
- An Groq API key (or equivalent LLM provider key)
- ~1GB disk space for the local vector store and seed dataset

### Installation

```bash
git clone https://github.com/Naden-Mohamed/Adaptive-Code-Intelligence-Assistant-.git
cd Adaptive-Code-Intelligence-Assistant
```
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```
```bash
cp .env.example .env          # then fill in your API key
```

### Seed the Vector Database

```bash
python vectorstore/preprocessor.py
```

This loads the HumanEval dataset, chunks problem/solution pairs, embeds them, and builds the initial index.

### Run the App

```bash
streamlit run app/main.py
```

## Usage

1. Open the Streamlit app.
2. Ask a question — e.g. *"Write a Python function to merge two sorted linked lists"* or paste code and ask *"Explain what this does."*
3. The app displays:
   - The route taken (explanation vs. generation)
   - Whether retrieval was used, and the evaluator's relevance verdict
   - The final code and explanation
4. Optionally, click **Run Code** to execute the generated solution in a sandbox and view the output.

---

## Evaluation

The `evaluation/` directory contains a scripted harness (not manual testing) covering:

- **Router accuracy** — classification accuracy against a labeled test set of explanation vs. generation prompts.
- **Retrieval quality** — agreement between the LLM evaluator's verdicts and human-labeled relevance judgments on a sample set.
- **Functional correctness** — pass@1 on a held-out subset of HumanEval problems, exercising the full generation pipeline end-to-end.

Run it with:

```bash
python evaluation/run_eval.py
```

Results are written to `evaluation/results.md`.

---

## Design Decisions

- **Separate evaluator LLM call vs. a similarity-score threshold** — a raw cosine similarity score doesn't capture semantic relevance well enough for code retrieval; an LLM judge with a clear rubric is more reliable, at the cost of one extra API call.
- **Shared output schema across RAG and non-RAG generation** — keeps the UI and ingestion logic agnostic to which path produced the result.
- **Deduplication before ingestion** — without a similarity gate, the self-learning loop would slowly fill the vector DB with near-identical entries and degrade retrieval quality over time.

---

## Known Limitations

- Two additional LLM calls per generation request (router + evaluator) increase latency and cost versus a single-call system — mitigated by using a smaller/cheaper model for routing and evaluation.
- Sandboxed execution blocks network access and restricts file system writes, but is not a substitute for a fully isolated container in a production/multi-tenant setting.
- The self-learning loop currently has no manual review step before insertion — acceptable for a personal knowledge base, but would need a moderation step before use in a shared/production environment.

## License

Apache License — see [`LICENSE`](./LICENSE) for details.
