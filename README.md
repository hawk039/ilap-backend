# ilap-backend
# ILAP – Indian Legal AI Platform

ILAP (Indian Legal AI Platform) is a **production-oriented legal AI backend** designed with a reliability-first mindset. Instead of free-form generation, ILAP enforces **strict source-of-truth retrieval**, **explainable proof layers**, and **hallucination-resistant architecture** to deliver trustworthy legal information.

This project is intentionally built **LLM-last**: deterministic systems first, generation only where it is safe and justified.

---

## Why ILAP

Most legal AI demos prioritize fluent answers. ILAP prioritizes **correctness, traceability, and refusal when uncertain**.

Key design principles:

* Authority over fluency
* Retrieval before generation
* Proof required for every answer
* Refusal is better than hallucination

---

## Core Features

* **Deterministic Retrieval-Augmented Pipeline (RAG)**
  Semantic retrieval over versioned Indian legal texts using vector embeddings.

* **Versioned Law Ingestion**
  Explicit ingestion pipelines for legal texts with version and source metadata.

* **Explainable Proof Layer**
  Every response includes cited sections, text snippets, and relevance scores.

* **Confidence Scoring & Refusal Logic**
  Low-confidence or unsupported queries result in safe refusal instead of speculation.

* **Hallucination-Resistant Architecture**
  No answer is generated without retrieved legal evidence.

* **Model-Agnostic AI Design (LLM-ready)**
  Generation is abstracted behind interfaces and can be added safely later.

---

## High-Level Architecture

```
Legal Texts
   ↓
Ingestion Script
   ↓
Vector Database (ChromaDB)
   ↓
Query → Retrieval → Proof → Response
```

* Ingestion is offline and deterministic
* Retrieval acts as the gatekeeper
* Response generation is constrained by proof availability

---

## Tech Stack

* **Backend**: Python, FastAPI
* **Vector Store**: ChromaDB
* **Embeddings**: Sentence Transformers (MiniLM)
* **Data Format**: Plain text (versioned)
* **API**: JSON-based, schema-validated

---

## Project Structure

```
ILAP/
├── app/                # FastAPI application
│   ├── api/            # API routes
│   ├── services/       # Retrieval & answer logic
│   └── main.py
├── config/             # Source-of-truth configuration
├── knowledge_base/     # Versioned legal texts (sample)
├── scripts/            # Ingestion scripts
├── schemas/            # Response & proof schemas
├── requirements.txt
└── README.md
```

---

## Current Capabilities

* Ingests real Indian legal text (sample scope)

* Stores semantic embeddings with structured legal metadata

* Retrieves relevant statutory sections using similarity search

* Implements intent-aware retrieval gating to prevent semantic false positives

* Returns explainable, citation-backed responses with proof excerpts

* Calibrates confidence scores based on retrieval strength and corroboration

* Safely refuses unsupported, non-legal, or low-confidence queries

* Enforces source-of-truth answering to eliminate hallucinations

* Built an evaluation harness to test legal retrieval precision, refusal correctness, and confidence calibration

* Designed regression-safe workflows for future legal updates



---

## Non-Goals (By Design)

* ❌ No free-form legal advice
* ❌ No unverifiable generation
* ❌ No black-box agent chains
* ❌ No silent fallbacks when data is missing

---

## Roadmap

* Improve section-aware chunking
* Multi-source corroboration
* Confidence calibration
* Automated update & drift detection
* Optional LLM-based summarization (strictly gated)

---

## Disclaimer

ILAP provides **informational legal content only**. It is not a substitute for professional legal advice.

---

## Author

Built as a systems-first GenAI project to demonstrate production-grade AI architecture, not a chatbot demo.

## Quick Start

```bash
# create venv
python -m venv .venv
source .venv/bin/activate

# install deps
pip install -r requirements.txt

# run ingestion (example)
python scripts/ingest_bns.py

# start API
uvicorn app.main:app --reload
