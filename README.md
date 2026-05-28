# ilap-backend-system
# ILAP – Platform Backend System

ILAP Platform Backend System is the **production-oriented application backend** for the ILAP ecosystem. While the AI service is responsible for grounded legal retrieval and answer generation, this backend is responsible for **identity, sessions, conversation orchestration, frontend-safe APIs, support workflows, category management, and production deployment controls**.

This project is intentionally built as an **integration-first platform layer**: it sits between client applications and the AI engine, enforcing authentication, request validation, category routing, session continuity, and operational safety.

---

## Why This Backend Exists

Most AI demos expose a single `/ask` route and leave the rest of the product undefined. ILAP separates concerns deliberately:

* The AI service focuses on retrieval, proof, and guarded legal answering
* This backend focuses on application state, identity, frontend contracts, and production-safe API behavior

Key design principles:

* Stable client-facing API contracts
* Authentication before privileged access
* Product state separated from AI inference
* Operational safety before convenience

---

## Core Features

* **Authentication and Session Management**
  Supports registration, login, refresh, logout, password reset, email verification, and authenticated session lookup.

* **Frontend-Safe Category Layer**
  Exposes six top-level legal categories for product UX while internally mapping them to the AI service's finer-grained `law_type` taxonomy.

* **Conversation Orchestration**
  Manages user conversations, multi-turn messages, idempotent asks, stored AI session IDs, and conversation history.

* **AI Service Integration**
  Forwards validated asks to the standalone AI service, stores normalized answer payloads, and preserves raw upstream payloads for traceability.

* **Category-Aware AI Follow-Ups**
  First-turn asks can be sent without `lawType`, then follow-up turns reuse the `law_type` inferred from the first returned citation.

* **Support and Contact Workflows**
  Provides endpoints for contact requests, support tickets, and early-access requests.

* **Admin Visibility**
  Includes admin-only APIs for users, conversations, support queues, and audit logs.

* **Production Deployment Controls**
  Supports Render deployment, PostgreSQL in production, health checks, schema migrations, environment validation, and stricter runtime safety rules.

* **Rate Limiting and Email Hooks**
  Prepared for Upstash-backed rate limiting and Resend-backed verification/reset flows.

---

## High-Level Architecture

```text
Frontend / Mobile Client
   ↓
FastAPI Platform Backend
   ↓
Auth + Sessions + Categories + Conversations + Support
   ↓
Standalone ILAP AI Service
   ↓
Grounded Legal Response
```

* The backend is the system of record for user identity and app-level state
* The AI service is the system of record for legal retrieval and answer generation
* The frontend should talk to this backend, not directly to the AI service
* Top-level product categories are backend-defined and AI taxonomy is mapped underneath

---

## Tech Stack

* **Backend Framework**: Python, FastAPI
* **ORM / Database Layer**: SQLAlchemy, Alembic
* **Default Local Database**: SQLite
* **Production Database**: PostgreSQL
* **Authentication**: JWT access and refresh sessions
* **Email Hooks**: Resend
* **Rate Limiting**: Upstash Redis or local in-memory fallback
* **Deployment Target**: Render
* **AI Integration**: External ILAP AI service over HTTP JSON

---

## Project Structure

```text
ilap_backend_system/
├── app/
│   ├── api/            # Route handlers and API wiring
│   ├── core/           # Settings, security, error helpers
│   ├── db/             # Engine, session, metadata setup
│   ├── integrations/   # External AI service client
│   ├── models/         # SQLAlchemy entities
│   ├── schemas/        # Request/response models
│   ├── services/       # Business logic
│   └── main.py         # FastAPI application factory
├── migrations/         # Alembic migrations
├── scripts/            # Deployment helpers
├── tests/              # API and flow tests
├── render.yaml         # Render blueprint
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Current Capabilities

* Registers users and issues access and refresh sessions

* Supports login, logout, refresh, session inspection, password reset, and email verification flows

* Supports authenticated profile retrieval and profile updates

* Exposes six product-facing legal categories:
  * Criminal Law
  * Property Law
  * Cyber Law
  * Consumer Rights
  * Employment Law
  * Family Law

* Maps those six top-level categories to the AI service's more detailed taxonomy through `aiLawTypes`

* Creates and stores user conversations with title, status, message history, AI session ID, and latest turn context

* Supports multi-turn ask flows with idempotency protection and stored normalized answer payloads

* Preserves AI citations, confidence, disclaimers, category notes, and proof payloads

* Supports a first-turn `lawType` omission strategy, then uses the first returned citation's `law_type` for later follow-up calls

* Provides support, contact, and early-access request submission endpoints

* Provides admin APIs for user, conversation, support, and audit visibility

* Ships with Swagger docs, health endpoints, and Render-ready startup behavior

---

## Top-Level Category Mapping

The backend keeps the frontend category picker intentionally simple while still aligning with the AI retrieval taxonomy.

Current mapping:

* `Criminal Law` → `criminal_law`, `criminal_procedure`, `evidence_law`, `constitutional_law`
* `Property Law` → `property_law`
* `Cyber Law` → `cyber_law`
* `Consumer Rights` → `consumer_rights`, `contract_law`, `insurance_law`
* `Employment Law` → `employment_law`
* `Family Law` → `family_law`, `tax_law`, `company_corporate_law`, `banking_finance_law`, `environmental_law`

This means:

* the frontend category screen should show only the six backend categories
* the AI layer can still operate with more detailed internal `law_type` values
* the backend remains the source of truth for product-facing category UX

---

## API Surface

Primary route groups:

* `/api/v1/auth`
  Registration, login, refresh, logout, session, forgot-password, reset-password, verify-email, resend-verification

* `/api/v1/me`
  Authenticated profile retrieval and updates

* `/api/v1/legal-categories`
  Category catalog with `lawType` and mapped `aiLawTypes`

* `/api/v1/conversations`
  Create, list, retrieve, update, delete, ask, and message history

* `/api/v1/support`
  Contact, support, and early-access requests

* `/api/v1/admin`
  Admin-only listing endpoints for users, conversations, support queues, and audit logs

Operational routes:

* `/`
* `/healthz`
* `/docs`
* `/openapi.json`

---

## Authentication Model

The backend uses a session-backed JWT flow:

* register or login returns:
  * user payload
  * access token
  * refresh token

* authenticated routes require the Bearer access token

* refresh rotates or renews session state through the stored refresh token path

* logout revokes the current session

Email verification can be enforced or relaxed depending on environment:

* `REQUIRE_VERIFIED_EMAIL=false`
  Useful for local development, staging, and early product integration

* `REQUIRE_VERIFIED_EMAIL=true`
  Intended for stricter production setups when email delivery is configured

---

## AI Integration Behavior

This backend does not perform legal retrieval itself. Instead, it calls the external ILAP AI service:

* Base URL comes from `AI_SERVICE_BASE_URL`
* Ask endpoint is:
  * `POST {AI_SERVICE_BASE_URL}/ask`

The backend sends:

* `query`
* optional `lawType`
* optional `sessionId`
* optional `contextTurnId`

The backend expects an answer payload with fields such as:

* `answer`
* `citations`
* `confidence`
* `disclaimer`
* `sessionId`
* `turnId`
* `categoryNote`
* `proof`

The backend stores:

* normalized answer data for frontend use
* raw AI payload for traceability and debugging

---

## Current Non-Goals

* ❌ This backend is not the legal reasoning engine
* ❌ It does not ingest statutes or manage Pinecone directly
* ❌ It does not let the frontend bypass auth and talk directly to AI
* ❌ It does not use AI output as the source of truth for the category picker

---

## Local Development

## Requirements

* Python 3.11+

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then run:

```bash
set -a
source .env
set +a
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Local docs:

* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Environment Variables

Important settings include:

* `ENVIRONMENT`
* `DEBUG`
* `AUTO_CREATE_SCHEMA`
* `SEED_REFERENCE_DATA`
* `ENABLE_ADMIN_BOOTSTRAP`
* `DOCS_ENABLED`

* `DATABASE_URL`
* `SYNC_DATABASE_URL`

* `JWT_SECRET_KEY`
* `ACCESS_TOKEN_TTL_MINUTES`
* `REFRESH_TOKEN_TTL_DAYS`
* `REQUIRE_VERIFIED_EMAIL`

* `AI_SERVICE_BASE_URL`
* `AI_SERVICE_TIMEOUT_SECONDS`
* `AI_STUB_MODE`

* `CORS_ORIGINS`

* `RESEND_API_KEY`
* `AUTH_EMAIL_FROM`
* `EMAIL_VERIFICATION_URL_TEMPLATE`
* `PASSWORD_RESET_URL_TEMPLATE`

* `UPSTASH_REDIS_REST_URL`
* `UPSTASH_REDIS_REST_TOKEN`

* `ADMIN_BOOTSTRAP_EMAIL`
* `ADMIN_BOOTSTRAP_PASSWORD`

Use:

* [`.env.example`](/Users/mayankdhyani/projects/ilap_backend_system/.env.example)
* [`.env.render.example`](/Users/mayankdhyani/projects/ilap_backend_system/.env.render.example)

as safe templates.

---

## Testing

Run the API tests with:

```bash
.venv/bin/pytest -q
```

The test suite uses temporary SQLite databases and validates:

* health checks
* registration and login
* conversation flow
* email verification token creation
* verified-email enforcement
* category mapping exposure
* AI `lawType` inference across follow-up turns

---

## Deployment

This repo includes:

* [render.yaml](/Users/mayankdhyani/projects/ilap_backend_system/render.yaml)
* [scripts/render-start.sh](/Users/mayankdhyani/projects/ilap_backend_system/scripts/render-start.sh)

Production behavior:

* startup runs `alembic upgrade head`
* service binds to `0.0.0.0:$PORT`
* production rejects SQLite
* production requires a non-default, 32+ character `JWT_SECRET_KEY`
* docs can remain enabled or be disabled through config

Recommended Render setup:

* Web Service for the backend
* Render Postgres for `DATABASE_URL` and `SYNC_DATABASE_URL`
* optional Resend for email flows
* optional Upstash Redis for rate limiting

Typical Render commands:

```bash
Build Command: pip install -r requirements.txt
Start Command: sh scripts/render-start.sh
```

---

## Frontend Integration Notes

The frontend should integrate against this backend, not the AI service directly.

Recommended frontend behavior:

* fetch categories from `/api/v1/legal-categories`
* do not hardcode legal categories in the client
* send top-level `lawType` values from the category picker
* use backend auth endpoints for registration, login, refresh, and logout
* use conversation endpoints for all ask/history flows

The category picker should display the six top-level categories only. The backend handles the mapping to detailed AI taxonomy under the hood.

---

## Disclaimer

This backend is part of the ILAP platform and is intended to support **informational legal product experiences**. It is not itself a legal reasoning engine and should be used together with the grounded ILAP AI service, not as a substitute for professional legal advice.

---

## Author

Built as the platform and product-state layer for ILAP, designed to turn a grounded legal AI engine into a deployable, frontend-ready application backend.
