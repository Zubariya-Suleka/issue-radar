# Design Decisions

A running record of the design choices in this project: what was considered, what was chosen, why, and what failed.
Newest entries go at the bottom. Entries are never deleted; superseded ones are marked as such.

Format: **Context** (the problem) → **Options** (real alternatives) → **Decision** → **Why** → **Consequences** (costs we accept).

---

## D-001 · Hardware budget and model size
*2026-10-11 · Status: accepted*

**Context.** Development machine: Intel i7-10510U (4 cores / 8 threads), 15.8 GB RAM with ~6.5 GB free in normal use, no usable GPU (no CUDA), Python 3.12. Everything — model, embeddings, Postgres, containers — must share that.

**Options.**
| Model | Approx. fp32 RAM | Verdict |
|---|---|---|
| SmolLM2-360M-Instruct | ~1.5 GB | Fits comfortably alongside everything else; fast enough for a tight dev loop on CPU |
| SmolLM2-1.7B-Instruct | ~7 GB (≈3.5 GB bf16) | Fits on paper, not next to Docker + Postgres + embeddings. bf16 halves RAM but this CPU has no native bf16, so it is slow |

**Decision.** SmolLM2-360M-Instruct is the baseline everywhere. Embeddings: all-MiniLM-L6-v2 on CPU. 1.7B is a later, measured experiment.

**Why.** Free RAM, not total RAM, is the binding constraint. A fast dev loop matters more than answer quality while building infrastructure.

**Consequences.** Generation quality is weak and function calling will be unreliable (to be measured in the SQL agent phase). The gateway's backend must be swappable so a bigger local model or an API can be plugged in without touching the services.

---

## D-002 · Monorepo
*2026-10-11 · Status: accepted*

**Context.** Three services (gateway, RAG, SQL agent) form one connected system; RAG and the agent call the gateway.

**Options.** (a) One monorepo with per-service packages. (b) Three repos with a shared client published or vendored.

**Decision.** Monorepo: `services/{gateway,rag,sql_agent}`, `libs/gateway_client`, one `docker-compose.yml`.

**Why.** One owner, one release cadence. A gateway API change and its client change land in one commit. One `compose up` runs the whole system. Separate repos only pay off with separate owners or release cycles.

**Consequences.** Each service still gets its own `pyproject.toml` and Dockerfile to keep images small and dependencies isolated.

---

## D-003 · Domain: open-source contribution finder
*2026-10-11 · Status: accepted*

**Context.** The project should be portfolio-grade, distinctive, useful in real life, built only on public data, and evaluable with real metrics. The key filter turned out to be: **does the data come with free ground-truth labels?** Without labels, RAG evaluation degrades to "looks good".

**Options considered and why they were rejected.**
| Idea | Why not |
|---|---|
| Supply chain / ecommerce RAG (Olist etc.) | Text-to-SQL over Olist is one of the most-tutorialed setups; no personal use |
| LinkedIn optimizer | Rewriting profiles is generation-heavy — the task a 360M model is worst at; scraping LinkedIn violates its terms; common pattern |
| Technical document formatter | Formatting is deterministic — pandoc / linters / Vale beat an LLM; no natural SQL agent |
| Personal assistant | A category, not a product; depends on reliable tool calling, which tiny models lack |
| Job-market intelligence | No free labels; would require hand-labelling |
| Dependency risk assistant (OSV) | GitHub Dependabot / advisory scanning already does this for free |
| Product recalls, public procurement | Good labels, low personal use |
| ML paper radar (arXiv + OpenAlex citations) | Strong runner-up: citation-based labels, recurring content. Not chosen — preference for the contribution finder's real-life value |
| EU AI Act assistant, case law, clinical trials, transit | Accuracy risk with a tiny model, low personal use, weak labels, or weak RAG half respectively |

**Decision.** **issue-radar**: a system that finds open-source issues worth contributing to and answers questions about how projects treat contributors.

**Why.**
- Real use: finding issues that are genuinely open to newcomers (no linked PR, recent maintainer activity, not already claimed) — beyond the label-only filtering of existing "good first issue" sites.
- Free labels: issues closed as duplicates (query → original) and PRs that say "Fixes #N" (issue → fixing PR).
- Real threat model for the gateway: issue text is untrusted input written by strangers and fed to an LLM — prompt injection via GitHub issues is a publicised attack on AI coding agents.
- Rich relational data for the SQL agent (issues, PRs, comments, labels, events), with exact, testable answers.

**Consequences.** Depends on the GitHub API (free but proprietary) — see D-006.

---

## D-004 · Source repositories
*2026-10-11 · Status: accepted*

**Context.** Need repos that are open-source licensed, mid-sized (fit on a laptop), relevant to Python/ML work, and rich in labels for evaluation. Counts below were taken from the GitHub search API on 2026-10-11.

| Repo | Licence | Role | Evidence |
|---|---|---|---|
| pandas-dev/pandas | BSD-3 | Primary source for duplicate-retrieval eval | 1,134 issues labelled `Duplicate Report`; 1,750 labelled `good first issue` |
| scikit-learn/scikit-learn | BSD-3 | Primary source for the contribution finder | 349 `good first issue`; also `Easy`, `Needs Triage`; only 13 closed with the `duplicate` close reason |
| pydantic/pydantic | MIT | Second duplicate source (eval not tied to one repo) | 108 labelled `duplicate` + 117 closed with `duplicate` reason |
| pola-rs/polars | MIT | Newcomer statistics | `first-contribution` label (direct ground truth); 102 closed with `duplicate` reason. Mostly Rust — personal contributions via the Python layer |

**Rejected.** fastapi/fastapi — routes almost everything to Discussions (1 open issue). huggingface/sentence-transformers — 0 duplicate-marked issues.

**Consequences.** Repos mark duplicates three different ways (a label, GitHub's `duplicate` close reason, a "Duplicate of #N" comment). Ingestion needs a normaliser that maps all three onto one `duplicate_of` field.

---

## D-005 · Free and open-source stack, end to end
*2026-10-11 · Status: accepted*

| Layer | Choice | Licence |
|---|---|---|
| Language / packaging | Python 3.12, uv | PSF / MIT-Apache |
| API | FastAPI, Uvicorn, Pydantic | MIT / BSD / MIT |
| Inference | transformers + PyTorch (CPU) | Apache 2.0 / BSD-3 |
| LLM | SmolLM2-360M-Instruct | Apache 2.0 |
| Embeddings | all-MiniLM-L6-v2 | Apache 2.0 |
| Data, vectors, keyword search | PostgreSQL + pgvector + Postgres full-text search | PostgreSQL Licence |
| Shared cache / limiter state (if needed) | Valkey | BSD-3 |
| Testing / linting | pytest, ruff | MIT |
| Synthetic PII test data | Faker | MIT |
| Prompt-injection benchmark | deepset/prompt-injections (662 rows) | Apache 2.0 |

**Notable choices.**
- **Valkey over Redis.** Redis left BSD in 2024 (RSAL/SSPL); Redis 8 added AGPLv3 back as an option. Valkey stayed BSD-3 under foundation governance — the simpler licence.
- **Postgres for everything (no separate vector DB).** Keyword search, vector search and the agent's data in one service: less RAM, and hybrid search becomes one SQL query.

---

## D-006 · Data policy and GitHub terms
*2026-10-11 · Status: accepted*

**Context.** GitHub's Acceptable Use Policy (section 7) allows research use of public, non-personal information only if resulting publications are open access, and forbids spam and selling personal information.

**Decision.**
- The repository is public; results are published openly.
- Raw issue / comment data is **not committed**. The repo contains ingestion code and eval labels as issue numbers only — anyone can rebuild the dataset.
- No emails or real names are stored. GitHub logins are replaced by opaque IDs; "first-time contributor" analysis only needs to tell users apart, not identify them.

---

## D-007 · Storage location: keep C: free
*2026-10-11 · Status: accepted (Docker part open)*

**Context.** C: has limited free space; model weights, datasets, Docker images and Postgres volumes can grow to tens of GB.

**Decision.** The repo lives at `G:\gitrepos\issue-radar`. The Hugging Face cache (`HF_HOME`) and local data directories will point to G:.

**Open.** Docker storage defaults to C: whichever engine is used (Docker Desktop's VM disk or the WSL distro disk). To decide before containerising: Docker Engine in WSL2 (open source, lower RAM, distro disk moved to G: with `wsl --manage <distro> --move`) vs Docker Desktop (proprietary but free for personal use, disk image location configurable).

---

## Open questions
- Docker Engine in WSL2 vs Docker Desktop (D-007).
- Project licence (MIT vs Apache-2.0).
- Gateway API shape: custom `/generate` vs OpenAI-compatible `/v1/chat/completions`.
- Main RAG eval axis: chunking strategies vs retrieval method (BM25 / dense / hybrid). Issue text is short and identifier-heavy, so chunking may matter mainly for long PR threads.
