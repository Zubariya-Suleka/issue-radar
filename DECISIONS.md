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

## D-008 · Gateway API: OpenAI-compatible
*2026-10-11 · Status: accepted*

**Context.** The gateway needs a public request/response format that the RAG service and SQL agent call.

**Options.** (a) Custom `POST /generate` taking a prompt string — simpler, fully our own design. (b) OpenAI-compatible `POST /v1/chat/completions` taking a list of messages.

**Decision.** OpenAI-compatible.

**Why.** Any existing client, library or tool can use the gateway by changing its base URL, so it doubles as a personal LLM proxy. Service code never knows where the model runs. The messages format is also what chat-tuned models (SmolLM2-Instruct) expect via their chat template.

**Consequences.** We implement a subset of the spec (no streaming, no tools at first) and must document which fields are supported and reject the rest explicitly rather than silently ignoring them.

---

## D-009 · Licence: MIT
*2026-10-11 · Status: accepted*

**Options.** MIT (short, widely understood) vs Apache-2.0 (explicit patent grant, more formal).

**Decision.** MIT. A portfolio project benefits most from the simplest, most familiar licence; patent terms add little here. Compatible with all dependencies in D-005.

---

## D-010 · Backend contract: a `typing.Protocol` over internal types
*2026-10-11 · Status: accepted*

**Context.** Gateway logic (cache, limiter, safety, logging) must not know what produces the text, so models can be swapped by config.

**Options.** (a) `typing.Protocol`: any class with a matching `name` and async `generate` qualifies; checked by the type checker. (b) `abc.ABC`: backends inherit a base class; a missing method fails at instantiation.

**Decision.** `Protocol`, with frozen dataclasses `GenerationRequest` / `GenerationResult` as the only types crossing the boundary. The OpenAI schema stays in `gateway.api`; backends never import it.

**Why.** Small interface, no inheritance ceremony, and test fakes are plain classes. Internal types decouple backends from the wire format and carry token counts for logging and limits. Frozen + tuples make requests hashable, which the cache can use.

**Consequences.** A backend with a wrong signature is caught by a type checker, not at runtime, so type checking should eventually run in CI.

---

## D-011 · Busy policy: one generation at a time, queue with timeout
*2026-10-11 · Status: accepted*

**Context.** On a 4-core CPU, concurrent generations each slow down roughly in proportion; concurrency adds latency, not throughput. And `model.generate()` blocks, so it must never run on the event loop.

**Options.** (a) Queue: wait up to N seconds for the model, then 503. (b) Reject immediately with 503 + Retry-After.

**Decision.** Queue with timeout (`GATEWAY_QUEUE_TIMEOUT_S`, default 30s), implemented as a `SerialBackend` wrapper around any backend; busy → 503 with an OpenAI-style error body.

**Why.** Eval runs fire many requests; queueing absorbs bursts without forcing every caller to write retry logic. Making it a wrapper keeps backends simple and applies the policy uniformly.

**Consequences.**
- The timeout covers waiting only; a started generation runs to completion (a CPU-bound thread can't be cancelled).
- If a client disconnects mid-generation and the server cancels the handler, the lock is released while the worker thread may still be running, briefly allowing two generations. Acceptable for now; revisit if it shows up in load tests.
- This protects the CPU. It is not rate limiting: the limiter (later) protects fairness between clients. Different jobs.

---

## D-012 · Tooling: uv workspace, one lockfile
*2026-10-11 · Status: accepted*

**Decision.** A root uv workspace with each service as a member: one `uv.lock`, one dev virtualenv, per-service `pyproject.toml` for Docker images. Heavy model dependencies (torch, transformers) sit behind the gateway's optional `local` extra so tests never need them. uv and Hugging Face caches live on G: (`UV_CACHE_DIR`, `HF_HOME`) per D-007.

**Note.** Starlette's test client now prefers `httpx2` (BSD-3, pydantic org) over `httpx`; the dev group uses it.

---

## D-013 · SmolLM backend: transformers, bf16, template defaults kept
*2026-10-11 · Status: accepted*

**Context.** First real backend behind the D-010 contract. Built on transformers 5.19 / torch 2.14 (CPU).

**Decisions and measurements.**
- **dtype: bf16, set explicitly.** transformers v5 loads in the checkpoint's dtype (bf16) by default. D-001 expected bf16 to be slow on this CPU, so both were measured, 64 greedy tokens, 4 threads: bf16 11.4 tok/s, fp32 11.7 tok/s — the same. bf16 wins on RAM (~0.7 GB vs ~1.45 GB). D-001's bf16 worry does not hold at this model size.
- **Prompt: the tokenizer's chat template with `add_generation_prompt=True`.** With no system message, SmolLM2's template injects *"You are a helpful AI assistant named SmolLM, trained by Hugging Face"* (~13 prompt tokens). Kept: it is the distribution the model was tuned on. Callers that need a different persona send their own system message, which replaces it.
- **temperature 0 → greedy (`do_sample=False`).** Sampling at temperature 0 divides logits by zero. When sampling, transformers also applies **top_k=50** by default; OpenAI's `temperature` means plain temperature sampling. Kept as a known divergence: a 360M model's low-probability tail is mostly noise.
- **finish_reason from the last token**, not the count: EOS can land exactly on the `max_tokens`-th token. `completion_tokens` includes the EOS token (it was computed, so it counts).
- **Blocking:** `asyncio.to_thread`, with `torch.inference_mode()` entered *inside* the worker (grad mode is thread-local).
- **Tests:** marked `slow`, deselected by default, skipped unless the weights are cached — the default `pytest` never loads or downloads a model.

**What failed / gotchas.**
- v5 `apply_chat_template(..., return_tensors="pt")` returns a `BatchEncoding` (dict), not a tensor; v4-era tutorial code breaks.
- The model repo is ~5.8 GB, ~5 GB of it ONNX variants we never use. `hf download` with positional filenames silently ignores `--include/--exclude`; download the eight needed files by name (command in `test_smollm_backend.py`).
- Windows without Developer Mode can't create symlinks, so the HF cache stores real copies. Harmless with one revision.

**Consequences.** ~1.5 s per 16 tokens means eval runs over hundreds of prompts take minutes; the cache matters.

---

## D-014 · Request logging: one JSON line per request
*2026-10-11 · Status: accepted*

**Context.** The cache, rate limiter, safety checks and backend comparisons all need data about real traffic: repeat rates, per-client volume, latency split, flagged requests.

**Options.** (a) Scattered log messages (started / generating / done). (b) One line per request with every field ("canonical log line"). (c) OpenTelemetry traces + Prometheus metrics.

**Decision.** (b), with stdlib `logging` and a small JSON formatter, written to stdout and `logs/gateway.jsonl` (gitignored; `GATEWAY_LOG_FILE`).
- **Middleware** writes the line when the request ends, so 422s (rejected before the route), 503s and crashes (500 + exception class) are logged too.
- Layers add fields through `annotate()`, backed by a per-request `ContextVar`: the route adds model, token counts, `max_tokens`, `temperature`, `prompt_sha256`; `SerialBackend` adds `queue_ms` and `generate_ms`. No timing data in the backend contract (D-010).
- **No prompt or response text by default** (D-006: issue text can carry personal data). `prompt_sha256` over (role, content) pairs shows repeated prompts without storing them; with `max_tokens` and `temperature` it is enough to measure what a cache would have hit. `GATEWAY_LOG_CONTENT=true` adds text for debugging.
- **Request ID:** a caller's `X-Request-ID` is kept if it matches `[A-Za-z0-9._-]{1,128}`, otherwise replaced (it lands in our logs); echoed in the response. Lets RAG / agent calls be followed across services later.

**Why.** One line = one row: loads straight into pandas or Postgres for analysis. (a) needs joining by ID before any question can be answered. (c) is built for many services on many hosts; at one gateway on a laptop, metrics can be derived from the lines.

**Consequences.**
- `client` is `null` until the rate limiter defines client identity.
- Uvicorn's own access log duplicates part of each line; run with `--no-access-log`.
- No rotation yet; the file grows until rotated or deleted.
- First measurement: a cold first request ran at ~7 tok/s vs 11.5 tok/s warm benchmark (D-013). A startup warm-up generation is a possible later fix.

---

## Open questions
- Docker Engine in WSL2 vs Docker Desktop (D-007).
- Default temperature (currently 0.0, greedy). Revisit with the cache design: caching non-greedy outputs changes their meaning.
- llama.cpp backend (llama-cpp-python, MIT; quantised GGUF) as a second local backend. Likely 2–4× faster on CPU than transformers; measure against D-013's 11.5 tok/s baseline and check output quality at Q8/Q4 before switching. Windows install needs a prebuilt CPU wheel or a C++ toolchain.
- Main RAG eval axis: chunking strategies vs retrieval method (BM25 / dense / hybrid). Issue text is short and identifier-heavy, so chunking may matter mainly for long PR threads.
