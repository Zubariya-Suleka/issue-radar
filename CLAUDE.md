# issue-radar — working notes for Claude Code

Find open-source issues worth contributing to. Three parts, built in order:
1. **LLM gateway** (`services/gateway`) — OpenAI-compatible FastAPI service, swappable model backend, cache, logging, rate limiting, safety checks. **In progress.**
2. **RAG service** — retrieval over GitHub issues/PRs of pandas, scikit-learn, pydantic, polars; evaluated with free labels (duplicate issues, issue → fixing PR).
3. **SQL agent** — function-calling agent over Postgres (issues, PRs, labels, contributors), with tests for wrong tool calls and bad SQL.

**Read [DECISIONS.md](DECISIONS.md) first.** It records every design choice, the rejected alternatives, and open questions. Add an entry for each new decision (including what failed).

## How we work
- **Concept first.** Before code for any component: explain the problem and why the design looks the way it does, in a few lines. The owner is strong in Python/SQL (skip basics), new to Docker, FastAPI and system design.
- **The owner decides.** Lay out real trade-offs with a recommendation; be direct when a choice looks wrong. Flag generic tutorial patterns.
- **Split.** Claude scaffolds boilerplate (structure, config, Dockerfiles, plumbing, interfaces, tests). The owner writes core logic — model backend, cache, rate limiter, retrieval, agent loop — and Claude reviews it honestly, explaining what to fix and why. Don't write those pieces unless asked.
- **Tests as we go, small commits** with meaningful messages. Commits use the repo's git identity only — never add Claude co-author trailers or signatures.
- **Free and open source end to end** (see D-005). Check licences before adding dependencies.

## Commands
```sh
uv sync                                    # dev install (fake backend only)
uv sync --all-packages --extra local       # + torch/transformers for SmolLM (plain --extra local fails at the workspace root)
uv run pytest                              # tests (fast; never load a model)
uv run ruff check . && uv run ruff format --check .
uv run uvicorn gateway.main:create_app --factory          # gateway on :8000
# PowerShell: $env:GATEWAY_BACKEND="smollm"; uv run uvicorn gateway.main:create_app --factory
```

## Environment gotchas (Windows laptop, CPU only, 16 GB RAM)
- **Keep C: free.** `UV_CACHE_DIR=G:\cache\uv` and `HF_HOME=G:\cache\huggingface` are set as user env vars. A terminal opened before they were set won't see them — check `echo $env:HF_HOME` before downloading models.
- **Docker** is not set up for this project yet; its storage defaults to C:. Decide engine and move storage to G: before containerising (D-007).
- Model: SmolLM2-360M-Instruct on CPU (D-001). Embeddings: all-MiniLM-L6-v2.

## Code map (gateway)
- `backends/base.py` — the contract: `ModelBackend` Protocol + frozen `GenerationRequest`/`GenerationResult` (D-010).
- `backends/serial.py` — one generation at a time, queue timeout → 503 (D-011).
- `backends/fake.py` — instant echo backend for tests. `backends/smollm.py` — **owner's stub to implement**.
- `api/schemas.py` + `api/routes.py` — OpenAI-compatible subset; unsupported fields → 422 (D-008).

## Current status — update at the end of each session
- Done: repo, licence, DECISIONS D-001..D-012, gateway skeleton (13 tests passing), pushed to github.com/Zubariya-Suleka/issue-radar.
- **Next (owner):** implement `SmolLMBackend` in `services/gateway/src/gateway/backends/smollm.py` (its docstring lists 7 questions to answer), then Claude reviews.
- After that: request logging → cache (revisit default temperature) → rate limiter → safety checks + prompt-injection benchmark (deepset/prompt-injections) → Docker.
