# issue-radar — working notes for Claude Code

Find open-source issues worth contributing to. Three parts, built in order:
1. **LLM gateway** (`services/gateway`) — OpenAI-compatible FastAPI service, swappable model backend, cache, logging, rate limiting, safety checks. **Code complete; Docker pending.**
2. **RAG service** — retrieval over GitHub issues/PRs of pandas, scikit-learn, pydantic, polars; evaluated with free labels (duplicate issues, issue → fixing PR).
3. **SQL agent** — function-calling agent over Postgres (issues, PRs, labels, contributors), with tests for wrong tool calls and bad SQL.

**Read [DECISIONS.md](DECISIONS.md) first.** It records every design choice, the rejected alternatives, and open questions. Add an entry for each new decision (including what failed).

## How we work
- **Claude writes the code, including core logic** (owner's choice, 2026-10-11, replacing the earlier "owner writes core logic" split). Make the design calls with the reasoning recorded in DECISIONS.md, then **explain at the end** of each piece: what was built, why it looks that way, what the numbers showed, and what the owner should check. The owner is strong in Python/SQL (skip basics), new to Docker, FastAPI and system design.
- **The owner decides** anything outward-facing or hard to undo (installing system software, Docker engine, pushing, spending). Flag generic tutorial patterns; be direct when a choice looks wrong.
- **Measure, don't assume.** Prefer a quick benchmark over a remembered rule of thumb (see D-013 bf16, D-017 detectors).
- **Tests as we go, small commits** with meaningful messages, committed as each piece lands. Commits use the repo's git identity only — never add Claude co-author trailers or signatures.
- **Free and open source end to end** (see D-005). Check licences before adding dependencies.

## Commands
```sh
uv sync                                    # dev install (fake backend only)
uv sync --all-packages --extra local --extra safety --group bench   # everything: SmolLM, TF-IDF detector, benchmarks
                                           # (plain `uv sync` removes extras/groups again; --all-packages is needed at the workspace root)
uv run pytest                              # tests (fast; never load a model)
uv run pytest -m slow                      # real-model tests (~30 s; skipped if weights not cached)
uv run ruff check . && uv run ruff format --check .
uv run uvicorn gateway.main:create_app --factory --no-access-log   # gateway on :8000
# PowerShell: $env:GATEWAY_BACKEND="smollm"; uv run uvicorn gateway.main:create_app --factory --no-access-log
uv run python services/gateway/benchmarks/prompt_injection.py      # detector benchmark; saves models/prompt_injection_tfidf.joblib
```
Useful settings (env or `.env`, prefix `GATEWAY_`): `BACKEND=smollm`, `SAFETY_DETECTOR=tfidf`, `SAFETY_MODE=flag|block|off`, `API_KEYS='{"rag":"..."}'`, `RATE_LIMIT_PER_MINUTE`, `CACHE_MAX_ENTRIES`, `LOG_CONTENT=true` (debug only). Full list: `config.py`.

## Environment gotchas (Windows laptop, CPU only, 16 GB RAM)
- **Keep C: free.** `UV_CACHE_DIR=G:\cache\uv` and `HF_HOME=G:\cache\huggingface` are set as user env vars. A terminal opened before they were set won't see them — check `echo $env:HF_HOME` before downloading models.
- **HF downloads:** model repos often carry GBs of ONNX variants. Download named files only (see `test_smollm_backend.py`); `hf download` ignores `--include` when filenames are given.
- **PowerShell 5.1** `Set-Content -Encoding utf8` writes a BOM, which breaks `pyproject.toml`. Edit files with the editor tools instead.
- **Windows timers** tick every ~15.6 ms: timing tests need margins.
- **Docker** is not set up for this project yet; its storage defaults to C:. Decide engine and move storage to G: before containerising (D-007).
- Model: SmolLM2-360M-Instruct on CPU (D-001). Embeddings: all-MiniLM-L6-v2.

## Code map (gateway)
Request path: middleware (request ID, log line) → API key + rate limit (dependencies) → route: injection check → `CachedBackend` → `SerialBackend` → model.
- `backends/base.py` — the contract: `ModelBackend` Protocol + frozen `GenerationRequest`/`GenerationResult` (D-010).
- `backends/serial.py` — one generation at a time, queue timeout → 503 (D-011).
- `backends/cached.py` — LRU of greedy generations, outside the queue (D-015).
- `backends/fake.py` — instant echo backend for tests. `backends/smollm.py` — SmolLM2 via transformers, bf16 (D-013).
- `api/schemas.py` + `api/routes.py` — OpenAI-compatible subset; unsupported fields → 422 (D-008).
- `api/deps.py` + `rate_limit.py` — Bearer key → client name, per-client token bucket → 429 (D-016). `api/errors.py` — OpenAI-style error bodies.
- `request_log.py` — middleware + `annotate()`: any layer adds fields to the request's single log line (D-014).
- `safety/detectors.py` — heuristic / TF-IDF / DeBERTa detectors, per-paragraph scoring. `safety/check.py` — flag/block on user messages (D-017).
- `benchmarks/prompt_injection.py` — the evidence behind D-017.

## Current status — update at the end of each session
- Done: DECISIONS D-001..D-017. Gateway: SmolLM backend, request logging, cache, API keys + rate limiting, prompt-injection detection with benchmark (58 fast + 7 slow tests). Cached on G:: SmolLM2 (~730 MB), DeBERTa detector (~740 MB, rejected in D-017 — can be deleted from G:\cache\huggingface). Local commits since b17f6ff are **not pushed** (owner to confirm).
- **Next:** Docker (owner decides engine and storage location, D-007) → start the RAG service: GitHub ingestion into Postgres with the duplicate normaliser (D-004), which also unblocks retraining the injection detector on all four repos.
- Candidates: llama.cpp backend as a measured speed experiment; startup warm-up generation (D-014); SQLite cache if restarts hurt (D-015).
