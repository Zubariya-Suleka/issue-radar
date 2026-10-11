# issue-radar

Find open-source issues worth contributing to, and ask questions about how projects treat contributors.
Runs entirely on a CPU-only laptop with small open models.

**Status:** design phase. See [DECISIONS.md](DECISIONS.md).

## Components (planned)
1. **LLM gateway**: FastAPI service in front of a swappable model backend, with caching, request logging, rate limiting and safety checks (input limits, prompt-injection and PII screening).
2. **RAG service**: retrieval over GitHub issues and PRs, evaluated against free ground truth (duplicate issues, issue → fixing-PR links).
3. **SQL agent**: function-calling agent over Postgres that answers questions about repos, issues and contributors.

Source repos: pandas, scikit-learn, pydantic, polars.

## Development
Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```sh
uv sync                                                   # install (fake backend only)
uv run pytest                                             # tests
uv run uvicorn gateway.main:create_app --factory          # gateway on :8000
uv sync --all-packages --extra local                      # + torch/transformers for SmolLM
GATEWAY_BACKEND=smollm uv run uvicorn gateway.main:create_app --factory
```
