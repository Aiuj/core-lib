# core-lib

Shared Python library (Aiuj/core-lib) providing LLM clients, embeddings, caching, job queues,
API utilities, centralized logging/OTLP, and settings management for AI apps. Consumed as a
git dependency by sibling repos: `mcp-doc-qa`, `agent-rfx`, `saas-admin`, `rfx-evaluation`.
Changes here can break downstream consumers — check `RELEASE_NOTES.md` conventions and mention
downstream impact when relevant.

Workflow rules (releases, `uv run` requirement) live in `.agents/AGENTS.md` — read that before
doing a release or running any Python script.

## Commands
- Test: `uv run pytest tests/ --disable-warnings -q` (or `make test`)
- Lint: `uv run ruff check core_lib tests` (or `make lint`)
- Install (dev): `uv sync` or `uv pip install -e .[dev]`
- Single test file: `uv run pytest tests/test_<name>.py -q`

Always use `uv run` for Python — never bare `python`.

## Directory map
- `core_lib/` — the package: `llm/`, `embeddings/`, `cache/`, `jobs/`, `api_utils/`, `tracing/`,
  `reranker/`, `classification/`, `ocr/`, `config/`, `locale/`, `utils/`, `scripts/`
- `tests/` — pytest suite, one `test_*.py` per feature area
- `docs/` — feature guides (one topic per file); `examples/` — runnable usage scripts
- `scripts/` — maintenance scripts (benchmarking, cache clearing, translation compilation)

## Context hygiene (avoid reading these wholesale)
- `README.md` (~37KB) and `RELEASE_NOTES.md` (~30KB) — grep for the section/version you need
  instead of reading the whole file.
- `uv.lock` (~460KB) — never read directly; use `uv tree` / `uv pip show <pkg>` instead.
- `.coverage` — binary SQLite file, not human-readable; don't open it.
- `docs/*.md` — many topic-specific guides; grep for the topic first, then read only the
  matching file rather than scanning the directory.
