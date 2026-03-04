# LLM Provider Benchmark Script

`scripts/benchmark_providers.py` tests every provider defined in an `llm_providers.yaml` across multiple capability dimensions to determine which models are suitable for different use cases (RAG answer generation, sheet classification, structured JSON output, etc.).

---

## Quick Start

```powershell
# From the core-lib directory
cd C:\Dev\github\core-lib

# Benchmark all configured providers in mcp-doc-qa
uv run python scripts/benchmark_providers.py --providers-file ..\mcp-doc-qa\llm_providers.yaml

# Benchmark all providers in the current directory
uv run python scripts/benchmark_providers.py

# Filter to a specific model
uv run python scripts/benchmark_providers.py --providers-file ..\mcp-doc-qa\llm_providers.yaml --filter "qwen3.5:4b"

# Filter to all Ollama providers, 5 runs per test
uv run python scripts/benchmark_providers.py --filter ollama --runs 5

# Only providers eligible for IQ5 (e.g. complex reasoning tasks)
uv run python scripts/benchmark_providers.py --iq 5
```

---

## Resolving the Providers File

The script resolves `llm_providers.yaml` in this order:

1. `--providers-file` CLI argument
2. `LLM_PROVIDERS_FILE` environment variable
3. `llm_providers.yaml` / `llm_providers.yml` in the current working directory
4. `llm_providers.yaml` in the parent directory of `cwd`
5. `llm_providers.yaml` next to the script root

---

## All CLI Options

| Flag | Default | Description |
|---|---|---|
| `--providers-file`, `-f` | auto-detected | Path to `llm_providers.yaml` |
| `--filter` | _(all)_ | Substring filter on `provider:model` (e.g. `ollama`, `qwen3`, `gemini`) |
| `--iq LEVEL` | _(all)_ | Only providers whose `min_level`/`max_level` range includes this IQ |
| `--include-disabled` | off | Include providers with `enabled: false` |
| `--runs`, `-n` | 3 | Number of repetitions per multi-run test |
| `--timeout` | 60.0 | Per-call timeout in seconds |
| `--test NAME` | _(all)_ | Run only specific test names (repeatable) |
| `--schema-in-prompt` | off | Append JSON schema to system prompt for ALL providers |
| `--no-schema-in-prompt-ollama` | off | Disable automatic schema-in-prompt for Ollama |
| `--output`, `-o` | `table` | Output format: `table`, `json`, or `csv` |
| `--quiet`, `-q` | off | Suppress per-run verbose output (only print summary table) |

---

## Test Suite

Nine tests are run by default, covering the full capability profile:

| Test Name | Category | Schema | Description |
|---|---|---|---|
| `ping` | connectivity | none | Connectivity check — minimal prompt/response |
| `basic_reasoning` | reasoning | none | Short open-ended technical question (2-3 sentence answer) |
| `complex_reasoning` | reasoning | none | Multi-concept technical reasoning (RRF vs score normalisation) |
| `simple_structured` | structured_output | `SimpleAnswer` | 2-field JSON: `answer` + `confidence` |
| `rfx_structured` | structured_output | `RFxAnswer` | 3-field RFx JSON — mirrors `mcp-doc-qa` `ComplianceAwareAnswer` |
| `rfx_rag` | rag | `RFxAnswer` | RAG-style: answer from provided context — full mcp-doc-qa pattern |
| `sheet_classification` | classify | `SheetClassification` | Classify an Excel sheet — mirrors agent-rfx LangGraph output |
| `multilingual` | reasoning | none | French question — tests language understanding and response |
| `thinking_structured` | thinking | `SimpleAnswer` | Think-mode enabled + structured output — tests think+JSON reliability |

> **Run counts**: `ping` and `thinking_structured` always run once (connectivity and thinking are expensive). All other tests run `--runs` times (default: 3).

Run a subset with `--test`:
```powershell
# Only run connectivity and structured output tests
uv run python scripts/benchmark_providers.py --test ping --test simple_structured --test rfx_structured
```

---

## Scoring

Two use-case scores (0–5 stars) are computed per provider:

### `mcp-doc-qa` score
Weighted for RAG answer generation with compliance classification:
- **JSON reliability** (0–3 pts): Average JSON success rate on `rfx_structured` + `rfx_rag`
- **Latency** (0–1 pt): avg < 3000ms = 1.0, < 6000ms = 0.5
- **Multilingual** (0–1 pt): `multilingual` success rate >= 67% = 1.0

### `agent-rfx` score
Weighted for Excel analysis and sheet classification:
- **JSON reliability** (0–2 pts): Average of `sheet_classification` + `rfx_structured` JSON rates
- **Sheet classification** (0–2 pts): `sheet_classification` JSON rate >= 67% = 2.0
- **Latency** (0–1 pt): avg < 5000ms = 1.0, < 10000ms = 0.5

---

## Ollama Structured Output Grounding

Per the [Ollama structured outputs docs](https://docs.ollama.com/capabilities/structured-outputs):

> *"It is ideal to also pass the JSON schema as a string in the prompt to ground the model's response."*

**By default, schema-in-prompt is automatically enabled for all Ollama providers.** This embeds a human-readable schema description in the system prompt alongside the `format=<json_schema>` API parameter. This is critical for smaller models (< 7B) that may not reliably produce valid JSON from the API parameter alone.

To disable it:
```powershell
uv run python scripts/benchmark_providers.py --no-schema-in-prompt-ollama
```

To force schema-in-prompt for all providers (including cloud APIs):
```powershell
uv run python scripts/benchmark_providers.py --schema-in-prompt
```

---

## Output Formats

### Table (default)

Verbose per-test output with a final summary and recommendations:

```
---------------------------------------------------------------
  ollama:qwen3.5:14b  (IQ0-7, priority=1, tier=standard)
---------------------------------------------------------------
  ping                            [OK]     210ms      36in/2out
    | pong
  rfx_structured       [1/3]     [OK]     820ms     181in/62out
    | {"answer": "Yes, RBAC is supported...", "compliance_category": "yes", ...}
  ...

  Summary for ollama:qwen3.5:14b:
    JSON success rate : 100%
    Avg latency       : 680ms
    mcp-doc-qa score  : ***** (5.0/5)
    agent-rfx score   : ***** (5.0/5)
```

### JSON output

Machine-readable output with full detail for all providers and test runs:

```powershell
uv run python scripts/benchmark_providers.py --output json > results/benchmark.json
```

```json
[
  {
    "provider": "ollama",
    "model": "qwen3.5:14b",
    "label": "ollama:qwen3.5:14b",
    "iq_range": "IQ0-7",
    "tier": "standard",
    "mcp_doc_qa_score": 5.0,
    "agent_rfx_score": 5.0,
    "json_success_rate": 1.0,
    "avg_latency_ms": 680.4,
    "tests": { ... }
  }
]
```

### CSV output

One row per provider for spreadsheet analysis:

```powershell
uv run python scripts/benchmark_providers.py --output csv > results/benchmark.csv
```

---

## Common Recipes

```powershell
# Compare all cloud providers side-by-side (quiet, table summary only)
uv run python scripts/benchmark_providers.py --filter "gemini\|alibaba\|openai" --quiet

# Find the fastest Ollama model for IQ3 tasks
uv run python scripts/benchmark_providers.py --filter ollama --iq 3 --test ping --test rfx_structured

# Regression check: run structured output tests with 5 runs each and save to JSON
uv run python scripts/benchmark_providers.py ^
    --test rfx_structured --test rfx_rag --test sheet_classification ^
    --runs 5 --output json > results/structured_output_regression.json

# Test a model that is currently disabled
uv run python scripts/benchmark_providers.py --filter "mistral:7b" --include-disabled

# Benchmark without schema-in-prompt to see baseline Ollama JSON reliability
uv run python scripts/benchmark_providers.py --filter ollama --no-schema-in-prompt-ollama --runs 5
```

---

## Pydantic Schemas Used

The script defines three self-contained schemas (no app-level imports needed):

| Schema | Fields | Mirrors |
|---|---|---|
| `SimpleAnswer` | `answer: str`, `confidence: high\|medium\|low` | Minimal 2-field sanity check |
| `RFxAnswer` | `answer`, `compliance_category: yes\|no\|partial\|custom\|unknown`, `compliance_reasoning?` | `mcp-doc-qa` `ComplianceAwareAnswer` |
| `SheetClassification` | `sheet_type: question_table\|info_sheet\|pricing\|other`, `confidence: float`, `reasoning?` | `agent-rfx` LangGraph sheet classifier |

---

## Prerequisites

- Python environment with `core-lib` installed (`uv sync` from `core-lib` root)
- Provider credentials set in `.env` or environment:
  - Ollama: `OLLAMA_HOST` (default `http://127.0.0.1:11434`)
  - Gemini: `GEMINI_API_KEY`
  - Alibaba / DashScope: `DASHSCOPE_API_KEY`
  - OpenAI: `OPENAI_API_KEY`
  - Azure: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`
- At least one Ollama model pulled if testing locally: `ollama pull qwen3.5:14b`
