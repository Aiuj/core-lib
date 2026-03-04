#!/usr/bin/env python3
"""LLM Provider Capability Benchmark

Tests every provider defined in an llm_providers.yaml file across multiple
capability dimensions to determine suitability for different use cases
(mcp-doc-qa answer generation, agent-rfx sheet classification, etc.)

Usage:
    # Benchmark all providers in a specific config file
    uv run python scripts/benchmark_providers.py --providers-file ../mcp-doc-qa/llm_providers.yaml

    # Filter to Ollama providers only, run 5 times per test
    uv run python scripts/benchmark_providers.py --providers-file llm_providers.yaml --filter ollama --runs 5

    # Test only providers eligible for IQ5, with schema-in-prompt Ollama grounding
    uv run python scripts/benchmark_providers.py --iq 5 --schema-in-prompt

    # JSON output for CI / analysis
    uv run python scripts/benchmark_providers.py --output json > results/benchmark.json

Environment:
    Requires the normal provider credentials to be set (.env or environment):
    GEMINI_API_KEY, DASHSCOPE_API_KEY, GOOGLE_CLOUD_PROJECT, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

# ---------------------------------------------------------------------------
# Bootstrap: make sure core_lib is importable when run from the scripts/ dir
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Schema definitions (self-contained - no app-level imports)
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field, ValidationError


class SimpleAnswer(BaseModel):
    """Minimal 2-field schema - fastest sanity check for structured output."""
    answer: str = Field(..., description="Direct answer to the question")
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence level in the answer"
    )


class RFxAnswer(BaseModel):
    """Mirrors mcp-doc-qa ComplianceAwareAnswer - key schema for RAG use cases."""
    answer: str = Field(..., description="The answer to the RFP question")
    compliance_category: Literal["yes", "no", "partial", "custom", "unknown"] = Field(
        default="unknown",
        description=(
            "Compliance status: 'yes' (fully compliant), 'no' (non-compliant), "
            "'partial' (partly met), 'custom' (needs clarification), 'unknown' (insufficient info)"
        )
    )
    compliance_reasoning: Optional[str] = Field(
        default=None,
        description="Brief reasoning for compliance categorisation (internal use)"
    )


class SheetClassification(BaseModel):
    """Mirrors agent-rfx sheet type classification output."""
    sheet_type: Literal["question_table", "info_sheet", "pricing", "other"] = Field(
        ..., description="Detected type of the Excel sheet"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    reasoning: Optional[str] = Field(
        default=None, description="Brief reasoning for classification"
    )


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkTest:
    name: str
    category: str  # connectivity | reasoning | structured_output | rag | classify | thinking
    system_message: str
    user_message: str
    schema: Optional[Type[BaseModel]] = None
    thinking_enabled: Optional[bool] = None   # None = use provider default
    description: str = ""


# Context snippet reused across RAG tests
_RAG_CONTEXT = """
Source 1 (2025-12-01): Our ERP system fully supports role-based access control (RBAC)
with row-level security. Users only see data within their assigned department or company
hierarchy. Administrators can configure security policies via the Access Management module.

Source 2 (2025-10-15): The reporting module integrates with the RBAC layer to enforce
data visibility at query time. All standard and custom reports respect the user's
permission scope automatically.
""".strip()

_SHEET_SAMPLE = """
| Row | Column A         | Column B | Column C   |
|-----|-----------------|----------|------------|
| 1   | Question        | Answer   | Compliance |
| 2   | Do you support SSO? | Yes, via SAML 2.0 and OAuth 2.0 | yes |
| 3   | Audit log retention? | 7 years | partial |
| 4   | Multi-tenant isolation? | Per-database schemas | yes |
""".strip()

BENCHMARK_TESTS: List[BenchmarkTest] = [
    BenchmarkTest(
        name="ping",
        category="connectivity",
        system_message="You are a helpful assistant. Answer concisely.",
        user_message="Say 'pong' and nothing else.",
        description="Connectivity check - minimal prompt/response",
    ),
    BenchmarkTest(
        name="basic_reasoning",
        category="reasoning",
        system_message="You are an expert software architect. Answer concisely (2-3 sentences max).",
        user_message=(
            "A multi-tenant SaaS application stores customer data in a shared PostgreSQL database. "
            "What is the best isolation strategy to prevent data leakage between tenants?"
        ),
        description="Short open-ended reasoning - no schema",
    ),
    BenchmarkTest(
        name="complex_reasoning",
        category="reasoning",
        system_message="You are a senior solutions architect. Be thorough but concise.",
        user_message=(
            "We are designing a RAG pipeline. We have both semantic (pgvector cosine similarity) "
            "and lexical (BM25 via OpenSearch) retrieval. Explain Reciprocal Rank Fusion (RRF) "
            "and when you would prefer it over simple score normalisation and summation."
        ),
        description="Multi-concept technical reasoning",
    ),
    BenchmarkTest(
        name="simple_structured",
        category="structured_output",
        system_message="Answer questions accurately and concisely. Respond in JSON.",
        user_message=(
            "Does PostgreSQL support JSONB natively?\n\n"
            "Respond with JSON matching this schema: "
            '{"answer": "<text>", "confidence": "high|medium|low"}'
        ),
        schema=SimpleAnswer,
        description="2-field structured output - minimal complexity",
    ),
    BenchmarkTest(
        name="rfx_structured",
        category="structured_output",
        system_message=(
            "You are an expert RFP response specialist. "
            "Answer the question and classify compliance. Respond in JSON."
        ),
        user_message=(
            "Question: Does the system support row-level security?\n\n"
            "Context: Our platform enforces row-level security through RBAC policies applied "
            "at query time. All reports and API responses respect the user's permission scope.\n\n"
            "Respond with JSON: "
            '{"answer": "<text>", "compliance_category": "yes|no|partial|custom|unknown", '
            '"compliance_reasoning": "<optional text>"}'
        ),
        schema=RFxAnswer,
        description="3-field RFx structured output - mirrors mcp-doc-qa ComplianceAwareAnswer",
    ),
    BenchmarkTest(
        name="rfx_rag",
        category="rag",
        system_message=(
            "Expert RFP response specialist. Craft accurate answers from the provided context only. "
            "Do not invent information not present in the context. Respond in JSON."
        ),
        user_message=(
            f"Q: Does the reporting module support row-level security and access rights management?\n\n"
            f"Context:\n{_RAG_CONTEXT}\n\n"
            "Respond with JSON: "
            '{"answer": "<text>", "compliance_category": "yes|no|partial|custom|unknown", '
            '"compliance_reasoning": "<optional text>"}'
        ),
        schema=RFxAnswer,
        description="RAG-style answer with context - full mcp-doc-qa pattern",
    ),
    BenchmarkTest(
        name="sheet_classification",
        category="classify",
        system_message=(
            "You are an expert Excel document analyser for RFx procurement processes. "
            "Classify the sheet type from its content. Respond in JSON."
        ),
        user_message=(
            "Analyse this Excel sheet sample and classify it:\n\n"
            f"{_SHEET_SAMPLE}\n\n"
            "Respond with JSON: "
            '{"sheet_type": "question_table|info_sheet|pricing|other", '
            '"confidence": 0.0-1.0, "reasoning": "<optional text>"}'
        ),
        schema=SheetClassification,
        description="Sheet classification - mirrors agent-rfx LangGraph tool output",
    ),
    BenchmarkTest(
        name="multilingual",
        category="reasoning",
        system_message="You are a helpful assistant. Respond in the same language as the question.",
        user_message=(
            "Le systeme supporte-t-il l'authentification unique (SSO) via SAML 2.0? "
            "Repondez en 1-2 phrases en francais."
        ),
        description="French language understanding and response",
    ),
    BenchmarkTest(
        name="thinking_structured",
        category="thinking",
        system_message=(
            "You are a careful analyst. Think step by step before answering. "
            "Respond in JSON."
        ),
        user_message=(
            "A vendor claims: 'Our system achieves 99.99% uptime'. "
            "Is this claim realistic for a cloud SaaS product? "
            "What questions should a procurement team ask to validate it?\n\n"
            "Respond with JSON: "
            '{"answer": "<your analysis>", "confidence": "high|medium|low"}'
        ),
        schema=SimpleAnswer,
        thinking_enabled=True,
        description="Thinking mode with structured output - tests think+JSON reliability",
    ),
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    success: bool
    latency_ms: float
    structured_returned: bool = False
    schema_valid: bool = False
    content_preview: str = ""
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_used: bool = False


@dataclass
class TestResult:
    test: BenchmarkTest
    runs: List[RunResult] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def success_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.success) / len(self.runs)

    @property
    def json_success_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.schema_valid) / len(self.runs)

    @property
    def avg_latency_ms(self) -> float:
        lats = [r.latency_ms for r in self.runs if r.success]
        return sum(lats) / len(lats) if lats else 0.0

    @property
    def p95_latency_ms(self) -> float:
        lats = sorted(r.latency_ms for r in self.runs if r.success)
        if not lats:
            return 0.0
        idx = max(0, int(len(lats) * 0.95) - 1)
        return lats[idx]

    @property
    def avg_tokens_in(self) -> float:
        vals = [r.input_tokens for r in self.runs if r.input_tokens > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_tokens_out(self) -> float:
        vals = [r.output_tokens for r in self.runs if r.output_tokens > 0]
        return sum(vals) / len(vals) if vals else 0.0


@dataclass
class ProviderReport:
    provider: str
    model: str
    priority: int
    min_iq: int
    max_iq: int
    tier: str
    thinking_configured: bool
    test_results: Dict[str, TestResult] = field(default_factory=dict)
    skip_reason: str = ""

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def iq_range(self) -> str:
        return f"IQ{self.min_iq}-{self.max_iq}"

    def get_json_success_rate(self) -> float:
        """Average JSON success rate across all structured-output tests."""
        structured_tests = [
            r for r in self.test_results.values()
            if r.test.schema is not None
        ]
        if not structured_tests:
            return 0.0
        rates = [t.json_success_rate for t in structured_tests]
        return sum(rates) / len(rates)

    def get_overall_latency(self) -> float:
        """Average latency across all successful tests."""
        lats = [r.avg_latency_ms for r in self.test_results.values() if r.avg_latency_ms > 0]
        return sum(lats) / len(lats) if lats else 0.0

    def mcp_doc_qa_score(self) -> float:
        """Suitability score 0-5 for mcp-doc-qa use case."""
        score = 0.0
        # JSON reliability (0-3): rfx_structured + rfx_rag most important
        rfx_s = self.test_results.get("rfx_structured")
        rfx_r = self.test_results.get("rfx_rag")
        json_rate = 0.0
        if rfx_s and rfx_r:
            json_rate = (rfx_s.json_success_rate + rfx_r.json_success_rate) / 2
        elif rfx_s:
            json_rate = rfx_s.json_success_rate
        elif rfx_r:
            json_rate = rfx_r.json_success_rate
        score += json_rate * 3.0

        # Latency (0-1): < 3000ms avg is acceptable
        avg_lat = self.get_overall_latency()
        if avg_lat > 0 and avg_lat < 3000:
            score += 1.0
        elif avg_lat > 0 and avg_lat < 6000:
            score += 0.5

        # Multilingual (0-1)
        ml = self.test_results.get("multilingual")
        if ml and ml.success_rate >= 0.67:
            score += 1.0
        elif ml and ml.success_rate > 0:
            score += 0.5

        return min(5.0, score)

    def agent_rfx_score(self) -> float:
        """Suitability score 0-5 for agent-rfx use case."""
        score = 0.0
        # JSON reliability (0-2)
        sheet = self.test_results.get("sheet_classification")
        rfx_s = self.test_results.get("rfx_structured")
        json_rate = 0.0
        if sheet and rfx_s:
            json_rate = (sheet.json_success_rate + rfx_s.json_success_rate) / 2
        elif sheet:
            json_rate = sheet.json_success_rate
        elif rfx_s:
            json_rate = rfx_s.json_success_rate
        score += json_rate * 2.0

        # Sheet classification pass (0-2)
        if sheet and sheet.json_success_rate >= 0.67:
            score += 2.0
        elif sheet and sheet.json_success_rate > 0:
            score += 1.0

        # Latency (0-1): agents can tolerate slightly more
        avg_lat = self.get_overall_latency()
        if avg_lat > 0 and avg_lat < 5000:
            score += 1.0
        elif avg_lat > 0 and avg_lat < 10000:
            score += 0.5

        return min(5.0, score)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def _build_schema_hint(schema: Type[BaseModel]) -> str:
    """Build a compact JSON schema hint for embedding in system prompts (Ollama grounding).

    Per https://docs.ollama.com/capabilities/structured-outputs:
    'It is ideal to also pass the JSON schema as a string in the prompt to
    ground the model's response.'
    """
    js = schema.model_json_schema()
    props = js.get("properties", {})
    required = js.get("required", [])
    lines = [f'Respond with valid JSON matching this schema for "{js.get("title", schema.__name__)}":']
    for name, prop in props.items():
        req_marker = " (required)" if name in required else " (optional)"
        typ = prop.get("type", "string")
        enum_vals = prop.get("enum") or _collect_enum_from_anyof(prop)
        if enum_vals:
            typ = " | ".join(str(v) for v in enum_vals)
        lines.append(f'  "{name}": {typ}{req_marker} -- {prop.get("description", "")}')
    return "\n".join(lines)


def _collect_enum_from_anyof(prop: Dict[str, Any]) -> List[Any]:
    """Extract enum values from anyOf/const patterns (Pydantic Literal serialisation)."""
    result = []
    for entry in prop.get("anyOf", []):
        if "const" in entry:
            result.append(entry["const"])
        if "enum" in entry:
            result.extend(entry["enum"])
    return result


def _run_single(
    client: Any,  # LLMClient
    test: BenchmarkTest,
    timeout_secs: float = 30.0,
    schema_in_prompt: bool = False,
) -> RunResult:
    """Execute one run of a single BenchmarkTest."""
    system = test.system_message
    if schema_in_prompt and test.schema is not None:
        system = system.rstrip() + "\n\n" + _build_schema_hint(test.schema)

    try:
        start = time.perf_counter()
        resp = client.chat(
            messages=test.user_message,
            structured_output=test.schema,
            system_message=system,
            thinking_enabled=test.thinking_enabled,
        )
        latency_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        return RunResult(
            success=False,
            latency_ms=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )

    # Parse response
    is_structured = resp.get("structured", False)
    content = resp.get("content", "")
    usage = resp.get("usage") or {}
    thinking = bool(resp.get("thinking"))

    input_tokens = usage.get("prompt_tokens") or usage.get("prompt_eval_count") or 0
    output_tokens = usage.get("completion_tokens") or usage.get("eval_count") or 0

    # Validate schema if expected
    schema_valid = False
    if test.schema is not None:
        if is_structured and isinstance(content, dict):
            try:
                test.schema.model_validate(content)
                schema_valid = True
            except (ValidationError, Exception):
                schema_valid = False
        else:
            # Try to extract JSON from text response via the shared parser
            text_content = str(content) if content else ""
            if text_content:
                try:
                    from core_lib.llm.json_parser import parse_structured_output
                    parsed = parse_structured_output(text_content, test.schema)
                    if parsed is not None:
                        schema_valid = True
                        is_structured = True
                        content = parsed
                except Exception:
                    pass

    # Build content preview
    if isinstance(content, dict):
        preview = json.dumps(content, ensure_ascii=False)[:120]
    else:
        preview = str(content)[:120] if content else ""
    preview = preview.replace("\n", " ").strip()

    success = bool(content) and not resp.get("error")

    return RunResult(
        success=success,
        latency_ms=latency_ms,
        structured_returned=is_structured,
        schema_valid=schema_valid,
        content_preview=preview,
        error=str(resp.get("error", "")),
        input_tokens=int(input_tokens or 0),
        output_tokens=int(output_tokens or 0),
        thinking_used=thinking,
    )


def run_provider_benchmark(
    config: Any,           # ProviderConfig
    tests: List[BenchmarkTest],
    n_runs: int = 3,
    timeout_secs: float = 30.0,
    schema_in_prompt: bool = False,
    verbose: bool = True,
) -> ProviderReport:
    """Run all benchmark tests against one provider and return a ProviderReport."""
    report = ProviderReport(
        provider=config.provider,
        model=config.model,
        priority=config.priority,
        min_iq=config.min_intelligence_level,
        max_iq=config.max_intelligence_level,
        tier=config.tier or "--",
        thinking_configured=config.thinking_enabled,
    )

    # Create client
    try:
        client = config.to_client()
    except Exception as exc:
        report.skip_reason = f"Client creation failed: {exc}"
        return report

    if verbose:
        _print_provider_header(report)

    for test in tests:
        # Connectivity and thinking tests only need 1 run
        runs_count = 1 if test.category in ("connectivity", "thinking") else n_runs

        result = TestResult(test=test)

        for run_idx in range(runs_count):
            rr = _run_single(client, test, timeout_secs=timeout_secs, schema_in_prompt=schema_in_prompt)
            result.runs.append(rr)

            if verbose:
                _print_run_result(test, run_idx + 1, runs_count, rr)

        report.test_results[test.name] = result

    if verbose:
        _print_provider_summary(report)

    return report


# ---------------------------------------------------------------------------
# Console output helpers
# ---------------------------------------------------------------------------

def _icon(value: bool) -> str:
    return "OK" if value else "x"


def _print_provider_header(report: ProviderReport) -> None:
    label = f"{report.label}  ({report.iq_range}, priority={report.priority}, tier={report.tier})"
    thinking_tag = " [thinking]" if report.thinking_configured else ""
    sep = "-" * min(80, len(label) + 10)
    print(f"\n{sep}")
    print(f"  {label}{thinking_tag}")
    print(sep)


def _print_run_result(
    test: BenchmarkTest, run_idx: int, total_runs: int, rr: RunResult
) -> None:
    run_label = f"[{run_idx}/{total_runs}]" if total_runs > 1 else ""
    lat = f"{rr.latency_ms:.0f}ms" if rr.latency_ms > 0 else "--"
    tok = f"  {rr.input_tokens}in/{rr.output_tokens}out" if rr.input_tokens else ""

    if not rr.success:
        status = "[ERROR]"
        detail = rr.error[:80] if rr.error else "no content"
    elif test.schema is not None:
        status = "[JSON] " if rr.schema_valid else "[text] "
        detail = rr.content_preview[:80] if rr.schema_valid else (rr.content_preview[:60] or "no json extracted")
    else:
        status = "[OK]   "
        detail = rr.content_preview[:80]

    think_tag = " [think]" if rr.thinking_used else ""
    print(f"  {test.name:<22} {run_label:<8} {status:<8} {lat:<8}{tok}{think_tag}")
    if detail:
        print(f"    | {detail}")


def _stars(score: float) -> str:
    filled = round(score)
    return "*" * filled + "." * (5 - filled)


def _pct(rate: float) -> str:
    return f"{rate * 100:.0f}%"


def _print_provider_summary(report: ProviderReport) -> None:
    mcp_score = report.mcp_doc_qa_score()
    rfx_score = report.agent_rfx_score()
    json_rate = report.get_json_success_rate()
    avg_lat = report.get_overall_latency()

    print(f"\n  Summary for {report.label}:")
    print(f"    JSON success rate : {_pct(json_rate)}")
    print(f"    Avg latency       : {avg_lat:.0f}ms")
    print(f"    mcp-doc-qa score  : {_stars(mcp_score)} ({mcp_score:.1f}/5)")
    print(f"    agent-rfx score   : {_stars(rfx_score)} ({rfx_score:.1f}/5)")


def _print_summary_table(reports: List[ProviderReport]) -> None:
    """Print final side-by-side comparison table."""
    print("\n" + "=" * 80)
    print("  BENCHMARK SUMMARY")
    print("=" * 80)

    # Header
    col_w = [28, 8, 10, 8, 8, 8, 8, 12, 12]
    headers = ["Provider:Model", "IQ", "Tier", "JSON%", "ms avg", "RAG", "Classify", "mcp-doc-qa", "agent-rfx"]
    row = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(row)
    print("-" * 76)

    for r in sorted(reports, key=lambda x: -(x.mcp_doc_qa_score() + x.agent_rfx_score())):
        if r.skip_reason:
            print(f"  {r.label:<28}  [SKIP]   {r.skip_reason[:50]}")
            continue

        label = r.label[:27]
        iq = r.iq_range
        tier = r.tier[:8]
        json_pct = _pct(r.get_json_success_rate())
        avg_lat = f"{r.get_overall_latency():.0f}"

        rag_r = r.test_results.get("rfx_rag")
        rag_ok = _icon(bool(rag_r and rag_r.json_success_rate > 0.5)) if rag_r else "-"

        cls_r = r.test_results.get("sheet_classification")
        cls_ok = _icon(bool(cls_r and cls_r.json_success_rate > 0.5)) if cls_r else "-"

        mcp = _stars(r.mcp_doc_qa_score())
        rfx = _stars(r.agent_rfx_score())

        cells = [label, iq, tier, json_pct, avg_lat, rag_ok, cls_ok, mcp, rfx]
        row = "  ".join(c.ljust(w) for c, w in zip(cells, col_w))
        print(row)

    print("=" * 80)


def _print_recommendations(reports: List[ProviderReport]) -> None:
    """Print top-ranked providers per use case."""
    good = [r for r in reports if not r.skip_reason]
    if not good:
        return

    def top_for(scorer, n: int = 3) -> List[ProviderReport]:
        return sorted(good, key=lambda r: -scorer(r))[:n]

    print("\n  RECOMMENDATIONS")
    print("  " + "-" * 60)

    use_case_scorers = [
        ("mcp-doc-qa  (answer generation + RAG)", lambda r: r.mcp_doc_qa_score()),
        ("agent-rfx   (sheet classification + tools)", lambda r: r.agent_rfx_score()),
        ("lightweight (simple tasks, low latency)", lambda r: max(0.0, 5 - (r.get_overall_latency() / 2000))),
    ]
    for label, scorer in use_case_scorers:
        ranked = top_for(scorer)
        print(f"\n  {label}:")
        for i, r in enumerate(ranked, 1):
            score = scorer(r)
            lat = r.get_overall_latency()
            json_rate = r.get_json_success_rate()
            print(f"    {i}. {r.label:<30} {_stars(score)}  "
                  f"JSON={_pct(json_rate)}  lat={lat:.0f}ms  {r.iq_range}")

    # Ollama structured output advisory
    ollama_failures = [
        r for r in good
        if r.provider == "ollama" and r.get_json_success_rate() < 0.8
    ]
    if ollama_failures:
        print("\n  [!] Ollama Structured Output Advisory")
        print("  " + "-" * 60)
        print("  The following Ollama models had <80% JSON success rate:")
        for r in ollama_failures:
            print(f"    - {r.label}  ({_pct(r.get_json_success_rate())})")
        print()
        print("  Per Ollama docs (https://docs.ollama.com/capabilities/structured-outputs):")
        print("  'It is ideal to also pass the JSON schema as a string in the prompt")
        print("   to ground the model's response.'")
        print()
        print("  Re-run with schema grounding enabled (default for Ollama):")
        print("  $ uv run python scripts/benchmark_providers.py --filter ollama --schema-in-prompt")
        print()
        print("  To fix in mcp-doc-qa: update PromptBuilder._compliance_system_template to append")
        print("  the ComplianceAwareAnswer schema when an Ollama provider is active.")


# ---------------------------------------------------------------------------
# JSON / CSV output
# ---------------------------------------------------------------------------

def _reports_to_json(reports: List[ProviderReport]) -> str:
    out = []
    for r in reports:
        tests_out = {}
        for name, tr in r.test_results.items():
            runs_out = []
            for rr in tr.runs:
                runs_out.append({
                    "success": rr.success,
                    "latency_ms": round(rr.latency_ms, 1),
                    "structured_returned": rr.structured_returned,
                    "schema_valid": rr.schema_valid,
                    "input_tokens": rr.input_tokens,
                    "output_tokens": rr.output_tokens,
                    "thinking_used": rr.thinking_used,
                    "error": rr.error or None,
                    "content_preview": rr.content_preview,
                })
            tests_out[name] = {
                "category": tr.test.category,
                "n_runs": tr.n_runs,
                "success_rate": round(tr.success_rate, 3),
                "json_success_rate": round(tr.json_success_rate, 3),
                "avg_latency_ms": round(tr.avg_latency_ms, 1),
                "p95_latency_ms": round(tr.p95_latency_ms, 1),
                "runs": runs_out,
            }
        out.append({
            "provider": r.provider,
            "model": r.model,
            "label": r.label,
            "priority": r.priority,
            "iq_range": r.iq_range,
            "min_iq": r.min_iq,
            "max_iq": r.max_iq,
            "tier": r.tier,
            "thinking_configured": r.thinking_configured,
            "skip_reason": r.skip_reason or None,
            "json_success_rate": round(r.get_json_success_rate(), 3),
            "avg_latency_ms": round(r.get_overall_latency(), 1),
            "mcp_doc_qa_score": round(r.mcp_doc_qa_score(), 2),
            "agent_rfx_score": round(r.agent_rfx_score(), 2),
            "tests": tests_out,
        })
    return json.dumps(out, indent=2, ensure_ascii=False)


def _reports_to_csv(reports: List[ProviderReport]) -> str:
    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "provider", "model", "iq_min", "iq_max", "tier", "priority",
        "thinking", "json_success_rate_pct", "avg_latency_ms",
        "mcp_doc_qa_score", "agent_rfx_score",
        "ping_ok", "rfx_structured_json_pct", "rfx_rag_json_pct",
        "sheet_classification_json_pct", "multilingual_ok",
    ])
    for r in reports:
        def rate(name: str) -> str:
            tr = r.test_results.get(name)
            return f"{tr.json_success_rate * 100:.0f}" if tr else "--"
        def ok(name: str) -> str:
            tr = r.test_results.get(name)
            return "1" if (tr and tr.success_rate > 0.5) else ("--" if not tr else "0")
        writer.writerow([
            r.provider, r.model, r.min_iq, r.max_iq,
            r.tier, r.priority, r.thinking_configured,
            f"{r.get_json_success_rate() * 100:.0f}",
            f"{r.get_overall_latency():.0f}",
            f"{r.mcp_doc_qa_score():.2f}",
            f"{r.agent_rfx_score():.2f}",
            ok("ping"), rate("rfx_structured"), rate("rfx_rag"),
            rate("sheet_classification"), ok("multilingual"),
        ])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LLM providers defined in llm_providers.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--providers-file", "-f",
        default=None,
        help=(
            "Path to llm_providers.yaml. "
            "Defaults to LLM_PROVIDERS_FILE env var, then searching up from cwd."
        ),
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only benchmark providers whose 'provider:model' string contains this substring",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include providers marked enabled: false",
    )
    parser.add_argument(
        "--iq",
        type=int,
        default=None,
        metavar="LEVEL",
        help="Only benchmark providers eligible for this intelligence level (0-10)",
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=3,
        help="Number of repetitions for multi-run tests (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-call timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--schema-in-prompt",
        action="store_true",
        help=(
            "Append JSON schema to system prompt for ALL providers. "
            "For Ollama this is already the default (per docs recommendation)."
        ),
    )
    parser.add_argument(
        "--no-schema-in-prompt-ollama",
        action="store_true",
        help="Disable automatic schema-in-prompt grounding for Ollama providers",
    )
    parser.add_argument(
        "--test",
        action="append",
        metavar="NAME",
        dest="test_filter",
        help="Run only tests with this name (repeatable, e.g. --test ping --test rfx_structured)",
    )
    parser.add_argument(
        "--output", "-o",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-run verbose output (only print summary table)",
    )
    return parser.parse_args()


def _find_providers_file(hint: Optional[str]) -> Optional[str]:
    """Resolve the providers file path."""
    if hint:
        return hint
    env_path = os.getenv("LLM_PROVIDERS_FILE")
    if env_path and Path(env_path).exists():
        return env_path
    # Search cwd and common parent dirs
    search_paths = [
        Path.cwd() / "llm_providers.yaml",
        Path.cwd() / "llm_providers.yml",
        Path.cwd().parent / "llm_providers.yaml",
        _ROOT / "llm_providers.yaml",
    ]
    for p in search_paths:
        if p.exists():
            return str(p)
    return None


def main() -> None:
    args = _parse_args()

    # Ensure stdout handles unicode gracefully on Windows
    if hasattr(sys.stdout, "encoding") and sys.stdout.encoding and \
            sys.stdout.encoding.lower() in ("cp1252", "ascii", "cp850"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # Suppress noisy log output that clutters benchmark results:
    # - provider_registry: missing-credential warnings (handled via is_configured())
    # - tracing: "No active span" context errors (irrelevant in CLI benchmark)
    # - service_usage: per-call token usage lines (noise when running many tests)
    # - langfuse: "Context error: No active span" warnings from langfuse SDK
    #
    # IMPORTANT: force-import langfuse.logger first so its module-level
    # langfuse_logger.setLevel(logging.WARNING) runs BEFORE we set CRITICAL.
    # If we set CRITICAL before the import, the import will reset it to WARNING.
    import logging
    try:
        import langfuse.logger  # noqa: F401 - triggers module-level setup
    except Exception:
        pass
    for _noisy in (
        "core_lib.llm.provider_registry",
        "core_lib.tracing",
        "core_lib.tracing.service_usage",
        "opentelemetry",
        "langfuse",   # suppresses "Context error: No active span" warnings
        "httpx",      # suppresses HTTP request noise from langfuse SDK
    ):
        logging.getLogger(_noisy).setLevel(logging.CRITICAL)

    # Resolve providers file
    providers_file = _find_providers_file(args.providers_file)
    if not providers_file:
        print(
            "ERROR: Could not find llm_providers.yaml. Use --providers-file to specify one.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.output == "table":
        print(f"\n[benchmark_providers] Loading: {providers_file}")

    # Load registry
    from core_lib.llm.provider_registry import ProviderRegistry
    registry = ProviderRegistry.from_file(providers_file)

    if not registry.providers:
        print("ERROR: No providers loaded from file.", file=sys.stderr)
        sys.exit(1)

    # Filter providers
    candidates = list(registry.providers)

    if args.iq is not None:
        candidates = registry.get_providers_for_level(args.iq)
        if args.output == "table":
            print(f"[benchmark_providers] IQ{args.iq} filter: {len(candidates)} matching providers")

    if not args.include_disabled:
        before = len(candidates)
        candidates = [p for p in candidates if p.enabled]
        if args.output == "table" and before != len(candidates):
            print(
                f"[benchmark_providers] Skipping {before - len(candidates)} disabled providers "
                f"(use --include-disabled to include)"
            )

    if args.filter:
        before = len(candidates)
        candidates = [
            p for p in candidates
            if args.filter.lower() in f"{p.provider}:{p.model}".lower()
        ]
        if args.output == "table":
            print(f"[benchmark_providers] --filter '{args.filter}': {before}->{len(candidates)} providers")

    # Check configured (credentials available)
    configured = []
    skipped_unconfigured = []
    for p in candidates:
        if p.is_configured():
            configured.append(p)
        else:
            skipped_unconfigured.append(p)

    if args.output == "table" and skipped_unconfigured:
        labels = ", ".join(f"{p.provider}:{p.model}" for p in skipped_unconfigured)
        print(
            f"[benchmark_providers] Skipping {len(skipped_unconfigured)} unconfigured "
            f"(missing credentials): {labels}"
        )

    if not configured:
        print("No configured providers to benchmark.", file=sys.stderr)
        sys.exit(1)

    if args.output == "table":
        print(
            f"[benchmark_providers] Benchmarking {len(configured)} provider(s) "
            f"with {args.runs} runs/test\n"
        )

    # Filter tests
    tests = list(BENCHMARK_TESTS)
    if args.test_filter:
        tests = [t for t in tests if t.name in args.test_filter]
        if not tests:
            print(f"ERROR: No tests matched filter: {args.test_filter}", file=sys.stderr)
            sys.exit(1)

    # Run benchmarks
    reports: List[ProviderReport] = []
    for config in sorted(configured, key=lambda p: p.priority):
        # Per Ollama docs: embed schema in prompt to ground small models
        use_schema_hint = args.schema_in_prompt or (
            config.provider == "ollama" and not args.no_schema_in_prompt_ollama
        )
        try:
            report = run_provider_benchmark(
                config=config,
                tests=tests,
                n_runs=args.runs,
                timeout_secs=args.timeout,
                schema_in_prompt=use_schema_hint,
                verbose=(args.output == "table" and not args.quiet),
            )
        except KeyboardInterrupt:
            print("\n[benchmark_providers] Interrupted -- printing partial results.")
            break
        except Exception as exc:  # pragma: no cover - runtime connectivity failures
            print(f"\n[ERROR] Provider {config.provider}:{config.model} crashed: {exc}")
            traceback.print_exc()
            rpt = ProviderReport(
                provider=config.provider,
                model=config.model,
                priority=config.priority,
                min_iq=config.min_intelligence_level,
                max_iq=config.max_intelligence_level,
                tier=config.tier or "--",
                thinking_configured=config.thinking_enabled,
                skip_reason=f"Crash: {exc}",
            )
            reports.append(rpt)
            continue
        reports.append(report)

    # Output
    if args.output == "json":
        print(_reports_to_json(reports))
    elif args.output == "csv":
        print(_reports_to_csv(reports))
    else:
        _print_summary_table(reports)
        _print_recommendations(reports)
        print()


if __name__ == "__main__":
    main()
