"""Compare native decision backends with synthetic input, or list models.

Run from an application directory containing llm_providers.yaml:
    uv run ../core-lib/examples/example_decision_usage.py --config llm_providers.yaml
    uv run ../core-lib/examples/example_decision_usage.py --config llm_providers.yaml --provider typesafe --list-models
"""

import argparse
import json

from core_lib.llm import Choice, Noul, Score, DecisionError, create_decision_client
from core_lib.llm.provider_registry import ProviderRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional llm_providers YAML/JSON path")
    parser.add_argument(
        "--provider",
        nargs="+",
        choices=["typesafe", "ollama-decision"],
        default=["typesafe", "ollama-decision"],
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List models without inference"
    )
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    registry = (
        ProviderRegistry.from_file(args.config)
        if args.config
        else ProviderRegistry.from_env()
    )
    questions = {
        "capability": Choice(
            "Which application capability fits the request?",
            {
                "knowledge": "Search company evidence and answer questions",
                "authoring": "Draft or revise a project document",
                "unsupported": "Tasks outside the application",
            },
        ),
        "needs_context": Noul("Is the referenced project unidentified?"),
        "ambiguity": Score(
            "How ambiguous is the request?",
            ["Clear", "Some ambiguity", "Very ambiguous"],
        ),
    }
    failures = 0
    for provider in args.provider:
        try:
            with create_decision_client(
                provider, registry=registry, timeout=args.timeout, max_retries=0
            ) as client:
                result = (
                    client.list_models()
                    if args.list_models
                    else client.decide(
                        state={
                            "request": "Draft an accompanying document for the selected project",
                            "object_type": "project",
                            "selected_project": "Demo project",
                        },
                        questions=questions,
                    )
                )
            print(
                json.dumps({"provider": provider, "result": result}, ensure_ascii=False)
            )
        except (ValueError, DecisionError) as exc:
            failures += 1
            print(json.dumps({"provider": provider, "error": str(exc)}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
