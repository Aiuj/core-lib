# Native decision models

`core_lib.llm.DecisionClient` evaluates a shared state against typed questions.
Use it for capability routing, classification, yes/no judgments and rubric
scores. It returns native structured answers; it does not generate chat text,
tool arguments or documents. Chat still uses `LLMClient` / `FallbackLLMClient`.

## Providers and models

| Registry provider | Model | Where it runs | Use |
| --- | --- | --- | --- |
| `typesafe` | `jev-latest` | Hosted TypeSafe API | Requested default for evaluating Jev's current stable release. Requires `TYPESAFE_API_KEY`. |
| `typesafe` | `jev-preview` | Hosted TypeSafe API | Evaluate preview releases separately; an alias can change. |
| `typesafe` | A versioned ID, e.g. `jev-1.13.0` | Hosted TypeSafe API | Pin a benchmark or tuned decision threshold. Record the returned `model`, even when requesting an alias. |
| `ollama-decision` | `nimble` | Local Ollama server | Requested local alternative; no API key required. |

The TypeSafe provider calls `/v1/systemone`; `/v1/models` returns account-visible
model names and descriptions. The aliases and versioned-ID behavior are described
in [TypeSafe's model reference](https://docs.typesafe.ai/models). Test each portal
language on representative intents rather than assume identical accuracy.

Ollama's decision capability requires **Ollama 0.35.0 or later** and compatible
local weights. Install the requested model with `ollama pull nimble`. Ordinary
chat models do not acquire decision capability by changing the provider name.
See [Ollama decision setup](https://docs.ollama.com/capabilities/decision).

## Configuration

Put these entries under the existing `providers` key in `llm_providers.yaml`:

```yaml
providers:
  - provider: typesafe
    model: jev-latest
    api_key: ${TYPESAFE_API_KEY:-}
    host: ${TYPESAFE_BASE_URL:-https://api.typesafe.ai}
    priority: 10
    usage: [decision]
    timeout: 30
    max_retries: 2

  - provider: ollama-decision
    model: nimble
    host: ${OLLAMA_HOST:-http://127.0.0.1:11434}
    priority: 20
    usage: [decision]
    timeout: 60
    keep_alive: 5m
```

`ProviderRegistry.decision_providers` lists configured, enabled decision entries.
`ProviderRegistry.providers` remains the chat selection surface and excludes
decision-only entries, regardless of priority or usage tags. `all_providers`
still includes both kinds for configuration inspection. Startup chat probes
do not select decision models; connectivity checks use model listings without
inference. Ollama model listing proves installation, not scoring support.

`ProviderConfig.to_client()` returns the correct client type; explicit
`to_decision_client()` makes the decision contract clear. Decision configurations
cannot be converted into chat configurations. No generation controls such as
temperature, thinking or tools are sent to System One.

With no YAML configuration, environment loading registers TypeSafe when
`TYPESAFE_API_KEY` exists. Set `OLLAMA_DECISION_MODEL=nimble` to register local
decisions alongside an existing chat model. Direct clients also accept
`OLLAMA_BASE_URL`, `OLLAMA_HOST`, `TYPESAFE_BASE_URL` and explicit credentials.
No caller needs to copy a key into source code.

For a ready-to-run synthetic comparison from the application directory:

```powershell
uv run ../core-lib/examples/example_decision_usage.py --config llm_providers.yaml
uv run ../core-lib/examples/example_decision_usage.py --config llm_providers.yaml --provider typesafe --list-models
```

## Evaluate both backends

```python
from core_lib.llm import Choice, Noul, Score, create_decision_client
from core_lib.llm.provider_registry import ProviderRegistry

registry = ProviderRegistry.from_file("llm_providers.yaml")
questions = {
    "capability": Choice(
        "Which capability should handle the request?",
        {
            "knowledge": "Find evidence or answer from company knowledge",
            "authoring": "Draft or revise an application document",
            "unsupported": "Work outside this application's capabilities",
        },
    ),
    "needs_context": Noul("Does the request refer to an unidentified object?"),
    "ambiguity": Score("How ambiguous is the request?", ["Clear", "Some ambiguity", "Very ambiguous"]),
}

for provider in ("typesafe", "ollama-decision"):
    with create_decision_client(provider, registry=registry) as client:
        result = client.decide(
            state={"request": "Draft an accompanying document", "object_type": "project"},
            questions=questions,
        )
        print(provider, result["model"], result["answers"], result["usage"], result["latency_ms"])
```

`system_one()` is an alias for `decide()`. Plain question dictionaries are also
accepted. Results preserve `answers`, `usage` and the resolved `model`, and add
`provider`, `requested_model` and `latency_ms`. Usage is logged through existing
core-lib tracing without the state, question text, API key or raw response.

For an explicit client without YAML:

```python
with create_decision_client("typesafe", model="jev-latest") as client:
    models = client.list_models()  # names, descriptions and release metadata
```

## Contract and reliability

Choice returns a selected label and option probabilities; Noul returns a yes/no
probability; Score returns a probability-weighted position on ordered levels.
The driver validates answer IDs/types, option membership, probability bounds
and distributions, confidence, score range and usage. Instructions may be text
or structured JSON. TypeSafe supports up to 255 Choice options and 10 Score
levels, documented in its [API reference](https://docs.typesafe.ai/api).

Local Ollama supports 2–26 Choice options / Score levels. Text-only request
bodies must fit within 64 KiB, counted as UTF-8 JSON by the driver. Model context
limits remain enforced by the server. This driver supports text/JSON state;
vision inputs are outside its current contract. See the
[Ollama System One API](https://docs.ollama.com/api/systemone).

HTTP 429/529 and transient gateway/server statuses use bounded backoff.
`Retry-After` is respected; a delay exceeding `max_retry_delay` surfaces the
error instead of retrying early. Transport timeouts are not automatically
replayed because the original inference may already have consumed tokens.
`DecisionError.status_code` is available for HTTP failures; error messages omit
request/response bodies. Invalid requests/results do not become routing choices.

There is intentionally no automatic cloud/local failover. Local evaluation must
not silently transmit its state to TypeSafe, and a benchmark must measure the
provider it requested. Select the fallback explicitly in the application.

Compare both drivers with the same allowed capability descriptions, minimized
state and held-out language/intent fixtures. Record resolved model, accuracy,
required-tool recall, cold/warm latency, tokens and provider costs. Confidence
is a decision signal, not permission to execute a tool or evidence that the
chosen action is correct. Pin a tested model before relying on tuned thresholds.
