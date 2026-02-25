# Wake-on-LAN (Infinity)

This guide explains how `core-lib` Wake-on-LAN (WoL) works for **Infinity embeddings** and how to test it.

## What it does

When an Infinity embedding host times out or is temporarily unreachable, the client can:

1. Use a short initial timeout for likely-sleeping hosts
2. Send a WoL magic packet to configured target(s)
3. Wait for the host to wake
4. Retry with an optional post-wake timeout
5. Fall back to other configured Infinity URLs if needed

WoL is applied once per URL in-process to avoid repeated wake storms.

## Configuration shape

Supported config:

```yaml
wake_on_lan:
  enabled: true
  initial_timeout_seconds: 2
  targets:
    - host: powerspec
      mac_address: FC:34:97:9E:C8:AF
      port: 7777
      broadcast_ip: 255.255.255.255
      wait_seconds: 20
      retry_timeout_seconds: 8
      max_attempts: 1
```

Notes:
- `host` is matched against the Infinity base URL host.
- If `targets` is omitted, a single-target shorthand is supported at the top level.
- Default WoL UDP port is `9` if not specified.

## Embeddings usage (YAML provider routing)

WoL is typically supplied through provider YAML for embeddings:

```yaml
embedding_providers:
  - provider: infinity
    base_url: http://emb-low:7997
    model: BAAI/bge-small-en-v1.5
    wake_on_lan:
      enabled: true
      initial_timeout_seconds: 2
      targets:
        - host: emb-low
          mac_address: FC:34:97:9E:C8:AF
          port: 7777
          wait_seconds: 0
          retry_timeout_seconds: 8
    priority: 2

  - provider: infinity
    base_url: http://emb-high:7997
    model: BAAI/bge-small-en-v1.5
    priority: 1
```

Then create your embedding client normally:

```python
from core_lib.embeddings import create_embedding_client

client = create_embedding_client(intelligence_level=5, usage="search")
```

## Diagnostic script

`scripts/diagnose_wol.py` loads the `.env` from your application and runs several
diagnostic modes. Use it to verify config, test reachability, and exercise the
full wake/retry path without writing any test code.

```bash
# Print resolved config (default when no flags given)
uv run python scripts/diagnose_wol.py --show-config

# Check if each host is reachable (uses WoL initial timeout for sleeping hosts)
uv run python scripts/diagnose_wol.py --health

# Simulate WoL decisions without sending packets or real HTTP
uv run python scripts/diagnose_wol.py --dry-run

# Send a magic packet to a configured host manually
uv run python scripts/diagnose_wol.py --send-wol --host powerspec

# Fire a real /embeddings request and observe retry/WoL behaviour live
uv run python scripts/diagnose_wol.py --probe

# Override URL and supply MAC ad-hoc (no config required)
uv run python scripts/diagnose_wol.py --probe \
    --url http://powerspec:7997 \
    --host powerspec \
    --mac FC:34:97:9E:C8:AF

# Point at a non-default .env file
uv run python scripts/diagnose_wol.py --show-config --env-file /path/to/.env

# Verbose output (shows debug logs from core-lib)
uv run python scripts/diagnose_wol.py --probe --verbose
```

### Diagnosing common problems

| Symptom | Mode to run | What to look for |
|---------|-------------|------------------|
| Not sure if WoL is wired up | `--show-config` | Targets section, `enabled: true` |
| Host times out immediately | `--dry-run` | Check `initial_timeout_seconds` is short |
| Host never wakes | `--send-wol` | Look for "Magic packet sent"; check permissions |
| Retry after wake still fails | `--probe --verbose` | Check `retry_timeout_seconds` is long enough |
| Wrong host matched | `--dry-run` | Verify `host` matches hostname in URL exactly |
| Broadcast unreachable | `--send-wol` | Hint printed – may need `sudo` or directed broadcast |

## Unit test scripts

Use `uv run` as standard in this repo.

### 1) Embeddings + Infinity WoL behavior tests

```bash
uv run pytest -q tests/test_infinity_wake_on_lan.py
```

Covers:
- initial timeout override for sleeping hosts
- one-time wake behavior
- retry-after-wake path
- normal failover path for non-target hosts

### 2) YAML wiring tests for embeddings/reranker provider configs

```bash
uv run pytest -q tests/test_vector_provider_yaml_config.py -k wake_on_lan
```

### 3) Full test suite (optional)

```bash
uv run pytest -q
```

## Related docs

- `docs/EMBEDDINGS_QUICK_REFERENCE.md`
- `docs/INFINITY_FAILOVER.md`
- `docs/INFINITY_PROVIDER.md`