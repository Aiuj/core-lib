# Wake-on-LAN

This guide explains how `core-lib` Wake-on-LAN (WoL) works for **Infinity
embeddings** and **Ollama LLM** providers, and how to test it.

The same `WakeOnLanStrategy` class is used by both stacks — configuration
shape and behaviour are identical regardless of whether the sleeping host
runs an Infinity embedding server or an Ollama model server.

## What it does

When an Infinity embedding host times out or is temporarily unreachable, the client can:

1. Use a short initial timeout for likely-sleeping hosts
2. Send a WoL magic packet to configured target(s)
3. Either **wait** for the host to wake (classic mode), or **immediately route to a secondary server** for a configurable warmup window (non-blocking mode)
4. Retry with an optional post-wake timeout
5. Fall back to other configured Infinity URLs if needed

WoL is applied once per URL in-process to avoid repeated wake storms.

## Non-blocking warmup mode (recommended)

The classic mode blocks the first request for `wait_seconds` while the server powers on.  
For interactive workloads (chat, search, …) where latency matters you can instead configure `warmup_seconds` to get **zero-latency failover**:

```
request arrives
  └─ main server times out → WoL magic packet sent (fire-and-forget)
       └─ route to secondary server immediately ◀── user gets an answer now
            └─ after warmup_seconds the main server is ready
                 └─ next request goes back to main
```

Set `warmup_seconds` at the strategy level. Target properties can be placed
directly under `wake_on_lan` (shorthand) or nested under `targets` — both are equivalent:

```yaml
# Shorthand — target fields directly under wake_on_lan (simpler for single server)
# enabled defaults to true when the wake_on_lan block is present
wake_on_lan:
  warmup_seconds: 30           # route to secondary for 30 s after WoL fires
  mac_address: FC:34:97:9E:C8:AF
  broadcast_ip: 82.66.214.52
  port: 7777
  wait_seconds: 0              # don't block — we are routing to secondary instead
```

The `targets` list form is equivalent and useful when you need to send packets
to multiple machines or want to be explicit:

```yaml
# targets list — identical result for a single server
wake_on_lan:
  warmup_seconds: 30
  targets:
    - mac_address: FC:34:97:9E:C8:AF
      broadcast_ip: 82.66.214.52
      port: 7777
      wait_seconds: 0
```

And configure a secondary server in the URL list:

```yaml
embedding_providers:
  - provider: infinity
    priority: 1
    base_url: http://emb-high:7997      # powerful main server (may sleep)
    model: BAAI/bge-large-en-v1.5
    wake_on_lan:
      warmup_seconds: 30
      wait_seconds: 0
      mac_address: FC:34:97:9E:C8:AF
      broadcast_ip: 82.66.214.52
      port: 7777

  - provider: infinity
    priority: 2
    base_url: http://emb-low:7997       # always-on secondary (smaller / cheaper)
    model: BAAI/bge-small-en-v1.5
```

With this setup:
- Requests normally go to `emb-high` (main, powerful).
- When `emb-high` first times out a WoL packet is sent **and** the request is served by `emb-low` (secondary) within milliseconds.
- For the next 30 s all requests are routed to `emb-low` transparently.
- After 30 s the client tries `emb-high` again; once it responds it becomes the preferred server.

## Classic blocking configuration

Supported config (classic mode — blocks until host is up):

```yaml
# Shorthand form — classic blocking mode
wake_on_lan:
  mac_address: FC:34:97:9E:C8:AF
  broadcast_ip: 82.66.214.52
  port: 7777
  wait_seconds: 20
  retry_timeout_seconds: 8
```

Notes:
- **`enabled`** defaults to `true` whenever a non-empty `wake_on_lan` block is
  present — you can omit it entirely.  Set `enabled: false` explicitly to
  disable WoL while keeping the config (e.g. for local development).
- **Shorthand vs `targets` list** — target fields (`mac_address`, `broadcast_ip`, `port`,
  `wait_seconds`, `retry_timeout_seconds`, `max_attempts`) can be placed **directly under
  `wake_on_lan`** (shorthand, single server) or nested inside a `targets` list (needed for
  multiple physical machines).  Both forms are equivalent for a single target.
- **`host`** is optional when `wake_on_lan` is nested inside a single provider entry — the
  strategy is already scoped to one URL so there is nothing to disambiguate. Omit it.
  Only set `host` when sharing one config across multiple base URLs (e.g. via a global
  `INFINITY_WAKE_ON_LAN` env var) and each URL maps to a different physical machine.
- **`broadcast_ip`** is the UDP destination for the magic packet (default **`255.255.255.255`**):
  - A **unicast IP** (e.g. your router's WAN IP) — works from containers and remote servers.
    `SO_BROADCAST` is not set. **Recommended for remote/container deployments.**
  - A **directed-broadcast** (e.g. `192.168.1.255`) — LAN only, requires sender on same subnet.
  - `255.255.255.255` — limited broadcast, LAN only, blocked by routers and container bridges.
- Default WoL UDP port is `9` if not specified. Must match your router's port-forward rule.
- If `targets` is omitted, a single-target shorthand is supported at the top level.
- **`max_attempts`** (optional, default **1**) — how many times to send the magic packet
  before giving up.  Increase to `2` on unreliable networks, but keep it low to avoid
  flooding the network.
- **`initial_timeout_seconds`** (optional, default **2 s** when `enabled: true`) — short
  connect timeout used on the first probe to detect sleeping hosts quickly.  If the host
  responds within this window it is considered awake and no WoL packet is sent.  Override
  only if your network needs a longer probe window (e.g. `5` for high-latency links).
- **`warmup_seconds`** (strategy-level, optional) — enables non-blocking mode.  Instead of
  sleeping until the host wakes up, the client routes requests to secondary providers for
  this many seconds and then retries the main host.  Set `wait_seconds: 0` on the target
  when using this mode.  See the [Non-blocking warmup mode](#non-blocking-warmup-mode-recommended)
  section above for a full example.

## LLM / Ollama usage (non-blocking warmup mode)

Configure `wake_on_lan` on the Ollama provider entry in your LLM provider
list.  When a request times out or is refused, a WoL packet is fired and
`FallbackLLMClient` immediately routes the request to the next configured
provider for the duration of `warmup_seconds`, then returns to the main
server transparently.

```python
from core_lib.llm import FallbackLLMClient

client = FallbackLLMClient.from_config([
    {
        "provider": "ollama",
        "model": "qwen3:8b",
        "host": "http://powerspec:11434",   # powerful sleeping server
        "priority": 1,
        "wake_on_lan": {
            "warmup_seconds": 30,            # route to secondary for 30 s
            "mac_address": "FC:34:97:9E:C8:AF",
            "broadcast_ip": "82.66.214.52",  # router WAN IP
            "port": 7777,
            "wait_seconds": 0,               # fire-and-forget
        },
    },
    {
        "provider": "ollama",
        "model": "qwen3:1.7b",
        "host": "http://always-on:11434",   # always-on fallback
        "priority": 2,
    },
])

# Use normally — WoL and secondary routing are transparent
response = client.chat("Summarise this document")
```

Behaviour summary:
- **primary healthy**: requests go to `powerspec` (fast, big GPU).
- **primary times out**: WoL packet sent instantly; **this** request is served
  by `always-on` (small, always running) with no extra wait.
- **for the next 30 s**: all requests go to `always-on`.
- **after 30 s**: `powerspec` is tried again; if it responds it becomes primary
  again and its health-tracker entry is cleared automatically.

> **Note** — in this mode the primary Ollama provider is intentionally **not**
> marked as unhealthy by `FallbackLLMClient` (the warmup window handles
> recovery, so the provider remains eligible to become primary again once the
> window expires).

## Embeddings usage (YAML provider routing)

WoL is typically supplied through provider YAML for embeddings:

```yaml
embedding_providers:
  - provider: infinity
    base_url: http://emb-low:7997
    model: BAAI/bge-small-en-v1.5
    wake_on_lan:
      mac_address: FC:34:97:9E:C8:AF
      broadcast_ip: 82.66.214.52   # router WAN IP
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

`diagnose-wol` is installed as an entry-point when the package is installed.
It loads the `.env` from your application and runs several diagnostic modes.
Use it to verify config, test reachability, and exercise the full wake/retry
path without writing any test code.

```bash
# Print resolved config (default when no flags given)
diagnose-wol --show-config

# Check if each host is reachable (uses WoL initial timeout for sleeping hosts)
diagnose-wol --health

# Simulate WoL decisions without sending packets or real HTTP
diagnose-wol --dry-run

# Send a magic packet to a configured host manually
diagnose-wol --send-wol --host powerspec

# Fire a real /embeddings request and observe retry/WoL behaviour live
diagnose-wol --probe

# Override URL and supply MAC ad-hoc (no config required)
diagnose-wol --probe \
    --url http://powerspec:7997 \
    --host powerspec \
    --mac FC:34:97:9E:C8:AF

# Point at a non-default .env file
diagnose-wol --show-config --env-file /path/to/.env

# Verbose output (shows debug logs from core-lib)
diagnose-wol --probe --verbose
```

When working inside the `core-lib` repo itself (before installing), run via `uv`:

```bash
uv run python core_lib/scripts/diagnose_wol.py --show-config
```

### Diagnosing common problems

| Symptom | Mode to run | What to look for |
|---------|-------------|------------------|
| Not sure if WoL is wired up | `--show-config` | Targets section, `enabled: true` |
| Host times out immediately | `--dry-run` | Check `initial_timeout_seconds` is short |
| Host never wakes | `--send-wol` | Look for "Magic packet sent"; check permissions |
| Retry after wake still fails | `--probe --verbose` | Check `retry_timeout_seconds` is long enough |
| Wrong host matched | `--dry-run` | Verify `host` matches hostname in URL exactly |
| Broadcast unreachable | `--send-wol` | Hint printed – may need `sudo` or a unicast `broadcast_ip` |
| Packet sent but host doesn't wake (remote/container) | `--send-wol --ip <WAN-IP>` | Set `broadcast_ip` to your router WAN IP instead of `255.255.255.255` |

### Verifying the magic packet arrives at the target

Before trusting the router port-forward rule or network path, confirm the
packet actually reaches the target machine while it is **still running**
(i.e. before it goes to sleep, or from a second terminal session):

**Option 1 — `tcpdump` (most detail, requires root/sudo)**

```bash
# Listen on the NIC that faces the sender; replace eth0 with your interface name
sudo tcpdump -i any udp port 9 or port 7777 -vv -X
```

Then send the packet from the sender side (`diagnose-wol --send-wol`).
You should see something like:

```
12:34:56.789012 IP 82.66.214.52.12345 > 192.168.1.204.9: UDP, length 102
        0x0000:  ffff ffff ffff         # 6 × 0xFF header
        0x0006:  fc34 979e c8af ...     # MAC repeated 16 ×
```

The payload is 102 bytes: 6 bytes of `0xFF` followed by the target MAC
address repeated 16 times.

**Option 2 — `nc` (netcat, no root needed)**

```bash
# -u = UDP, -l = listen, -p = port
nc -u -l -p 9
# or on some distros:
nc -u -l 9
```

A magic packet will appear as binary garbage in the terminal — that is
expected. Any output at all means the packet arrived.

> **Tip** — if nothing arrives, work backwards:
> 1. Check the port-forward rule on your router (external UDP → internal host UDP 9).
> 2. Try sending directly to the LAN IP instead of the WAN IP to isolate whether
>    the issue is the router rule or the packet generation itself.
> 3. On Linux, `sudo ufw allow 9/udp` (or equivalent) to ensure the port is not
>    blocked by the host firewall.

## Unit test scripts

Use `uv run` as standard in this repo.

### 1) Embeddings + Infinity WoL behaviour tests

```bash
uv run pytest -q tests/test_infinity_wake_on_lan.py
```

Covers:
- initial timeout override for sleeping hosts
- one-time wake behavior
- retry-after-wake path
- normal failover path for non-target hosts
- non-blocking warmup: `is_in_warmup()`, secondary routing, return-to-main

### 2) Ollama LLM WoL behaviour tests

```bash
uv run pytest -q tests/test_ollama_wake_on_lan.py
```

Covers:
- `OllamaProvider.is_in_warmup()` before/during/after window
- non-blocking re-raise (no second attempt on same host)
- `FallbackLLMClient` skips warmup providers and does not mark them unhealthy

### 2) YAML wiring tests for embeddings/reranker provider configs

```bash
uv run pytest -q tests/test_vector_provider_yaml_config.py -k wake_on_lan
```

### 3) Full test suite (optional)

```bash
uv run pytest -q
```

## Related docs

- [docs/EMBEDDINGS_QUICK_REFERENCE.md](EMBEDDINGS_QUICK_REFERENCE.md)
- [docs/INFINITY_FAILOVER.md](INFINITY_FAILOVER.md)
- [docs/INFINITY_PROVIDER.md](INFINITY_PROVIDER.md)
- [docs/FALLBACK_LLM_CLIENT.md](FALLBACK_LLM_CLIENT.md)
- [docs/LLM_QUICK_REFERENCE.md](LLM_QUICK_REFERENCE.md)