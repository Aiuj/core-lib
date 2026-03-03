# Wake-on-LAN

Use Wake-on-LAN (WoL) in `core-lib` to wake sleeping Infinity/Ollama hosts when a request times out.
This guide focuses on practical setup, validation, and troubleshooting.

## What it does

`core-lib` uses the same `WakeOnLanStrategy` for:
- Infinity embeddings providers
- Ollama LLM providers

On timeout/refusal, the client can:
1. Send a magic packet
2. Either wait for wake (**blocking mode**) or route immediately to secondary providers (**non-blocking warmup mode**)
3. Retry/return to primary after warmup

WoL is applied once per URL in-process to prevent repeated wake storms.

## Recommended mode: non-blocking warmup

For interactive workloads, use `warmup_seconds` and set `wait_seconds: 0`.
This avoids blocking user requests while the primary host wakes.

### Embeddings YAML example

```yaml
embedding_providers:
  - provider: infinity
    priority: 1
    base_url: http://emb-high:7997         # primary (may sleep)
    model: BAAI/bge-large-en-v1.5
    wake_on_lan:
      warmup_seconds: 30                   # route to secondary for 30s
      wait_seconds: 0                      # non-blocking
      mac_address: FC:34:97:9E:C8:AF
      broadcast_ip: 82.66.214.52
      port: 7777

  - provider: infinity
    priority: 2
    base_url: http://emb-low:7997          # always-on secondary
    model: BAAI/bge-small-en-v1.5
```

Behavior:
- Primary healthy: requests use `emb-high`
- Primary timeout: WoL packet sent, request immediately served by `emb-low`
- During warmup window: requests continue on `emb-low`
- After warmup: primary is tried again and resumes if healthy

### Ollama fallback example

```python
from core_lib.llm import FallbackLLMClient

client = FallbackLLMClient.from_config([
    {
        "provider": "ollama",
        "model": "qwen3:8b",
        "host": "http://powerspec:11434",   # primary (may sleep)
        "priority": 1,
        "wake_on_lan": {
            "warmup_seconds": 30,
            "wait_seconds": 0,
            "mac_address": "FC:34:97:9E:C8:AF",
            "broadcast_ip": "82.66.214.52",
            "port": 7777,
        },
    },
    {
        "provider": "ollama",
        "model": "qwen3:1.7b",
        "host": "http://always-on:11434",
        "priority": 2,
    },
])
```

## Blocking mode (simple alternative)

Use this when you prefer waiting for the primary host:

```yaml
wake_on_lan:
  mac_address: FC:34:97:9E:C8:AF
  broadcast_ip: 82.66.214.52
  port: 7777
  wait_seconds: 20
  retry_timeout_seconds: 8
```

## Key config notes

- `enabled` defaults to `true` when `wake_on_lan` is present.
- `warmup_seconds` enables non-blocking mode; pair with `wait_seconds: 0`.
- `initial_timeout_seconds` defaults to `2` when WoL is enabled.
- `port` defaults to `9` if omitted.
- `max_attempts` defaults to `1`.
- `broadcast_ip` guidance:
  - Router WAN IP (unicast): best for remote/container deployments
  - Directed broadcast (e.g. `192.168.1.255`): LAN only
  - `255.255.255.255`: LAN only, often blocked by routers/bridges
- Use `targets` only when waking multiple machines; single-target shorthand is enough for most setups.

## Quick validation

```bash
# Show resolved WoL config
diagnose-wol --show-config

# Check host reachability
diagnose-wol --health

# Simulate behavior without real network/HTTP
diagnose-wol --dry-run

# Send a magic packet
diagnose-wol --send-wol --host 82.66.214.52

# Exercise full request path
diagnose-wol --probe --verbose
```

Inside this repo (without installing package entry points):

```bash
uv run python core_lib/scripts/diagnose_wol.py --show-config
```

## End-to-end runbook (VPS + Podman + LLM)

Use this sequence to validate the complete remote wake flow for an app running in a Podman container.

### 0) Preconditions

- Target server WoL is enabled in BIOS/UEFI and OS NIC settings.
- Router forwards external UDP port (example: `7777`) to target LAN host UDP WoL port (`9` or custom).
- `wake_on_lan.broadcast_ip` is set to router WAN/public IP (not `255.255.255.255`) for remote/container use.
- App container has correct `mac_address`, `broadcast_ip`, `port`.

### 1) Verify WoL config from inside the app container

```bash
# On VPS host
sudo podman ps
sudo podman exec -it doc-qa diagnose-wol --show-config

# If diagnose-wol entry point is not installed in container
sudo podman exec -it <app_container> python -m core_lib.scripts.diagnose_wol --show-config
```

Expected:
- WoL target appears with correct MAC/IP/port.
- `enabled` is true (or implied by block presence).

### 2) Validate magic packet is sent from container path

```bash
# sudo podman exec -it <app_container> diagnose-wol --send-wol --host <router_wan_ip>
sudo podman exec -it doc-qa diagnose-wol --send-wol --host 82.66.214.52 --verbose
```

Expected:
- Command reports magic packet sent.
- No DNS/permission/socket errors.

### 3) Confirm packet arrives on target host (while target is awake)

Run one of these on target machine before sending packet:

```bash
# Preferred (detailed)
sudo tcpdump -i any udp port 9 or port 7777 -vv -X

# Alternative (quick)
nc -u -l -p 9
```

Then repeat step 2 from container.

Expected:
- `tcpdump` shows UDP packet with WoL signature (6 bytes `FF` + MAC repeated 16 times).
- If nothing arrives, issue is network path/router/firewall, not app logic.

### 4) Test real wake from sleep/off state

1. Put target server to sleep:

```bash
sudo systemctl suspend
```

2. Trigger WoL from container (`--send-wol`) or trigger an LLM request that should cause WoL.
3. Watch target boot/wake and then verify LLM endpoint is reachable.

Example post-wake check:

```bash
curl http://<target_lan_or_service_ip>:11434/api/tags
```

### 5) Validate LLM failover/warmup behavior through app

For non-blocking mode (`warmup_seconds` + `wait_seconds: 0`):

1. Keep primary LLM host sleeping/unreachable.
2. Send a real app LLM request.
3. Follow app container logs:

```bash
sudo podman logs -f <app_container>
```

Expected sequence:
- timeout/refusal on primary
- WoL send event
- request served by secondary provider
- after warmup window, primary retried and restored when healthy

### 6) Isolate each layer if it still fails

- **Layer A (config):** `diagnose-wol --show-config`
- **Layer B (send path):** `diagnose-wol --send-wol --verbose`
- **Layer C (network delivery):** `tcpdump`/`nc` on target
- **Layer D (service readiness):** `curl` target LLM API after wake
- **Layer E (application routing):** app logs during live LLM request

## Troubleshooting

| Symptom | Check |
|---------|-------|
| WoL seems ignored | `diagnose-wol --show-config` and confirm `wake_on_lan` is loaded |
| Host never wakes | `--send-wol`; validate MAC, UDP port, router forwarding, firewall |
| Retry still fails after wake | Increase `retry_timeout_seconds` |
| Requests pause too long | Use non-blocking mode: `warmup_seconds` + `wait_seconds: 0` |
| Works on LAN, fails in container/remote | Set `broadcast_ip` to router WAN IP (not `255.255.255.255`) |
| Packet "sent" but never seen on target | Run `tcpdump` on target while sending; verify router UDP forward and host firewall |
| Packet arrives but machine does not wake | Recheck BIOS/UEFI WoL + NIC power settings; verify MAC is for active NIC |
| Wakes correctly but first LLM request still fails | Increase `warmup_seconds` and/or request timeout to allow model/service cold start |

## Useful tests

```bash
uv run pytest -q tests/test_infinity_wake_on_lan.py
uv run pytest -q tests/test_ollama_wake_on_lan.py
uv run pytest -q tests/test_vector_provider_yaml_config.py -k wake_on_lan
```

## Related docs

- [docs/INFINITY_PROVIDER.md](INFINITY_PROVIDER.md)
- [docs/INFINITY_FAILOVER.md](INFINITY_FAILOVER.md)
- [docs/FALLBACK_LLM_CLIENT.md](FALLBACK_LLM_CLIENT.md)
- [docs/EMBEDDINGS_QUICK_REFERENCE.md](EMBEDDINGS_QUICK_REFERENCE.md)
- [docs/LLM_QUICK_REFERENCE.md](LLM_QUICK_REFERENCE.md)