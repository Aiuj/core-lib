#!/usr/bin/env python3
"""Clear cache entries managed by core-lib cache registry.

Usage examples:
  uv run python scripts/clear_cache.py
  uv run python scripts/clear_cache.py --scope global
  uv run python scripts/clear_cache.py --scope company --company-id 123
  uv run python scripts/clear_cache.py --flushdb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core_lib.config import StandardSettings, initialize_settings
from core_lib.cache import (
    set_cache,
    get_cache,
    get_cache_client,
    cache_clear_all,
    cache_clear_global,
    cache_clear_company,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clear core-lib cache entries")
    parser.add_argument(
        "--scope",
        choices=["all", "global", "company"],
        default="all",
        help="Cache scope to clear (default: all)",
    )
    parser.add_argument(
        "--company-id",
        default=None,
        help="Company ID to clear (required when --scope company)",
    )
    parser.add_argument(
        "--flushdb",
        action="store_true",
        help="Flush entire Redis/Valkey DB (dangerous; ignores scope)",
    )
    parser.add_argument(
        "--include-llm-health",
        action="store_true",
        help="Also clear LLM provider health keys (llm:health:*)",
    )
    parser.add_argument(
        "--dotenv-dir",
        default=None,
        help="Directory containing .env to load (defaults to current working directory search)",
    )
    return parser.parse_args()


def clear_llm_health(client: object) -> int:
    """Delete LLM provider health keys and return deleted key count."""
    try:
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = client.scan(cursor=cursor, match="llm:health:*", count=200)
            if keys:
                deleted += int(client.delete(*keys))
            if cursor == 0:
                break
        return deleted
    except Exception:
        return 0


def main() -> int:
    args = parse_args()

    if args.scope == "company" and not args.company_id:
        print("Error: --company-id is required when --scope company")
        return 2

    dotenv_paths = [Path(args.dotenv_dir)] if args.dotenv_dir else None
    initialize_settings(
        settings_class=StandardSettings,
        force=True,
        setup_logging=True,
        dotenv_paths=dotenv_paths,
    )

    ok = set_cache(provider="auto")
    if not ok:
        print("Cache is not configured or unavailable (set_cache returned False).")
        return 1

    cache = get_cache()
    if cache is False:
        print("Cache is disabled or unavailable.")
        return 1

    client = get_cache_client()
    if client is None:
        print("No underlying cache client available.")
        return 1

    try:
        if args.flushdb:
            client.flushdb()
            print("✅ Cache DB flushed successfully (flushdb).")
            return 0

        if args.scope == "all":
            cache_clear_all()
            print("✅ Cleared all registered cache entries.")
        elif args.scope == "global":
            cache_clear_global()
            print("✅ Cleared global cache entries.")
        else:
            cache_clear_company(args.company_id)
            print(f"✅ Cleared cache entries for company_id='{args.company_id}'.")

        if args.include_llm_health:
            deleted = clear_llm_health(client)
            print(f"✅ Cleared LLM provider health keys: {deleted}")

        return 0
    except Exception as exc:
        print(f"❌ Failed to clear cache: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
