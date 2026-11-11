#!/usr/bin/env python3
"""Create or verify the SHA-256 digest of a release's tracked Git tree."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

MANIFEST = "RELEASE_INTEGRITY.json"
SCHEME = "git-tree-sha256-v1"


def git(*args):
    return subprocess.check_output(["git", *args])


def digest_tree(tree):
    entries = git("ls-tree", "-r", "-z", "--full-tree", tree).split(b"\0")
    digest = hashlib.sha256(b"git-tree-sha256-v1\0")
    process = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for entry in entries:
            if not entry:
                continue
            metadata, path = entry.split(b"\t", 1)
            mode, kind, oid = metadata.split(b" ")
            if path == MANIFEST.encode():
                continue
            if kind != b"blob":
                raise ValueError(f"Unsupported Git tree entry: {path!r} ({kind.decode()})")
            process.stdin.write(oid + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3 or header[1] != b"blob":
                raise ValueError(f"Could not read Git blob for {path!r}")
            size = int(header[2])
            digest.update(mode + b"\0" + str(len(path)).encode() + b"\0" + path + b"\0" + str(size).encode() + b"\0")
            remaining = size
            while remaining:
                chunk = process.stdout.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError(f"Truncated Git blob for {path!r}")
                digest.update(chunk)
                remaining -= len(chunk)
            if process.stdout.read(1) != b"\n":
                raise ValueError(f"Invalid Git blob framing for {path!r}")
    finally:
        process.stdin.close()
        process.stdout.close()
        process.wait()
    if process.returncode:
        raise ValueError("git cat-file failed")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("create", help="Hash the staged index and write the release manifest")
    verify = sub.add_parser("verify", help="Verify the manifest committed at a Git revision")
    verify.add_argument("revision", help="Commit or tag to verify")
    args = parser.parse_args()
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    if args.command == "create":
        tree = git("write-tree").decode().strip()
        digest = digest_tree(tree)
        (root / MANIFEST).write_text(json.dumps({"scheme": SCHEME, "sha256": digest}, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"{MANIFEST}: {digest}; stage this file before committing")
    else:
        revision = args.revision
        tree = git("rev-parse", "--verify", f"{revision}^{{tree}}").decode().strip()
        manifest = json.loads(git("show", f"{revision}:{MANIFEST}"))
        if set(manifest) != {"scheme", "sha256"} or manifest["scheme"] != SCHEME:
            raise ValueError("Unsupported or invalid release integrity manifest")
        actual = digest_tree(tree)
        if manifest["sha256"] != actual:
            raise ValueError(f"Release integrity mismatch: expected {manifest['sha256']}, computed {actual}")
        print(f"Verified {revision}: SHA-256 {actual}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"Release integrity error: {exc}", file=sys.stderr)
        sys.exit(1)
