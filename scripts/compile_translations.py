#!/usr/bin/env python3
"""
Compile translation files (.po -> .mo) for core-lib.
This script uses Python's built-in msgfmt functionality.
"""
import os
import subprocess
import sys
from pathlib import Path

def compile_translations():
    """Compile all .po files to .mo files."""
    locale_dir = Path(__file__).parent.parent / "core_lib" / "locale"
    
    if not locale_dir.exists():
        print(f"❌ Locale directory not found: {locale_dir}")
        return 1
    
    compiled = 0
    failed = 0
    
    for po_file in locale_dir.rglob("*.po"):
        mo_file = po_file.with_suffix(".mo")
        
        try:
            # Try using msgfmt first (gettext)
            result = subprocess.run(
                ["msgfmt", str(po_file), "-o", str(mo_file)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"✅ Compiled: {po_file.relative_to(locale_dir)}")
                compiled += 1
            else:
                raise Exception(result.stderr)
        except (FileNotFoundError, Exception) as e:
            # Fallback: Use Python's msgfmt module
            try:
                import polib
                po = polib.pofile(str(po_file))
                po.save_as_mofile(str(mo_file))
                print(f"✅ Compiled (via polib): {po_file.relative_to(locale_dir)}")
                compiled += 1
            except ImportError:
                print(f"⚠️  Skipping {po_file.relative_to(locale_dir)} - install gettext or polib")
                failed += 1
            except Exception as e2:
                print(f"❌ Failed to compile {po_file.relative_to(locale_dir)}: {e2}")
                failed += 1
    
    print(f"\n📊 Summary: {compiled} compiled, {failed} failed")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(compile_translations())
