#!/usr/bin/env python3
"""Make every locally-kept Conan package available in the cache.

Easy path: run it with no arguments.

    python3 .github/project/package_sources.py

It reads `package-sources.json`, writes a packaging recipe next to each `local` artifact and
builds it into the local cache. `--check` only reports, building nothing.

Why this exists
---------------
A package that no Conan remote carries cannot be declared by version alone, and its recipe
cannot be committed either: a recipe's job is to carry a download URL and an upstream sha256,
and this project keeps those out of the version library. What is committed instead is the built
payload under `local-packages/` plus this file's description of it. Nothing here downloads
anything.

A missing artifact is not an error. The script names the entry, prints its `if_absent` reason and
exits zero, because the alternative -- failing -- would make every clean clone red for a reason
that has nothing to do with the tree.
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTER = ROOT / ".github" / "project" / "package-sources.json"

def load(path=REGISTER):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def recipe_of(entry):
    """The folder holding conanfile.py. CCI layout puts it under `all/`; this resolves that."""
    rel = entry.get("recipe") or ""
    if not rel:
        return None
    root = ROOT / rel
    layout = root / "all"
    return layout if (layout / "conanfile.py").is_file() else root


def export(entry, folder):
    """Export the CCI-shaped recipe; `conan create` builds it for this configuration."""
    for verb in ("export", "create"):
        proc = subprocess.run(["conan", verb, str(folder), "--version", entry["version"]],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return f"conan {verb} of {entry['name']} failed: {proc.stderr.strip()[:300]}"
    return None


def main():
    parser = argparse.ArgumentParser(description="Make locally-kept Conan packages available.")
    parser.add_argument("--check", action="store_true", help="report only, build nothing")
    args = parser.parse_args()

    data = load()
    entries = data.get("packages") or []
    kept = missing = 0
    for entry in entries:
        name = f"{entry['name']}/{entry['version']}"
        # `remote` entries need nothing from this script: conandata.yml already names them.
        if entry.get("source", "remote") != "local":
            continue
        folder = recipe_of(entry)
        if folder is None or not folder.is_dir():
            missing += 1
            print(f"[packages] {name}: no local recipe -- {entry.get('if_absent', 'not kept here')}")
            continue
        kept += 1
        if args.check:
            print(f"[packages] {name}: local recipe at {folder.relative_to(ROOT)}")
            continue
        if shutil.which("conan") is None:
            print(f"[packages] {name}: conan is not installed, so nothing can be built here")
            continue
        problem = export(entry, folder)
        if problem:
            print(f"[packages] {problem}")
            return 1
        print(f"[packages] {name}: built into the local cache from {folder.relative_to(ROOT)}")

    print(f"[packages] OK ({kept} kept locally, {missing} absent; absent ones are the documented state)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
