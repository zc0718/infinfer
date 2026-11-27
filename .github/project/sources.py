#!/usr/bin/env python3
"""Where an external thing comes from, and what "absent" means.

Easy path: run it with no arguments for a table of every declared source.

A library, a compiler and an emulator are consumed the same way by this project's checks, and they
fail the same way when the machine lacks them. This module answers one question for all three:
*can I have it here, and if not, what do I say instead of a traceback?*

Kinds
-----
recipe     -- ConanCenter does not carry it; its CCI-shaped recipe lives under local-packages/.
toolchain  -- a compiler: an environment override, a house Conan package, or the PATH.
tool       -- an executable the checks shell out to (an emulator, for instance).

Every kind answers with `(value, how, absent_reason)`. A caller that gets `value is None` must skip
and print `absent_reason`; it must never raise, because "this machine does not have it" is not a
finding about the repository.
"""
import json
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTER = ROOT / ".github" / "project" / "package-sources.json"
TOOLCHAIN_REGISTER = ROOT / ".github" / "project" / "toolchains.json"


def load(path=REGISTER):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def load_tools(path=TOOLCHAIN_REGISTER):
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return data.get("tool_sources") or ()


def all_entries(data=None):
    data = data or load()
    out = []
    for entry in data.get("packages") or ():
        out.append({**entry, "kind": entry.get("kind") or "recipe"})
    for entry in list(data.get("tools") or ()) + list(load_tools()):
        out.append(entry)
    return out


def find(name, data=None):
    for entry in all_entries(data):
        if entry.get("name") == name:
            return entry
    return None


def _conan_cache_binary(binary):
    for pattern in (f"b/*/p/bin/{binary}", f"*/p/bin/{binary}"):
        for hit in sorted((pathlib.Path.home() / ".conan2/p").glob(pattern)):
            return str(hit)
    return None


def resolve(name, data=None):
    """(value, how, absent_reason). `value is None` means: skip, and say why."""
    entry = find(name, data)
    if entry is None:
        return None, "undeclared", f"'{name}' is not declared in package-sources.json"
    absent = entry.get("if_absent", "not available on this machine")

    if entry["kind"] == "recipe":
        rel = entry.get("recipe") or ""
        folder = ROOT / rel
        if rel and (folder / "all" / "conanfile.py").is_file():
            return str(folder / "all"), "in-tree recipe", ""
        return None, "in-tree recipe", absent

    env_key = entry.get("env") or ""
    if env_key and os.environ.get(env_key):
        return os.environ[env_key], f"${env_key}", ""

    for binary in entry.get("provides") or ():
        if entry["kind"] == "toolchain":
            hit = _conan_cache_binary(binary)
            if hit:
                return hit, "~/.conan2 cache", ""
        on_path = shutil.which(binary)
        if on_path:
            return on_path, "PATH", ""
    return None, entry.get("obtain", "unknown"), absent


def resolve_binary(binary, data=None):
    """Find whichever declared entry provides `binary`. Used by callers that know the executable
    name but not the register's entry name -- `isa_gate` is the first of those."""
    for entry in all_entries(data):
        if binary in (entry.get("provides") or ()):
            return resolve(entry["name"], data)
    return None, "undeclared", f"no declared source provides '{binary}'"


def main():
    data = load()
    entries = all_entries(data)
    missing = 0
    for entry in entries:
        value, how, reason = resolve(entry["name"], data)
        if value:
            print(f"[sources] {entry['name']:<15} {entry['kind']:<9} {value}  ({how})")
        else:
            missing += 1
            print(f"[sources] {entry['name']:<15} {entry['kind']:<9} absent -- {reason}")
    print(f"[sources] OK ({len(entries) - missing} available, {missing} absent; "
          f"absent ones are the documented state)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
