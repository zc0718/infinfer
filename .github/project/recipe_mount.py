#!/usr/bin/env python3
"""The second Conan track: recipes CCI does not carry, mounted from a declared path.

Why a mount point rather than a commit
-------------------------------------
A recipe's job is to carry a download URL and an upstream sha256, and this project keeps those
out of the version library. So the recipe cannot live in the tree, which used to mean CI could
never build the leg that needs it. The mount makes that leg reachable without smuggling the
recipe in: `.github/project/recipe-mount.json` names the path, and this script turns the path
into a package in the local cache when it is there.

What "missing" means
---------------------
Not an error. A runner without the mount prints the `if_missing` reason for that mount and the
exit code stays 0, because the alternative -- failing -- would make every clean clone red for a
reason that has nothing to do with the tree. The leg that needs the mount is the one that must
say it was skipped, and it does.

Usage
-----
  python3 .github/project/recipe_mount.py --check     # what is mounted, what is not
  python3 .github/project/recipe_mount.py --export    # export every present mount
  python3 .github/project/recipe_mount.py --build     # export and build every present mount
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTER = ROOT / ".github" / "project" / "recipe-mount.json"


def load(path=REGISTER):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def resolve(entry):
    """Absolute path of a mount, or None when it is not on this machine."""
    rel = entry.get("recipe") or entry.get("artifact") or ""
    if not rel:
        return None
    candidate = ROOT / rel
    return candidate if candidate.exists() else None


def report(data):
    present = missing = 0
    for entry in data.get("mounts", []):
        path = resolve(entry)
        name = f"{entry['package']}/{entry['version']}"
        if path is None:
            missing += 1
            print(f"[mount] absent  {name}: {entry.get('if_missing', 'not mounted')}")
        else:
            present += 1
            print(f"[mount] present {name}: {path.relative_to(ROOT)}")
    return present, missing


def export(entry, path, build=False):
    ref = [str(path), "--version", entry["version"]]
    for step, verb in ((False, "export"), (True, "create")):
        if step and not build:
            break
        proc = subprocess.run(["conan", verb, *ref], capture_output=True, text=True)
        if proc.returncode != 0:
            return f"conan {verb} of {path.relative_to(ROOT)} failed: {proc.stderr.strip()[:300]}"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    if shutil.which("conan") is None:
        print("[mount] conan is not installed, so no mount can be used on this machine")
        return 0

    data = load()
    present, missing = report(data)
    if args.export or args.build:
        for entry in data.get("mounts", []):
            path = resolve(entry)
            if path is None:
                continue
            problem = export(entry, path, build=args.build)
            if problem:
                print(f"[mount] {problem}")
                return 1
            print(f"[mount] {'built' if args.build else 'exported'} {entry['package']}/{entry['version']}")
    print(f"[mount] OK ({present} present, {missing} absent; absent mounts are the documented state)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
