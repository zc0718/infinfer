#!/usr/bin/env python3
"""Project-owned scaffold contract: compare this tree with the pinned template.

The template ships `.github/misc/scaffold_check.py`, which compares every
template-owned file byte for byte and can never accept a declared fork: its
`main()` adds the tolerated-entry findings to the same problem list the byte
comparison produced and never subtracts the tolerated paths, so a tolerated file
that differs is reported twice and the run fails. That makes the first honest
fork (a board path, a workflow line) permanently red, and a permanently red gate
is the same as no gate.

This gate keeps the useful half of the contract and makes the fork list
*executable data* instead of prose (the pattern the foc project converged on):

  G1 every `checked` path matches the pinned upstream sha256, unless tolerated
  G2 a tolerated entry must exist locally and must actually differ from upstream
     (a stale tolerance hides the next real divergence, so it fails)
  G3 no local file on the template-owned surface is missing from `checked`
  G4 `tolerated` may not grow past `tolerance_budget`
  G5 every `template_text` path still hashes to the pinned value: `LICENSE` and `NOTICE` came
     from the template, are not on its checked surface, and were until now protected by nothing
     but discipline. Changing one is allowed -- it means changing the pin, which is a decision
     that has to appear in a diff instead of happening quietly.

`--pin` regenerates `checked` from the upstream tree, so the sha list is never
hand-typed. `--selftest` proves each rule can fail on its own fixture.

由项目自己拥有的 scaffold 契约：模板自带的 `scaffold_check.py` **永远无法接受声明
过的分叉**（`main()` 把 tolerated 校验结果与字节比对结果并进同一个列表，从不把
tolerated 路径减掉），于是第一个诚实的分叉就让门禁永久变红——而永久红等于没门禁。
本门禁保留契约中有用的那一半，并把分叉清单变成**可执行的数据**（foc 项目最终收敛
到的做法），四条规则见上。
"""
import argparse
import fnmatch
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / ".github/project/scaffold.json"
PATTERNS_FILE = ".github/misc/scaffold-paths.txt"
BASELINE = ROOT / ".github/project/scaffold-baseline"
UPSTREAM_URL = "https://github.com/HeT-FTI/fcpp.git"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_patterns(base):
    """Read the upstream-owned path patterns from the pinned tree."""
    path = Path(base) / PATTERNS_FILE
    if not path.is_file():
        raise SystemExit(f"scaffold_gate: {PATTERNS_FILE} missing under {base}")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def surface(base, patterns):
    """Template-owned files present in a tree, as posix paths.

    A git worktree is read through `git ls-files`, not the filesystem: an ignored
    artifact (`__pycache__/*.pyc` born from a local run) is not part of the
    contract and must not be mistaken for a new fork.
    """
    base = Path(base)
    if (base / ".git").exists():
        out = subprocess.run(
            ["git", "-C", str(base), "ls-files", "-z"],
            capture_output=True, text=True, check=True,
        ).stdout
        files = [item for item in out.split("\0") if item]
    else:
        files = [
            path.relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.is_file() and ".git" not in path.parts
            and "__pycache__" not in path.parts and path.suffix != ".pyc"
        ]
    return {f for f in files if any(fnmatch.fnmatchcase(f, p) for p in patterns)}


def load_manifest():
    if not MANIFEST.is_file():
        raise SystemExit(f"scaffold_gate: {MANIFEST} is missing")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def audit(local, upstream, manifest):
    """Apply G1-G4 to one (local, upstream) pair and return problem strings."""
    problems = []
    patterns = load_patterns(upstream)
    checked = manifest.get("checked", {})
    tolerated = {entry["path"]: entry for entry in manifest.get("tolerated", [])}
    up_surface = surface(upstream, patterns)
    local_surface = surface(local, patterns)

    for rel, expected in sorted(checked.items()):
        actual_path = Path(local) / rel
        if not actual_path.is_file():
            problems.append(f"G1 {rel}: missing locally")
            continue
        if actual_path.read_bytes() == (Path(upstream) / rel).read_bytes():
            if rel in tolerated:
                problems.append(f"G2 {rel}: tolerated but identical to upstream (stale entry)")
            continue
        if rel not in tolerated:
            problems.append(f"G1 {rel}: differs from upstream and is not tolerated")
        elif sha256(Path(upstream) / rel) != expected:
            problems.append(f"G1 {rel}: pinned sha256 does not match upstream bytes")

    for rel, entry in sorted(tolerated.items()):
        if rel not in checked:
            problems.append(f"G3 {rel}: tolerated but not in the checked set")
        if not str(entry.get("why", "")).strip() or not str(entry.get("exit", "")).strip():
            problems.append(f"G2 {rel}: tolerated entry needs a non-empty why and exit")

    for rel in sorted(local_surface - set(checked) - set(tolerated)):
        marker = "" if rel in up_surface else " (not in the pinned release)"
        problems.append(f"G3 {rel}: on the template surface but not in checked{marker}")

    budget = manifest.get("tolerance_budget")
    if isinstance(budget, int) and len(tolerated) > budget:
        problems.append(f"G4 tolerated grew to {len(tolerated)} > budget {budget}")

    # G5 -- the template's own prose, pinned by hash. This runs against the local tree only: the
    # upstream copy is fetched for G1-G3, but these two files are not on the template surface the
    # patterns describe, and that is exactly why nothing noticed them before.
    for rel, entry in sorted((manifest.get("template_text") or {}).items()):
        expected = str(entry.get("sha256", "")).strip()
        actual_path = Path(local) / rel
        if not expected:
            problems.append(f"G5 {rel}: pinned without a sha256")
        elif not actual_path.is_file():
            problems.append(f"G5 {rel}: missing locally")
        elif sha256(actual_path) != expected:
            problems.append(
                f"G5 {rel}: the bytes changed since the pin -- if this is intended, update "
                f"template_text.{rel}.sha256 in the same commit so the change is visible"
            )
        if not str(entry.get("why", "")).strip():
            problems.append(f"G5 {rel}: pinned without a why")
    return problems


def fetch_upstream(tag, workdir):
    """Offline baseline first, then an explicit source, then a shallow clone."""
    for candidate in (BASELINE / f"v{tag}", BASELINE):
        if candidate.is_dir() and (candidate / PATTERNS_FILE).is_file():
            return candidate
    explicit = ROOT / "SCAFFOLD_SOURCE_UNSET" if not _env_source() else Path(_env_source())
    if explicit.is_dir():
        return explicit
    dest = Path(workdir) / "upstream"
    done = subprocess.run(
        ["git", "clone", "-q", "--depth", "1", "--branch", f"v{tag}", UPSTREAM_URL, str(dest)],
        text=True, capture_output=True,
    )
    if done.returncode != 0:
        raise SystemExit(f"scaffold_gate: cannot fetch v{tag}: {done.stderr.strip()}")
    return dest


def _env_source():
    import os
    return os.environ.get("SCAFFOLD_SOURCE", "")


def pin(upstream, manifest):
    """Rewrite `checked` from the upstream tree so the sha list is generated."""
    patterns = load_patterns(upstream)
    manifest["checked"] = {
        rel: sha256(Path(upstream) / rel) for rel in sorted(surface(upstream, patterns))
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[scaffold_gate] pinned {len(manifest['checked'])} file(s) from the upstream tree")


def selftest():
    """One fixture per rule, each built so only that rule can fail."""
    cases = []

    def build(local_mutator):
        work = Path(tempfile.mkdtemp())
        up = work / "up"
        lo = work / "lo"
        (up / ".github/misc").mkdir(parents=True)
        # A glob pattern, so that adding a file below really lands on the
        # template-owned surface (a literal pattern would not match a new name).
        (up / ".github/misc/scaffold-paths.txt").write_text("*.txt\n", encoding="utf-8")
        for name, body in (("a.txt", "alpha\n"), ("b.txt", "beta\n")):
            (up / name).write_text(body, encoding="utf-8")
        shutil.copytree(up, lo)
        local_mutator(up, lo)
        manifest = {
            "scaffold": "test",
            # Generated, never hand-typed: the fixture surface includes the
            # patterns file itself, exactly like the real one does.
            "checked": {rel: sha256(up / rel) for rel in sorted(surface(up, load_patterns(up)))},
            "tolerated": [],
        }
        return up, lo, manifest

    def clean(up, lo):
        return None

    cases.append(("clean tree passes", clean, None))
    cases.append((
        "G1 a modified checked file is red",
        lambda up, lo: (lo / "a.txt").write_text("ALPHA\n", encoding="utf-8"),
        "G1 a.txt: differs from upstream and is not tolerated",
    ))
    cases.append((
        "G2 a stale tolerance is red",
        lambda up, lo: None,
        "G2 b.txt: tolerated but identical to upstream (stale entry)",
    ))
    cases.append((
        "G3 an undeclared surface file is red",
        lambda up, lo: (lo / "c.txt").write_text("gamma\n", encoding="utf-8"),
        "G3 c.txt: on the template surface but not in checked",
    ))
    cases.append((
        "G4 a tolerance budget breach is red",
        lambda up, lo: (lo / "b.txt").write_text("BETA\n", encoding="utf-8"),
        "G4 tolerated grew to 2 > budget 1",
    ))
    # G5's fixtures pin a file that is deliberately *not* on the template surface, which is the
    # situation LICENSE and NOTICE are in: nothing else in this gate would look at them. The name
    # has no `.txt` suffix on purpose -- the fixture surface is `*.txt`, and a pinned file that also
    # landed there would fail G3 as well, which would make the fixture prove the wrong rule.
    cases.append((
        "G5 a changed template text is red",
        lambda up, lo: (lo / "NOTICE").write_text("edited\n", encoding="utf-8"),
        "G5 NOTICE: the bytes changed since the pin",
    ))
    cases.append((
        "G5 an unchanged template text is green",
        lambda up, lo: (lo / "NOTICE").write_text("notice\n", encoding="utf-8"),
        None,
    ))

    failures = 0
    for index, (name, mutate, expected) in enumerate(cases):
        up, lo, manifest = build(mutate)
        if index == 2:  # the stale-tolerance fixture needs b.txt declared
            manifest["tolerated"] = [{"path": "b.txt", "why": "fixture", "exit": "fixture"}]
        if index == 4:  # budget fixture: two declared entries, budget of one
            manifest["tolerated"] = [
                {"path": "a.txt", "why": "fixture", "exit": "fixture"},
                {"path": "b.txt", "why": "fixture", "exit": "fixture"},
            ]
            manifest["tolerance_budget"] = 1
        if index in (5, 6):  # the two G5 fixtures: pin the text the mutator writes (or does not)
            manifest["template_text"] = {
                "NOTICE": {
                    "sha256": hashlib.sha256(b"notice\n").hexdigest(),
                    "why": "fixture",
                }
            }
        problems = audit(lo, up, manifest)
        joined = "\n".join(problems)
        if expected is None:
            ok = not problems
        else:
            # The assertion quotes this rule's own wording, so a neighbouring
            # fixture cannot satisfy it by accident.
            ok = expected in joined
        print(f"[selftest] {'OK  ' if ok else 'FAIL'} {name}")
        if not ok:
            failures += 1
            print(f"[selftest]   expected: {expected!r}\n[selftest]   got: {problems}")
        shutil.rmtree(lo.parent, ignore_errors=True)
    if failures:
        print(f"[selftest] {failures} case(s) FAILED", file=sys.stderr)
        return 1
    print(f"[selftest] {len(cases)} case(s) behave as specified")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", action="store_true", help="regenerate `checked` from upstream")
    parser.add_argument("--selftest", action="store_true", help="prove every rule can fail")
    parser.add_argument("--list", action="store_true", help="show the contract summary")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    manifest = load_manifest()
    tag = str(manifest.get("scaffold", "")).strip()
    if not tag:
        raise SystemExit("scaffold_gate: scaffold.json needs a `scaffold` tag")
    if args.list:
        print(f"scaffold {tag}: {len(manifest.get('checked', {}))} checked, "
              f"{len(manifest.get('tolerated', []))} tolerated")
        for entry in manifest.get("tolerated", []):
            print(f"  {entry['path']}: {entry['why']}")
        return 0

    with tempfile.TemporaryDirectory() as workdir:
        upstream = fetch_upstream(tag, workdir)
        if args.pin:
            pin(upstream, manifest)
            manifest = load_manifest()
        problems = audit(ROOT, upstream, manifest)
        if problems:
            for line in problems:
                print(f"::error:: {line}")
            print(f"[scaffold_gate] FAIL ({len(problems)} problem(s))", file=sys.stderr)
            return 1
    print(f"[scaffold_gate] OK ({len(manifest.get('checked', {}))} file(s) match v{tag}, "
          f"{len(manifest.get('tolerated', []))} declared fork(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
