#!/usr/bin/env python3
"""Project-owned scope rules: the two things a tree-wide grep gets wrong.

Both rules below started life as a paragraph in a local note, and both were
re-decided by a later pass over the tree because the note had no teeth. They are
data in `scope.json` and checked here, so the reasoning survives a machine
change and a red result names the rule:

  S1 an identity-sweep exception still holds: the literal that an external
     system hardcodes is present the declared number of times, and the
     project-prefixed lookalike has not crept in. Renaming such a literal is the
     intuitive "clean up the old name" move, and it silently breaks the external
     system instead of this build.
  S2 an out-of-scope switch still holds its declared value, so a route this plan
     does not own cannot be switched on by accident.
  S3 every entry carries `why` and `exit`: an exception without a reason is how
     a contract becomes folklore.

`--selftest` proves each rule fails on its own fixture.

项目自己的范围规则：这两条都曾经只是本地笔记里的一段话，都因为“笔记没有牙”而被
后来的一次全树扫描重新决定。现在它们是 `scope.json` 里的数据并由门禁检查，理由能
跨机器存活，红灯也会指名是哪条规则：

  S1 身份清扫的例外仍然成立：外部系统写死的字面量出现次数符合声明，且项目前缀的
     近似物没有混进来。改这种字面量是“清理旧名字”的直觉动作，它不会弄坏构建，
     只会悄悄弄坏外部系统。
  S2 本计划不拥有的开关保持声明值，属于别人的路由不会被顺手打开。
  S3 每条声明都写 `why` 与 `exit`：没有理由的例外会变成口口相传的规矩。
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(".github/project/scope.json")


def load_manifest(root):
    path = Path(root) / MANIFEST
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _dotted(data, key):
    node = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(key)
        node = node[part]
    return node


def rule_identity(root, manifest):
    """S1: the external-system literals are untouched."""
    problems = []
    for entry in manifest.get("identity_sweep_exceptions", []):
        relative = entry["path"]
        target = Path(root) / relative
        if not target.is_file():
            problems.append(f"S1 {relative}: declared path is missing")
            continue
        text = target.read_text(encoding="utf-8")
        found = text.count(entry["must_contain"])
        if found != entry["count"]:
            problems.append(
                f"S1 {relative}: `{entry['must_contain']}` appears {found} time(s), "
                f"the contract says {entry['count']} (an identity sweep must not rename it)")
        lookalike = entry.get("must_not_contain")
        if lookalike and lookalike in text:
            problems.append(
                f"S1 {relative}: `{lookalike}` reappeared; the literal belongs to the "
                "external system, not to this project's prefix")
    return problems


def rule_out_of_scope(root, manifest):
    """S2: switches this plan does not own keep their value."""
    problems = []
    for entry in manifest.get("out_of_scope", []):
        target = Path(root) / entry["path"]
        if not target.is_file():
            problems.append(f"S2 {entry['path']}: declared path is missing")
            continue
        with target.open(encoding="utf-8") as handle:
            data = json.load(handle)
        try:
            actual = _dotted(data, entry["key"])
        except KeyError:
            problems.append(f"S2 {entry['path']}: `{entry['key']}` is not declared")
            continue
        if actual != entry["must_be"]:
            problems.append(
                f"S2 {entry['path']}: `{entry['key']}` is {json.dumps(actual)}, "
                f"this plan requires {json.dumps(entry['must_be'])}")
    return problems


def rule_declarations(manifest):
    """S3: an exception without a reason is folklore."""
    problems = []
    for bucket in ("identity_sweep_exceptions", "out_of_scope"):
        for index, entry in enumerate(manifest.get(bucket, [])):
            for key in ("why", "exit"):
                if not str(entry.get(key, "")).strip():
                    label = entry.get("path") or f"{bucket}[{index}]"
                    problems.append(f"S3 {label}: `{key}` must say something")
    return problems


def check(root):
    manifest = load_manifest(root)
    return (rule_identity(root, manifest)
            + rule_out_of_scope(root, manifest)
            + rule_declarations(manifest))


def selftest():
    """Prove every rule fails on a fixture that breaks only that rule."""
    failures = 0
    green = {
        "_comment": ["fixture"],
        "identity_sweep_exceptions": [{
            "path": "ops.yml",
            "must_contain": "/tmp/fcpp_benchmark_",
            "count": 2,
            "must_not_contain": "infinfer_benchmark",
            "why": "fixture",
            "exit": "fixture",
        }],
        "out_of_scope": [{
            "path": "meta.json",
            "key": "workflow_triggers.on_board",
            "must_be": False,
            "why": "fixture",
            "exit": "fixture",
        }],
    }
    cases = [
        ("a renamed external literal is red", "S1",
         {"ops.yml": "/tmp/infinfer_benchmark_a\n/tmp/infinfer_benchmark_b\n"}, {}),
        ("a half-renamed literal is red", "S1",
         {"ops.yml": "/tmp/fcpp_benchmark_a\n/tmp/fcpp_benchmark_b\n/tmp/fcpp_benchmark_c\n"}, {}),
        ("the project prefix creeping back is red", "S1",
         {"ops.yml": "/tmp/fcpp_benchmark_a\n/tmp/fcpp_benchmark_b\n/tmp/infinfer_benchmark_x\n"}, {}),
        ("an out-of-scope switch flipped on is red", "S2", {},
         {"meta.json": {"workflow_triggers": {"on_board": True}}}),
        ("an exception with no exit is red", "S3", {}, {}, "exit"),
    ]
    with tempfile.TemporaryDirectory(prefix="scope-selftest-") as workdir:
        root = Path(workdir)
        def write(manifest, files):
            (root / ".github" / "project").mkdir(parents=True, exist_ok=True)
            (root / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
            for name, body in files.items():
                if isinstance(body, str):
                    (root / name).write_text(body, encoding="utf-8")
                else:
                    (root / name).write_text(json.dumps(body), encoding="utf-8")
        # green fixture first: a gate that cannot be green proves nothing
        write(green, {"ops.yml": "/tmp/fcpp_benchmark_a\n/tmp/fcpp_benchmark_b\n",
                      "meta.json": {"workflow_triggers": {"on_board": False}}})
        baseline = check(root)
        if baseline:
            print(f"[scope:selftest] FAIL the good fixture reported {baseline}")
            failures += 1
        else:
            print("[scope:selftest] PASS the good fixture is green")
        for case in cases:
            label, rule, files, extra = case[0], case[1], case[2], case[3]
            drop = case[4] if len(case) > 4 else None
            manifest = json.loads(json.dumps(green))
            if drop:
                manifest["identity_sweep_exceptions"][0][drop] = ""
            payload = {"ops.yml": "/tmp/fcpp_benchmark_a\n/tmp/fcpp_benchmark_b\n",
                       "meta.json": {"workflow_triggers": {"on_board": False}}}
            payload.update(files)
            payload.update({k: v for k, v in extra.items() if not isinstance(v, dict)})
            if extra and isinstance(next(iter(extra.values())), dict):
                payload.update(extra)
            write(manifest, payload)
            problems = check(root)
            if not problems or not problems[0].startswith(rule + " "):
                print(f"[scope:selftest] FAIL {label}: expected {rule}, got {problems}")
                failures += 1
            else:
                print(f"[scope:selftest] PASS {label}")
    if failures:
        print(f"[scope:selftest] FAIL ({failures} case(s))", file=sys.stderr)
        return 1
    print(f"[scope:selftest] {len(cases) + 1} case(s) behave as specified")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true",
                        help="prove each rule fails on its own fixture")
    args = parser.parse_args()
    if args.selftest:
        return selftest()

    problems = check(ROOT)
    if problems:
        print(f"[scope_gate] FAIL ({len(problems)} problem(s))")
        for problem in problems:
            print(f"::error:: {problem}")
        return 1
    manifest = load_manifest(ROOT)
    print(f"[scope_gate] OK ({len(manifest.get('identity_sweep_exceptions', []))} "
          f"identity exception(s), {len(manifest.get('out_of_scope', []))} "
          f"out-of-scope switch(es))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
