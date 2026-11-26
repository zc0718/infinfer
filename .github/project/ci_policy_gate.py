#!/usr/bin/env python3
"""Workflow policy: the rules that are cheap to break and expensive to notice.

GitHub's own scanner suspended a workflow of ours once for a shape that looks like exfiltration
(`toJSON(secrets)` handed to every job) -- it came back as `completed/action_required` with zero
jobs, and the API refused to approve it. That lesson is the first rule here.

The workflows are template-owned files, so this gate only *reads* them: recognising a shape is
possible without editing anything, and the two files we did fork are already declared in
`scaffold.json`.

Rules:

  Y1 no workflow dumps the whole secret context (`toJSON(secrets)`, or `secrets` passed as a
     whole object to something that serialises it): pass the keys you need one by one
  Y2 every `uses:` names a reference: either a local workflow (`./...`) or `owner/repo@ref`.
     A bare `owner/repo` silently follows whatever the marketplace serves
  Y3 no `uses:` reference is a moving branch (`@main`, `@master`, `@HEAD`, `@latest`, `@develop`):
     a tag or a sha pins what ran, a branch makes "it passed yesterday" a statement about
     someone else's repository
  Y4 the repository has at least one workflow: an empty policy check is a vacuous one

Usage:

    python3 .github/project/ci_policy_gate.py             # check
    python3 .github/project/ci_policy_gate.py --selftest  # prove each rule can fail
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MOVEING_REFS = ("main", "master", "head", "latest", "develop", "trunk")
USES = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)", re.M)
SECRET_DUMP = re.compile(r"toJSON\s*\(\s*secrets\s*\)", re.I)


def workflows(root):
    return sorted((pathlib.Path(root) / ".github/workflows").glob("*.y*ml"))


def check(root):
    problems = []
    reported = []
    files = workflows(root)

    # Y4 -- nothing to check is not the same as nothing wrong.
    if not files:
        problems.append("ci-policy: no workflow found under .github/workflows, so Y1-Y3 pass vacuously")
        return problems, reported

    for path in files:
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")

        # Y1 -- the shape that got a workflow suspended.
        if SECRET_DUMP.search(text):
            problems.append(
                f"ci-policy: {rel} dumps the whole secret context (`toJSON(secrets)`); pass the keys "
                f"each job needs -- this exact shape reads as exfiltration to GitHub's scanner"
            )

        for ref in USES.findall(text):
            # Y2 -- a reference, not a bare repository name.
            if not (ref.startswith("./") or "@" in ref):
                problems.append(
                    f"ci-policy: {rel} uses '{ref}' without a ref; a bare owner/repo follows whatever "
                    f"the marketplace serves that day"
                )
                continue
            if ref.startswith("./"):
                continue
            # Y3 -- and not a moving branch.
            tag = ref.rsplit("@", 1)[1]
            if tag.lower() in MOVEING_REFS:
                problems.append(
                    f"ci-policy: {rel} pins '{ref}' to the branch '{tag}'; a tag or a sha is what "
                    f"makes 'it passed' a statement about this run"
                )

    total_refs = sum(len(USES.findall(p.read_text(encoding="utf-8", errors="replace"))) for p in files)
    reported.append(f"{len(files)} workflow(s), {total_refs} action reference(s) checked")
    return problems, reported


def selftest():
    import tempfile

    failures = 0
    with tempfile.TemporaryDirectory(prefix="ci-policy-selftest-") as workdir:
        root = pathlib.Path(workdir)
        (root / ".github/workflows").mkdir(parents=True)

        def run(text, name="w.yml"):
            for stale in (root / ".github/workflows").glob("*.yml"):
                stale.unlink()
            (root / ".github/workflows" / name).write_text(text, encoding="utf-8")
            return check(root)[0]

        clean = (
            "jobs:\n"
            "  a:\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: ./.github/workflows/other.yml\n"
            "      - run: echo ${{ secrets.TOKEN }}\n"
        )
        problems = run(clean)
        if not problems:
            print("[ci-policy:selftest] PASS Y1..Y3 a pinned workflow is green")
        else:
            failures += 1
            print(f"[ci-policy:selftest] FAIL expected a clean run, got {problems}")

        problems = run("env:\n  ALL: ${{ toJSON(secrets) }}\n" + clean)
        if any("dumps the whole secret context" in p for p in problems):
            print("[ci-policy:selftest] PASS Y1 a secret dump is red")
        else:
            failures += 1
            print(f"[ci-policy:selftest] FAIL Y1 expected the secret-dump report, got {problems}")

        problems = run("jobs:\n  a:\n    steps:\n      - uses: actions/checkout\n")
        if any("without a ref" in p for p in problems):
            print("[ci-policy:selftest] PASS Y2 a bare repository reference is red")
        else:
            failures += 1
            print(f"[ci-policy:selftest] FAIL Y2 expected the missing-ref report, got {problems}")

        problems = run("jobs:\n  a:\n    steps:\n      - uses: actions/checkout@main\n")
        if any("pins 'actions/checkout@main' to the branch" in p for p in problems):
            print("[ci-policy:selftest] PASS Y3 a moving branch is red")
        else:
            failures += 1
            print(f"[ci-policy:selftest] FAIL Y3 expected the moving-branch report, got {problems}")

        empty = pathlib.Path(tempfile.mkdtemp())
        (empty / ".github/workflows").mkdir(parents=True)
        problems = check(empty)[0]
        if any("pass vacuously" in p for p in problems):
            print("[ci-policy:selftest] PASS Y4 an empty policy check is red")
        else:
            failures += 1
            print(f"[ci-policy:selftest] FAIL Y4 expected the empty-tree report, got {problems}")

    if failures:
        print(f"[ci-policy:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[ci-policy:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="prove each rule can fail")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    problems, reported = check(ROOT)
    if problems:
        for problem in problems:
            print(f"[ci-policy] {problem}", file=sys.stderr)
        print(f"[ci-policy] FAIL ({len(problems)} problem(s))", file=sys.stderr)
        return 1
    print(f"[ci-policy] OK ({len(reported)} check(s): workflow policy holds)")
    for line in reported:
        print(f"[ci-policy]   {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
