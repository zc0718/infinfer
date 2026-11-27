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
  Y5 no heavy check declared as not-in-CI (`toolchains.json:heavy_checks`) is run by `ci.sh` or
     by any workflow. The project-gates job has no conan cache, no house cross toolchain and no
     qemu, so running one there is red for reasons that have nothing to do with the tree -- and
     the usual answer to that redness is a skip, which is how a check stops meaning anything.
     T5 owns *declaring* the heavy checks; Y5 owns keeping CI out of them.

Usage:

    python3 .github/project/ci_policy_gate.py             # check
    python3 .github/project/ci_policy_gate.py --selftest  # prove each rule can fail
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MOVEING_REFS = ("main", "master", "head", "latest", "develop", "trunk")
USES = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)", re.M)
SECRET_DUMP = re.compile(r"toJSON\s*\(\s*secrets\s*\)", re.I)
HEAVY_MANIFEST = ".github/project/toolchains.json"
# The scripts a heavy-check command names. A token has to look like a path to a runnable file:
# `conan create …` has none and contributes nothing, which is correct -- conan is the tool, not a
# script we could accidentally wire into CI.
SCRIPT_TOKEN = re.compile(r"[\w./-]+\.(?:py|sh|rb|pl)")


def workflows(root):
    return sorted((pathlib.Path(root) / ".github/workflows").glob("*.y*ml"))


def heavy_scripts(root):
    """`[(check id, script path)]` for the checks declared as not-in-CI.

    `None` means the manifest is absent (nothing to guard in this tree); a **string** means it is
    there and unreadable, which is reported rather than passed -- a policy rule that cannot run is
    not a policy rule that holds.
    """
    path = pathlib.Path(root) / HEAVY_MANIFEST
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return f"{HEAVY_MANIFEST} cannot be read ({error}), so Y5 cannot run"
    scripts = []
    for entry in manifest.get("heavy_checks") or []:
        for token in SCRIPT_TOKEN.findall(str(entry.get("command", ""))):
            scripts.append((str(entry.get("id", "?")), token))
    return scripts


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

    # Y5 -- CI must not run what CI has been told it cannot run.
    heavy = heavy_scripts(root)
    if heavy is None:
        reported.append("Y5 skipped: no toolchains.json in this tree")
    elif isinstance(heavy, str):
        problems.append(f"ci-policy: {heavy}")
    elif not heavy:
        problems.append(
            "ci-policy: toolchains.json declares no heavy_checks, so Y5 has nothing to guard -- "
            "an empty policy check is a vacuous one"
        )
    else:
        runners = []
        ci = pathlib.Path(root) / ".github/project/ci.sh"
        if ci.is_file():
            runners.append((".github/project/ci.sh", ci.read_text(encoding="utf-8", errors="replace")))
        runners += [(path.relative_to(root).as_posix(),
                     path.read_text(encoding="utf-8", errors="replace")) for path in files]
        for check_id, script in heavy:
            for where, text in runners:
                if script in text:
                    problems.append(
                        f"ci-policy: {where} runs '{script}', which toolchains.json declares as the "
                        f"heavy check '{check_id}' that CI cannot run; say where it *does* run in "
                        f"`runs_from` (T5 keeps that declaration honest) and keep CI out of it"
                    )
        reported.append(f"{len({script for _, script in heavy})} heavy-check script(s) kept out of CI")
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

        # Y5 -- two fixtures, because the rule has to be able to *go green* as well: one where
        # ci.sh runs a declared heavy check, one where it does not.
        heavy_root = pathlib.Path(workdir) / "heavy"
        (heavy_root / ".github/project").mkdir(parents=True)
        (heavy_root / ".github/workflows").mkdir(parents=True)
        (heavy_root / ".github/workflows/w.yml").write_text(clean, encoding="utf-8")
        (heavy_root / ".github/project/toolchains.json").write_text(
            json.dumps({"heavy_checks": [{"id": "kat-chains",
                                           "command": "python3 qa/kat/run_kat.py --chain all"}]}),
            encoding="utf-8",
        )
        (heavy_root / ".github/project/ci.sh").write_text(
            "python3 qa/kat/run_kat.py --chain all\n", encoding="utf-8")
        problems = check(heavy_root)[0]
        if any("declares as the heavy check 'kat-chains'" in p for p in problems):
            print("[ci-policy:selftest] PASS Y5 a heavy check that CI runs is red")
        else:
            failures += 1
            print(f"[ci-policy:selftest] FAIL Y5 expected the heavy-check report, got {problems}")

        (heavy_root / ".github/project/ci.sh").write_text(
            "echo no heavy check here\n", encoding="utf-8")
        problems = check(heavy_root)[0]
        if not problems:
            print("[ci-policy:selftest] PASS Y5 a heavy check kept out of CI is green")
        else:
            failures += 1
            print(f"[ci-policy:selftest] FAIL Y5 expected a clean run, got {problems}")

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
