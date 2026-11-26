#!/usr/bin/env python3
"""Project-owned toolchain rules: one foundation, or a named debt.

G-46 exists because the same tree was being built by two different compilers:
the KAT matrix resolved everything from the Conan cache (arm-toolchain /
rv-toolchain), while the template's cross-compile job apt-installed
`gcc-aarch64-linux-gnu` and `gcc-arm-none-eabi`. Nothing was red, and the two
sets of evidence were not comparable - the failure mode is silent, which is why
it is a gate and not a paragraph.

  T1 nothing on the surface we own names a third-party toolchain. Our own
     scripts must resolve compilers from the declared house packages, by version
     probe, the way `workspace/qa/kat/run_kat.py` does.
  T2 every debt entry carries `path`, `contains`, `why` and `exit`. Debt that is
     not written down with a removal condition becomes folklore, and the next
     person "fixes" it by adding another compiler.
  T3 every debt entry is still accurate: the literal it names is still present.
     A declaration that outlives the code it describes is worse than none - it
     reads as current while pointing at nothing.
  T4 every board profile names a declared house package and a triplet that
     package actually provides. A board is itself a claim (the m33 cell cost a
     day to a profile that was wrong), so it has to agree with the toolchain
     table instead of inventing a compiler.
  T5 the heavy checks that CI cannot run are still declared *and located*: a
     command, the profiles it needs, where it runs, and why not in CI. A heavy
     check that only exists in someone's shell history is the same disease as a
     gate nobody runs.

`--selftest` proves each rule fails on its own fixture.

项目自己的工具链规则：只允许一套地基，或者把债务写清楚。

G-46 的由来是同一棵树被两套编译器构建：KAT 矩阵全部从 Conan 缓存解析
（arm-toolchain / rv-toolchain），而模板的交叉编译 job 用 apt 装
`gcc-aarch64-linux-gnu` 与 `gcc-arm-none-eabi`。当时没有一条门禁是红的，而两套证据
根本不可比——这种失败是静默的，所以它必须是一条门禁，而不是一段话。

  T1 我们自己拥有的面上不得出现第三方工具链的名字；自有脚本必须像
     `workspace/qa/kat/run_kat.py` 那样，按声明的版本探针从 house 包解析。
  T2 每条债务都要写 `path` / `contains` / `why` / `exit`；没有退出条件的债务会变成
     口口相传，下一个人只会再加一个编译器来"修复"它。
  T3 每条债务仍然成立：它点名的字面量还在文件里。比代码活得久的声明比没有更糟——
     它看起来是最新的，却指着空处。

`--selftest` 用各自的 fixture 证明每条规则都会红。
"""
import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(".github/project/toolchains.json")
# The manifest holds the forbidden names as data, and this file holds them as
# fixture text; scanning either would make the gate report itself.
SELF_NAMES = {"toolchains.json", "toolchain_gate.py"}


def load_manifest(root):
    with (Path(root) / MANIFEST).open(encoding="utf-8") as handle:
        return json.load(handle)


def iter_surface_files(root, manifest):
    """Every text file on the surface we own, minus generated build output."""
    root = Path(root)
    skip = [root / entry["path"] for entry in manifest.get("skip_dirs", [])
            if isinstance(entry, dict) and entry.get("path")]
    suffixes = tuple(manifest.get("text_suffixes", []))
    for surface in manifest.get("owned_surface", []):
        base = root / surface
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.name in SELF_NAMES:
                continue
            if any(str(path).startswith(str(directory)) for directory in skip):
                continue
            if suffixes and path.suffix not in suffixes:
                continue
            yield path


def rule_owned_surface(root, manifest):
    """T1: our own files resolve compilers from the house packages only."""
    problems = []
    names = manifest.get("third_party", [])
    for path in iter_surface_files(root, manifest):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:  # unreadable is a problem, not a skip
            problems.append(f"T1 {path}: cannot read ({exc})")
            continue
        for name in names:
            if name in text:
                problems.append(
                    f"T1 {path}: names the third-party compiler '{name}'; "
                    f"resolve it from the house packages instead")
    return problems


def rule_declarations(root, manifest):
    """T2: a debt entry without a reason and an exit is folklore."""
    problems = []
    for index, entry in enumerate(manifest.get("debt", [])):
        for field in ("path", "contains", "why", "exit"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                problems.append(
                    f"T2 debt[{index}] ({entry.get('path', '?')}): "
                    f"missing or empty '{field}'")
    # An exemption from the scan needs a reason too: an unexplained skip is how a
    # rule quietly stops covering the one directory that mattered.
    # 扫描豁免也需要理由：来路不明的 skip 会让规则悄悄漏掉那个真正重要的目录。
    for index, entry in enumerate(manifest.get("skip_dirs", [])):
        if not isinstance(entry, dict):
            problems.append(f"T2 skip[{index}]: expected an object with path/why")
            continue
        for field in ("path", "why"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                problems.append(
                    f"T2 skip[{index}] ({entry.get('path', '?')}): "
                    f"missing or empty '{field}'")
    return problems


def rule_debt_is_current(root, manifest):
    """T3: the drift a debt entry describes must still be there."""
    problems = []
    root = Path(root)
    for index, entry in enumerate(manifest.get("debt", [])):
        relative = entry.get("path")
        needle = entry.get("contains")
        if not relative or not needle:
            continue  # T2 already reported it
        path = root / relative
        if not path.is_file():
            problems.append(
                f"T3 debt[{index}]: '{relative}' no longer exists; "
                f"drop the entry or fix the path")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if needle not in text:
            problems.append(
                f"T3 debt[{index}]: '{relative}' no longer contains "
                f"'{needle}'; the declaration is stale, re-check the drift")
    return problems


def rule_boards_agree(root, manifest):
    """T4: every board names a declared house package and one of its targets."""
    problems = []
    root = Path(root)
    boards_dir = root / manifest.get("boards_dir", "")
    if not boards_dir or not boards_dir.is_dir():
        return problems  # a tree without board profiles has nothing to agree on
    house = manifest.get("house", {})
    for path in sorted(boards_dir.glob("*.json")):
        try:
            board = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"T4 {path.name}: cannot read ({exc})")
            continue
        toolchain = board.get("toolchain") or {}
        package = toolchain.get("package")
        triplet = toolchain.get("triplet")
        if package not in house:
            problems.append(
                f"T4 {path.name}: toolchain.package '{package}' is not a declared "
                f"house package")
            continue
        targets = house[package].get("targets", [])
        if triplet not in targets:
            problems.append(
                f"T4 {path.name}: triplet '{triplet}' is not one of "
                f"{package}'s targets {targets}")
    return problems


def rule_heavy_checks(root, manifest):
    """T5: what CI cannot run must still say where it runs and why not."""
    problems = []
    for index, entry in enumerate(manifest.get("heavy_checks", [])):
        if not isinstance(entry, dict):
            problems.append(f"T5 heavy[{index}]: expected an object")
            continue
        for field in ("id", "command", "runs_from", "why_not_in_ci"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                problems.append(
                    f"T5 heavy[{index}] ({entry.get('id', '?')}): "
                    f"missing or empty '{field}'")
        profiles = entry.get("profiles")
        if not isinstance(profiles, list) or not profiles:
            problems.append(
                f"T5 heavy[{index}] ({entry.get('id', '?')}): "
                f"missing or empty 'profiles'")
    return problems


def rule_build_shape(root):
    """T6 (PLAN `gate/build`): the build shape the metadata declares is the one CMake sets.

    `metadata.build_cstd` / `build_cppstd` / `cmake_version` are read by the recipe and by CI, while
    `CMakeLists.txt` states the same three things to CMake. Nothing compared them, and the failure is
    quiet in both directions: a tree that builds as C11 while the metadata says otherwise breaks only
    on the day someone trusts the metadata.
    """
    problems = []
    meta_path = Path(root) / "metadata.json"
    cmake_path = Path(root) / "CMakeLists.txt"
    if not meta_path.is_file() or not cmake_path.is_file():
        return problems
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cmake = cmake_path.read_text(encoding="utf-8")

    # The template does not hard-code the standard: `CMakeLists.txt` reads `build_cstd` out of
    # metadata.json itself (`string(JSON BUILD_CSTD ...)`) and passes `${BUILD_CSTD}` to the
    # targets. So the rule is not "two literals must match" -- that version of it compared a
    # quantity with no second copy and could never fail. What can fail is the wiring: a build that
    # hard-codes a different number, or one that stops reading the metadata field at all.
    for field, var, prop in (("build_cstd", "BUILD_CSTD", "C_STANDARD"),
                             ("build_cppstd", "BUILD_CPPSTD", "CXX_STANDARD")):
        declared = str(meta.get(field, "")).strip()
        if not declared:
            continue
        wired = f'"{field}"' in cmake and "${" + var + "}" in cmake
        literal = re.search(rf"{prop}\s+(\d+)\b", cmake)
        if not wired:
            if not literal:
                problems.append(
                    f"T6 metadata.{field} is {declared} but CMakeLists.txt neither reads it nor sets "
                    f"{prop} to a number -- the field is not connected to the build")
            elif literal.group(1) != declared:
                problems.append(
                    f"T6 metadata.{field} is {declared} but CMakeLists.txt hard-codes {prop} "
                    f"{literal.group(1)} -- one of the two is fiction")

    # `metadata.cmake_version` is the CMake the *recipe installs* (conanfile.py: build_requires
    # cmake/{_cmake}), not the minimum `CMakeLists.txt` demands. The two are different quantities
    # with names that look alike, which this rule found by being wrong first: it compared them for
    # equality and reported a real-looking defect that was not one. What does have to hold is that
    # the installed CMake is new enough for the declared build.
    installed = str(meta.get("cmake_version", "")).strip()
    match = re.search(r"cmake_minimum_required\s*\(\s*VERSION\s+([\d.]+)", cmake)
    if installed and match:

        def as_tuple(text):
            return tuple(int(part) for part in re.findall(r"\d+", text))

        if as_tuple(installed) < as_tuple(match.group(1)):
            problems.append(
                f"T6 the recipe installs cmake/{installed} but CMakeLists.txt requires "
                f"{match.group(1)} -- the build tool the recipe pins cannot configure this tree")
    return problems


def check(root):
    manifest = load_manifest(root)
    return (rule_owned_surface(root, manifest)
            + rule_declarations(root, manifest)
            + rule_debt_is_current(root, manifest)
            + rule_boards_agree(root, manifest)
            + rule_heavy_checks(root, manifest)
            + rule_build_shape(root))


def _fixture(root, manifest, surface_file, surface_text, debt_file=None,
             debt_text=None, board_text=None, extra=None):
    """Lay out a miniature tree: a manifest, one owned file, one debt file, plus `extra`."""
    root = Path(root)
    (root / ".github/project").mkdir(parents=True, exist_ok=True)
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pairs = [(surface_file, surface_text), (debt_file, debt_text)]
    boards_dir = manifest.get("boards_dir")
    if boards_dir and board_text is not None:
        pairs.append((f"{boards_dir}/fixture.json", board_text))
    for relative, text in pairs + list((extra or {}).items()):
        if relative is None or text is None:
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def selftest():
    good_board = ('{"toolchain": {"package": "arm-toolchain/11.3.rel1", '
                  '"triplet": "arm-none-eabi"}}')
    base_manifest = {
        "house": {"arm-toolchain/11.3.rel1": {"targets": ["arm-none-eabi"]}},
        # Built from fragments so this file never contains the literal it bans.
        "third_party": ["aarch64-linux-" + "gnu-gcc"],
        "owned_surface": ["workspace"],
        "skip_dirs": [],
        "text_suffixes": [".py"],
        "boards_dir": "workspace/boards",
        "heavy_checks": [{
            "id": "cross-build",
            "command": "conan create . -pr:h {profile}",
            "profiles": ["armv8"],
            "runs_from": "preflight",
            "why_not_in_ci": "no conan on the gates job",
        }],
        "debt": [{
            "path": ".github/workflows/ci.yml",
            "contains": "apt-" + "compiler",
            "why": "template-owned",
            "exit": "authorised scaffold edit",
        }],
    }
    cases = []

    def case(name, expected, manifest=None, surface="import os\n",
             debt="apt-compiler\n", board=good_board, extra=None):
        return (name, expected, manifest or base_manifest, surface, debt, board, extra)

    cases.append(case("clean tree is green", None))
    cases.append(case(
        "an owned file naming a third-party compiler is red", "T1 ",
        surface="CC = 'aarch64-linux-" + "gnu-gcc'\n"))
    cases.append(case(
        "a debt entry without an exit is red", "T2 ",
        manifest={**base_manifest, "debt": [{
            "path": ".github/workflows/ci.yml",
            "contains": "apt-" + "compiler",
            "why": "template-owned",
            "exit": "",
        }]}))
    cases.append(case(
        "a stale debt entry is red", "T3 ", debt="nothing relevant here\n"))
    cases.append(case(
        "a scan exemption without a reason is red", "T2 ",
        manifest={**base_manifest, "skip_dirs": [{"path": "workspace/build"}]}))
    cases.append(case(
        "a board naming an undeclared toolchain is red", "T4 ",
        board='{"toolchain": {"package": "apt:gcc", "triplet": "arm-none-eabi"}}'))
    cases.append(case(
        "a board naming a triplet its package does not provide is red", "T4 ",
        board='{"toolchain": {"package": "arm-toolchain/11.3.rel1", '
              '"triplet": "riscv-none-elf"}}'))
    cases.append(case(
        "T6 a C standard the build hard-codes differently", "T6 ",
        extra={"metadata.json": '{"build_cstd": "99"}',
               "CMakeLists.txt": "set(CMAKE_C_STANDARD 11)\n"}))
    cases.append(case(
        "T6 a metadata standard the build never reads", "T6 ",
        extra={"metadata.json": '{"build_cstd": "11"}',
               "CMakeLists.txt": "set(CMAKE_POSITION_INDEPENDENT_CODE ON)\n"}))
    cases.append(case(
        "a heavy check without a location is red", "T5 ",
        manifest={**base_manifest, "heavy_checks": [{
            "id": "cross-build", "command": "conan create .",
            "profiles": ["armv8"], "runs_from": "",
            "why_not_in_ci": "no conan in CI",
        }]}))

    failures = 0
    for name, expected, manifest, surface, debt, board, extra in cases:
        temp = tempfile.mkdtemp(prefix="toolchain-selftest-")
        try:
            _fixture(temp, manifest, "workspace/app/tool.py", surface,
                     ".github/workflows/ci.yml", debt, board, extra)
            problems = check(temp)
            if expected is None:
                if problems:
                    print(f"[toolchain:selftest] FAIL: {name} -> {problems}",
                          file=sys.stderr)
                    failures += 1
            elif not any(problem.startswith(expected) for problem in problems):
                print(f"[toolchain:selftest] FAIL: {name} expected {expected!r}, "
                      f"got {problems}", file=sys.stderr)
                failures += 1
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    if failures:
        print(f"[toolchain:selftest] FAIL ({failures} case(s))", file=sys.stderr)
        return 1
    print(f"[toolchain:selftest] {len(cases)} case(s) behave as specified")
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
        print(f"[toolchain_gate] FAIL ({len(problems)} problem(s))")
        for problem in problems:
            print(f"::error:: {problem}")
        return 1
    manifest = load_manifest(ROOT)
    print(f"[toolchain_gate] OK ({len(manifest.get('house', {}))} house "
          f"toolchain(s), {len(manifest.get('debt', []))} declared debt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
