#!/usr/bin/env python3
"""Build the k0_actquant KAT and run it on every target this box can host.

Order matters and is the whole point of the P1.3 design:

  1. host emit   - run the chain on the host, write the expectation blob. This is
                   the *only* producer of expectations; nothing is hand-typed.
  2. emit image  - turn that blob into a C header of raw bytes.
  3. host check  - rebuild the same driver against the baked header, so the host
                   is the first target to prove the comparison path works.
  4. targets     - cross-compile the identical driver per board and run it via
                   run_kat.py (exit code is the verdict).

Cross-target equality is bitwise by construction: every build compares the same
memcmp'd bytes against the same baked arrays.
"""
import argparse
import glob
import json
import hashlib
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
# Same root-by-marker rule as run_kat.py: the tree moved on 2026-11-27 and a fixed depth
# would have kept pointing one level short.
ROOT = next(p for p in (HERE, *HERE.parents) if (p / "metadata.json").is_file())
BOARDS = HERE.parent / "qemu" / "boards"
# The emitters are siblings of this chain directory; naming them through `HERE` keeps the two
# halves of the pipeline from being able to disagree about where they live.
PREP = HERE.parent / "prep"
# Same cache rule as run_kat.py: build output stays out of the tree, so `.gitignore` needs no
# entry for it. Kept identical on purpose -- two scripts with two different output roots would be
# a divergence nobody would notice until a stale expectation turned up.
CACHE = (pathlib.Path(os.environ.get("INFRINFER_QA_CACHE",
                                    pathlib.Path.home() / ".cache" / "infinfer" / "qa"))
         / hashlib.sha256(str(ROOT).encode("utf-8")).hexdigest()[:8])
BUILD = CACHE / "build"
GENERATED = CACHE / "generated"

KERNEL_SOURCES = [
    ROOT / "src/kernel/contract.c",
    ROOT / "src/kernel/scalar/activation.c",
    ROOT / "src/kernel/scalar/quant.c",
    ROOT / "src/kernel/scalar/register.c",
    ROOT / "src/kernel/scalar/relu.c",
]
KAT_SOURCES = [HERE / "kat_chain.c", HERE / "kat_driver.c"]
INCLUDES = ["-I", str(ROOT / "include"), "-I", str(HERE)]


def run(argv, **kwargs):
    """Run a command, echoing it first so a failure is reproducible by hand."""
    print("[run_k0] $ " + " ".join(str(a) for a in argv), flush=True)
    return subprocess.run([str(a) for a in argv], check=True, **kwargs)


def find_tool(env_name, pattern, binary):
    override = os.environ.get(env_name)
    if override:
        candidate = pathlib.Path(override) / binary
        if candidate.is_file():
            return candidate
        raise SystemExit(f"run_k0: {env_name}={override} has no {binary}")
    hits = glob.glob(str(pathlib.Path.home() / ".conan2/p/b" / pattern / "p/bin" / binary))
    if not hits:
        raise SystemExit(
            f"run_k0: no {binary} in the Conan cache (set {env_name} to its bin dir)")
    return pathlib.Path(max(hits, key=lambda p: pathlib.Path(p).stat().st_mtime))


def load_board(name):
    return json.loads((BOARDS / f"{name}.json").read_text(encoding="utf-8"))


def build_host_emit(cc):
    out = BUILD / "k0_emit"
    run([cc, "-O2", "-std=c11", "-DKAT_EMIT", *INCLUDES, *KAT_SOURCES, *KERNEL_SOURCES,
         "-lm", "-o", out])
    blob = BUILD / "k0_expected.bin"
    run([out, blob])
    return blob


def emit_header(blob):
    header = GENERATED / "kat_k0_image.h"
    run([sys.executable, PREP / "emit_kat_image.py", blob, header])
    return header


def build_host_check(cc):
    out = BUILD / "k0_host_check"
    run([cc, "-O2", "-std=c11", *INCLUDES, *KAT_SOURCES, *KERNEL_SOURCES, "-lm", "-o", out])
    done = subprocess.run([str(out)], capture_output=True, text=True)
    print(f"[run_k0] host check -> exit {done.returncode}", flush=True)
    if done.returncode != 0:
        if done.stderr:
            print(done.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"run_k0: host KAT failed with {done.returncode}")
    return done.returncode


def arm_ld_for(board, out_ld):
    """The M-profile image must sit where the model's reset vector points.

    AN505 (M33) resets into a 32K SRAM window at 0x10000000 and keeps its 4M RAM
    at 0x00400000; every other board is happy with vectors at 0x0 and RAM at
    0x20000000 (all measured). Rather than a second linker script, rewrite the
    two MEMORY lines from the board profile.
    """
    text = (HERE.parent / "qemu/smoke/arm/smoke.ld").read_text(encoding="utf-8")
    ld = board.get("ld", {})
    flash = ld.get("flash")
    ram = ld.get("ram")
    if flash:
        text = text.replace("FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 4M",
                            f"FLASH (rx)  : ORIGIN = {flash['origin']}, LENGTH = {flash['length']}")
    if ram:
        text = text.replace("RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 4M",
                            f"RAM   (rwx) : ORIGIN = {ram['origin']}, LENGTH = {ram['length']}")
    out_ld.write_text(text, encoding="utf-8")
    return out_ld


def build_target(board_name, gcc_arm, gcc_rv):
    board = load_board(board_name)
    tag = board_name.replace("-", "_")
    if board["qemu"].endswith("arm"):
        ld = arm_ld_for(board, BUILD / f"{tag}.ld")
        elf = BUILD / f"k0_{tag}.elf"
        run([gcc_arm, f"-mcpu={board['cpu']}", "-mthumb", "-O2", "-std=c11", "-nostartfiles",
             "--specs=nosys.specs", *INCLUDES, "-T", ld, *KAT_SOURCES, *KERNEL_SOURCES,
             "-lm", "-o", elf])
        image = BUILD / f"k0_{tag}.bin"
        run([gcc_arm.parent / "arm-none-eabi-objcopy", "-O", "binary", elf, image])
        return image
    elf = BUILD / f"k0_{tag}.elf"
    run([gcc_rv, f"-march={board['march']}", f"-mabi={board['mabi']}", "-mcmodel=medany",
         "-O2", "-std=c11", "-nostartfiles", *INCLUDES, "-T", HERE / "kat_rv.ld",
         "-Wl,-e,_start", *KAT_SOURCES, *KERNEL_SOURCES, "-lm", "-o", elf])
    return elf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cc", default=os.environ.get("CC", "cc"))
    ap.add_argument("--boards", nargs="*", default=None,
                    help="board stems to run (default: every board profile)")
    ap.add_argument("--skip-targets", action="store_true", help="host pipeline only")
    args = ap.parse_args()

    BUILD.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)

    print("[run_k0] 1/4 host emit", flush=True)
    blob = build_host_emit(args.cc)
    print("[run_k0] 2/4 emit image header", flush=True)
    emit_header(blob)
    print("[run_k0] 3/4 host check", flush=True)
    build_host_check(args.cc)
    if args.skip_targets:
        return 0

    print("[run_k0] 4/4 targets", flush=True)
    gcc_arm = find_tool("ARM_EABI_BIN", "arm-*", "arm-none-eabi-gcc")
    gcc_rv = find_tool("RV_ELF_BIN", "rv-*", "riscv-none-elf-gcc")
    boards = args.boards or sorted(p.stem for p in BOARDS.glob("*.json"))
    failures = []
    for board_name in boards:
        try:
            image = build_target(board_name, gcc_arm, gcc_rv)
        except subprocess.CalledProcessError as exc:
            print(f"[run_k0] BUILD FAILED {board_name} (rc={exc.returncode})", file=sys.stderr)
            failures.append(board_name)
            continue
        done = subprocess.run([sys.executable, str(HERE.parent / "qemu/run_kat.py"),
                               board_name, str(image)])
        if done.returncode != 0:
            failures.append(board_name)

    if failures:
        print(f"[run_k0] FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"[run_k0] all {len(boards)} targets pass (host + {len(boards)} QEMU)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
