#!/usr/bin/env python3
"""Build one KAT chain and run it on the host plus every QEMU board.

The order is the whole point of the P1.3 design:

  1. host emit   - run the chain on the host, write the expectation blob. This is
                   the *only* producer of expectations; nothing is hand-typed.
  2. emit image  - turn that blob into a C header of raw bytes.
  3. host check  - rebuild the same driver against the baked header, so the host
                   is the first target to prove the comparison path works.
  4. targets     - cross-compile the identical driver per board and run it via
                   run_kat.py's board profiles (exit code is the verdict).

`-ffp-contract=off` is pinned for **every** compiler, including the host: `fc`
accumulates `acc += x*w`, and on an FPU target GCC will happily fuse that into an
FMA while x86-64 SSE2 at -O2 will not. That single difference moves the last bit
of a dot product, which would make "cross-target bitwise equality" a property of
the compiler's default rather than of the kernels. Pinning contraction off makes
the comparison meaningful; whether the *library* build should pin it too is
recorded as a gap, not decided here.
"""
import argparse
import glob
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
# Root by marker, not by depth. This tree has already moved once (`workspace/qa` ->
# `.github/project/qa` on 2026-11-27) and a `parents[N]` constant goes silently wrong when it
# happens again -- `metadata.json` sits at the repository root by INV-11 and is the marker.
# 根目录按标记找，不按层数：这棵树已经搬过一次（2026-11-27 从 `workspace/qa` 到
# `.github/project/qa`），`parents[N]` 那种常数下次就会静静地错；`metadata.json` 由 INV-11
# 固定放在仓根，就是那个标记。
ROOT = next(p for p in (HERE, *HERE.parents) if (p / "metadata.json").is_file())
BOARDS = HERE.parent / "qemu" / "boards"
# The emitters are siblings of this chain directory, and they are named through `HERE` rather
# than a repo-relative string: the two halves of the pipeline move together or not at all.
PREP = HERE.parent / "prep"
# Build output does not live in the tree. It never was source, and keeping it out means the
# repository needs no ignore rule for it -- `.gitignore` is exactly the template's file, with
# nothing added. The cache is keyed by the repository path so two checkouts on one machine cannot
# mistake each other's objects for fresh ones; `INFRINFER_QA_CACHE` overrides the location.
# 构建产物不放进树里。它本来就不是源码，放外面就使仓库不需要为它写任何忽略规则——
# `.gitignore` 就是模板原文件，一个字节没加。缓存按仓库路径分键，所以同一台机器上的两个
# checkout 不会互相当成“上次编好的”；位置可用 `INFRINFER_QA_CACHE` 覆盖。
CACHE = (pathlib.Path(os.environ.get("INFRINFER_QA_CACHE",
                                    pathlib.Path.home() / ".cache" / "infinfer" / "qa"))
         / hashlib.sha256(str(ROOT).encode("utf-8")).hexdigest()[:8])
BUILD = CACHE / "build"
GENERATED = CACHE / "generated"

KERNEL_SOURCES = [
    ROOT / "src/kernel/contract.c",
    # Every scalar body, by glob rather than by name. The explicit list this replaced was a
    # trap: `scalar/register.c` names the new kernel, the link then wants its body, and the
    # only symptom is `undefined reference` at KAT build time -- which no CI gate runs. Adding
    # `mul` (2026-11-26) failed exactly that way (gaps G-94).
    #
    # 按 glob 取全部 scalar 体，而不是逐个点名。原来那份显式清单是个坑：`scalar/register.c`
    # 里写了新 kernel，链接时就缺它的体，而唯一的症状是 KAT 构建期的 `undefined reference`
    # ——那一步没有任何 CI 门禁会跑。2026-11-26 加 `mul` 时正是这样炸的（gaps G-94）。
    *sorted((ROOT / "src/kernel/scalar").glob("*.c")),
]
# The L2 operators live in C++17 (`src/runtime`). A chain that calls them has to be
# linked by the C++ driver even though every source file it *writes* is C.
CPP_SOURCES = sorted((ROOT / "src/runtime").glob("*.cpp"))
CHAINS_NEEDING_CPP = {"k2", "k2m", "k2i8"}
CHAIN_SOURCES = {
    "k0": [HERE / "kat_chain.c"],
    "k1": [HERE / "kat_chain_k1.c"],
    "k2": [HERE / "kat_chain_k2.c", HERE / "blob_graph.c"],
    # The blob itself arrives as a generated `.S` (`.incbin`), added in `sources()`: 1.4 MB
    # spelled as a C array is ~7 MB of source for every target build to parse.
    "k2m": [HERE / "kat_chain_k2m.c", HERE / "blob_graph.c"],
    # The int8 chain runs the same executor over a v3 blob; it is small enough that the
    # blob, the input and the expectation all travel as C arrays.
    #
    # int8 链让同一个执行器跑 v3 blob；它小到可以让 blob、输入与期望都以 C 数组旅行。
    "k2i8": [HERE / "kat_chain_k2i8.c", HERE / "blob_graph.c"],
    "k3": [HERE / "kat_chain_k3.c"],
}
CHAIN_FLAGS = {
    "k0": [],
    "k1": ["-DKAT_CHAIN_K1"],
    "k2": ["-DKAT_CHAIN_K2"],
    "k2m": ["-DKAT_CHAIN_K2M"],
    "k2i8": ["-DKAT_CHAIN_K2I8"],
    "k3": ["-DKAT_CHAIN_K3"],
}
CHAIN_IDS = {
    "k0": "k0_actquant",
    "k1": "k1_mlp3",
    "k2": "k2_conv_micro",
    "k2m": "k2m_mobilenet",
    "k2i8": "k2_i8_conv",
    "k3": "k3_tier_parity",
}
# The MobileNet chain runs on the two FPU MCUs only. The claim it carries is a memory row
# (flash / RAM / peak arena) on a Cortex-M with an FPU; adding the soft-float M3 and the
# RISC-V boards would multiply a 1.4 MB image by six to answer a question nobody asked.
# k2m_mobilenet 只在两个带 FPU 的 MCU 上跑。它要交的是带 FPU 的 Cortex-M 上的一行内存数字
# （flash / RAM / 峰值 arena）；把无 FPU 的 M3 和 RISC-V 板也拉进来，只是把 1.4 MB 的镜像
# 复制六份，去回答一个没人问的问题。
K2M_BOARDS = ("m4-an386", "m7-an500")
# k3 is the only chain that compiles tier sources: it compares the tier row of a cell against the
# scalar row **on the target**, which is the evidence every other chain cannot give (k0/k1/k2 are
# written on top of the scalar registry). Built per tier, because the other tiers' sources do not
# belong in the image and their register functions would not link.
TIER_CHAINS = {"k3"}
MATRIX = ROOT / "src/kernel/matrix.yml"
TIER_CELLS_HEADER = GENERATED / "kat_k3_cells.h"
CPP_FLAGS = ["-fno-exceptions", "-fno-rtti", "-fno-threadsafe-statics"]
INCLUDES = ["-I", str(ROOT / "include"), "-I", str(HERE), "-I", str(CACHE)]
FP_FLAGS = ["-ffp-contract=off"]
# A KAT image should contain what it calls and nothing else. Without this the
# link pulls whole library objects in: the k0 image was 3.4 KB of intent and
# 11.4 KB of mostly-unreached C library, and on one target model the extra code
# was what took the run down. Trimming is not an optimisation here, it is part
# of the measurement staying about the kernels.
#
# KAT 镜像里只该有它调用的东西。不加裁剪时，链接会把整份库对象拉进来：k0 的意图是
# 3.4 KB，实际镜像 11.4 KB 而大半是没被调用的 C 库，其中一部分在某个目标模型上正
# 是让运行失败的原因。这里的裁剪不是优化，而是让测量只关于内核本身。
TRIM_FLAGS = ["-ffunction-sections", "-fdata-sections"]
TRIM_LINK = ["-Wl,--gc-sections"]


def run(argv, **kwargs):
    """Run a command, echoing it first so a failure is reproducible by hand."""
    print("[run_kat] $ " + " ".join(str(a) for a in argv), flush=True)
    return subprocess.run([str(a) for a in argv], check=True, **kwargs)


def find_tool(env_name, pattern, binary, probe):
    """Resolve the house toolchain from the Conan cache, by declared version.

    "Newest file in the cache wins" is not a version policy: the cache holds
    several trains at once (11.3, 12.3, 14.3 for RISC-V), so an mtime pick swaps
    the compiler the board profile asked for without saying anything -- and a
    target built with the wrong train is not the target that was claimed. The
    profile declares `version_probe`; the compiler that reports it is the one
    used, and anything else is an error that names what was found.

    “缓存里最新的那个”不是版本策略：缓存同时放着多个 train（RISC-V 侧有 11.3 /
    12.3 / 14.3），按 mtime 选会**悄悄**换掉板型 profile 要的编译器，而用错 train
    构建出来的目标就不是被宣称的那个目标。profile 声明 `version_probe`，报出它的
    编译器才会被采用；否则报错并列出实际找到的东西。
    """
    override = os.environ.get(env_name)
    if override:
        candidate = pathlib.Path(override) / binary
        if candidate.is_file():
            return candidate
        raise SystemExit(f"run_kat: {env_name}={override} has no {binary}")
    hits = sorted(glob.glob(str(pathlib.Path.home() / ".conan2/p/b" / pattern / "p/bin" / binary)))
    if not hits:
        raise SystemExit(
            f"run_kat: no {binary} in the Conan cache (set {env_name} to its bin dir)")
    found = []
    for hit in hits:
        done = subprocess.run([hit, "--version"], capture_output=True, text=True, check=False)
        first = (done.stdout.splitlines() or [""])[0].strip()
        found.append((hit, first))
        if probe in first:
            return pathlib.Path(hit)
    detail = "\n".join(f"  {hit}: {first}" for hit, first in found)
    raise SystemExit(
        f"run_kat: no cached {binary} reports {probe!r}; refusing to guess\n{detail}")


_TOOL_CACHE = {}


def tool_for(env_name, pattern, binary, probe):
    """Resolve one compiler per (binary, probe) pair and remember it."""
    key = (binary, probe)
    if key not in _TOOL_CACHE:
        _TOOL_CACHE[key] = find_tool(env_name, pattern, binary, probe)
    return _TOOL_CACHE[key]


def load_board(name):
    return json.loads((BOARDS / f"{name}.json").read_text(encoding="utf-8"))


def sources(chain, tier=None):
    extra = []
    if chain in TIER_CHAINS:
        # Every `.c` of the tier, `register.c` included: the chain looks the rows up by name, so a
        # tier built without its registry cannot be compared at all.
        extra = sorted((ROOT / f"src/kernel/{tier}").glob("*.c"))
        if not extra:
            raise SystemExit(f"run_kat: no sources under src/kernel/{tier}/")
    elif chain == "k2m":
        # The blob as a linker-visible object rather than as 1.4 MB of C source. Generated
        # next to the header by `emit_k2m_blob_header()`, and listed here so every build of
        # this chain -- host and target -- links the same bytes.
        extra = [GENERATED / "kat_k2m_blob.S"]
    if chain in CHAINS_NEEDING_CPP:
        # The composed operators call `infr_kernel_init` on every entry, and that function
        # names all five register functions, so an image that links only the scalar bodies
        # cannot link at all. Every tier goes in, by glob: the files whose ISA guard is off
        # compile to nothing, so a target only gains its own tier's bodies. `kat_check` pins
        # the reference tier, so what the chain computes does not change.
        #
        # Only the chains that reach the composed operators need this. `k3` builds a single
        # tier on purpose and globs it itself, so adding the tiers here would define every
        # symbol twice (measured: `multiple definition of infr_arm_m_relu_i8`).
        #
        # 组合算子每次入口都会调 `infr_kernel_init`，而它点名了五个注册函数，只链接
        # scalar 体的镜像根本链不起来。所以整档按 glob 加进来：ISA 守卫关着的文件编成空，
        # 目标端只多出自己那一档的体。`kat_check` 钉住参考档，链算出来的东西不变。
        # 只有走组合算子的链需要这份清单。`k3` 故意只建一档并自己 glob，在这里再加会
        # 让每个符号定义两次（实测：`multiple definition of infr_arm_m_relu_i8`）。
        extra = [
            *extra,
            ROOT / "src/kernel/dispatch.c",
            *sorted((ROOT / "src/kernel/x86").glob("*.c")),
            *sorted((ROOT / "src/kernel/arm_neon").glob("*.c")),
            *sorted((ROOT / "src/kernel/arm_m").glob("*.c")),
            *sorted((ROOT / "src/kernel/rvv").glob("*.c")),
        ]
    return [*CHAIN_SOURCES[chain], HERE / "kat_driver.c", *KERNEL_SOURCES, *extra]


def cxx_for(cc):
    """The C++ driver that pairs with a C compiler (cc -> c++, gcc -> g++)."""
    text = str(cc)
    if text.endswith("gcc"):
        return pathlib.Path(text[:-3] + "g++")
    if text.endswith("cc"):
        return pathlib.Path(text[:-2] + "c++")
    return pathlib.Path("c++")


def objects_for(chain, cc, cxx, out_dir, extra_flags, opt="-O2", tier=None):
    """Compile every source of a chain, each with the compiler its language needs.

    Compiling C as C++ is not a shortcut: `uint32_t` is `unsigned long` on the ARM
    target, so `unsigned int *` stops converting and the chain stops building. Each
    language keeps its own compiler; only the *link* is done by the C++ driver, which
    is all a C++ runtime actually requires.
    """
    out_dir.mkdir(parents=True, exist_ok=True)  # every caller gets its own directory now
    objects = []
    for src in sources(chain, tier):
        obj = out_dir / f"{chain}_{pathlib.Path(src).parent.name}_{pathlib.Path(src).stem}.o"
        run([cc, opt, "-std=c11", *CHAIN_FLAGS[chain], *extra_flags, *FP_FLAGS, *TRIM_FLAGS,
             *INCLUDES, "-c", src, "-o", obj])
        objects.append(obj)
    cpp_extra = []
    if chain in CHAINS_NEEDING_CPP:
        cpp_extra = CPP_FLAGS
        for src in CPP_SOURCES:
            obj = out_dir / f"{chain}_{src.stem}.o"
            run([cxx, opt, "-std=c++17", *CPP_FLAGS, *extra_flags, *FP_FLAGS, *TRIM_FLAGS,
                 *INCLUDES, "-c", src, "-o", obj])
            objects.append(obj)
    return objects


def artifact_text(cc, elf, tool, objects=None):
    """Dump the part of the ELF that proves a compile flag actually took effect.

    The assertions downstream are measured, not assumed: on armv7hf `-mfpu=
    neon-vfpv4` moves Tag_FP_arch from VFPv3 to VFPv4 and adds "with Fused-MAC"
    to Tag_Advanced_SIMD_arch, while AArch64 static ELFs from this toolchain carry
    no .ARM.attributes FP tags at all - so the aarch64 side has to be checked in
    the disassembly instead.

    下游断言来自实测而非推测：armv7hf 加 `-mfpu=neon-vfpv4` 会把 Tag_FP_arch 从 VFPv3 变成 VFPv4，
    并给 Tag_Advanced_SIMD_arch 加上 "with Fused-MAC"；而本工具链产出的 AArch64 静态 ELF
    根本没有 .ARM.attributes 的浮点标签，所以 aarch64 一侧只能到反汇编里查。
    """
    if tool == "objdump-objects":
        # Disassemble only what we compiled. The alternative - grepping the linked
        # ELF - would always find SIMD somewhere inside static glibc, which makes
        # the check vacuous on one side and false on the other.
        #
        # 只反汇编我们自己编译的对象：直接 grep 链接后的 ELF 一定会在静态 glibc 里找到
        # SIMD，那样断言一边是空话、另一边是假红。
        binary = pathlib.Path(str(cc).replace("-gcc", "-objdump"))
        parts = []
        for obj in objects or []:
            parts.append(subprocess.run([str(binary), "-d", str(obj)],
                                        capture_output=True, text=True).stdout)
        return "".join(parts)
    binary = pathlib.Path(str(cc).replace("-gcc", f"-{tool}"))
    argv = [str(binary), "-A", str(elf)] if tool == "readelf" else [str(binary), "-d", str(elf)]
    return subprocess.run(argv, capture_output=True, text=True).stdout


def check_artifact(cc, elf, asserts, objects=None):
    """Refuse a variant whose flags did not reach the artifact.

    A variant exists to cover a capability, so "it compiled and passed" is not the
    whole claim: `-mfpu=neon-vfpv4` that silently lost the flag would still pass
    every KAT while covering nothing. These assertions are what makes the coverage
    real.

    变体存在的意义就是覆盖一项能力，所以"能编能跑"不是全部论断：一个静默丢掉的
    `-mfpu=neon-vfpv4` 仍然能通过所有 KAT，却什么都没覆盖。这些断言就是把覆盖变真的那一步。
    """
    for rule in asserts or []:
        text = artifact_text(cc, elf, rule.get("tool", "readelf"), objects)
        what = rule.get("what", "artifact")
        if "expect" in rule and rule["expect"] not in text:
            raise RuntimeError(f"{elf.name}: {what}: expected {rule['expect']!r} in the artifact")
        if "forbid_re" in rule and re.search(rule["forbid_re"], text):
            raise RuntimeError(f"{elf.name}: {what}: forbidden pattern {rule['forbid_re']!r} is present")


def variants_of(board):
    """The base build, followed by every declared variant of that board."""
    return [None, *board.get("variants", [])]



def emit_k2_model_header():
    """Both k2 generated headers: the weights/input the hand-written chain bakes in,
    and the blob image the executor parses.

    They come from one file apiece and one file only -- the blob -- so a model change
    cannot land on one path and miss the other. 两个头都只从一个来源（blob）生成，
    所以模型改动不可能只落在一条路径上。
    """
    run([sys.executable, PREP / "emit_k2_model_header.py",
         ROOT / "test_package/resources/kat/k2_conv_micro/model.blob",
         ROOT / "test_package/resources/kat/k2_conv_micro/input.bin",
         GENERATED / "kat_k2_model.h"])
    run([sys.executable, PREP / "emit_blob_image.py",
         "--name", "kat_k2_blob",
         ROOT / "test_package/resources/kat/k2_conv_micro/model.blob",
         GENERATED / "kat_k2_blob.h"])


def emit_k2i8_headers():
    """Emit the three C arrays the int8 chain travels with, all from one committed fixture.

    The expectation is the part that matters: it comes from `expect.i8`, which the generator
    computed with an integer reference **before** any executor ran. The chain compares its own
    output against those bytes on the target, so a target run is checked against a separate
    implementation of the rule rather than against a host run of itself.

    The arrays are named `*_bytes` because `kat_k2i8_expect` is already a function in the chain's
    header -- the first version of this emitter generated an array of that name and every build of
    the chain failed to compile, which is a cheap lesson to only need once.

    三个 C 数组都来自同一个入库夹具。要紧的是期望：它来自 `expect.i8`，而那是生成器在任何执行器
    跑起来之前用整数参考算出的。链在目标端把自己的输出与那些字节相比，于是目标运行对齐的是**那份
    规则的另一个实现**，而不是它自己的宿主运行。

    数组名带 `_bytes`，因为 `kat_k2i8_expect` 已经是链头里的函数——本发射器的第一版就生成了这个
    同名数组，于是链的每次构建都编不过。这个教训一次就够。
    """
    fixture = ROOT / "test_package/resources/kat/k2_i8_conv"
    run([sys.executable, PREP / "emit_blob_image.py", "--name", "kat_k2i8_blob",
         fixture / "model.blob", GENERATED / "kat_k2i8_blob.h"])

    def raw_header(source, name, out):
        data = pathlib.Path(source).read_bytes()
        body = ",".join(str(byte) for byte in data)
        pathlib.Path(out).write_text(
            f"/* Generated by run_kat.py from {pathlib.Path(source).name}; do not edit. */\n"
            "#pragma once\n"
            f"static const unsigned char {name}[] = {{{body}}};\n"
            f"#define {name.upper()}_LEN {len(data)}u\n")

    raw_header(fixture / "input.i8", "kat_k2i8_input_bytes", GENERATED / "kat_k2i8_input.h")
    raw_header(fixture / "expect.i8", "kat_k2i8_expect_bytes", GENERATED / "kat_k2i8_expect.h")


def emit_k2m_blob_header():
    """Generate the MobileNet blob, then its `.incbin` object and its size header.

    The model is generated rather than committed: it is 1.4 MB of reproducible LCG weights,
    and a checked-in binary of that size would be an artifact nobody can review. Two steps,
    one source -- `emit_mobilenet_blob.py` writes `model.blob`, and everything the targets
    see is derived from that file.

    模型是生成的，不进仓：它是 1.4 MB 可复现的 LCG 权重，把这么大的二进制提交进去等于放了
    一个没人能 review 的产物。两步、一个来源——`emit_mobilenet_blob.py` 写出 `model.blob`，
    目标看到的东西全部从那个文件派生。
    """
    override = os.environ.get("INFRINFER_K2M_BLOB")
    if override:
        # The chain is data-driven -- input size, arena and hash all come from whatever blob is
        # emitted -- so a real, imported MobileNetV3 runs through exactly the same code path as
        # the synthetic one. The blob is named by the environment because it is 2 MB of weights
        # produced by a machine-level export, not something to commit or to rebuild silently.
        #
        # 这条链是数据驱动的——输入尺寸、arena、哈希全部来自实际发射的那个 blob——所以真实的、导入
        # 来的 MobileNetV3 与合成网走的是同一条代码路径。blob 由环境变量指定，因为那是 2 MB 权重、
        # 由机器级导出产生，既不该入库也不该被悄悄重建。
        model = pathlib.Path(override).expanduser().resolve()
        if not model.is_file():
            raise SystemExit(f"run_kat: INFRINFER_K2M_BLOB names {model}, which is not a file")
        print(f"[run_kat] k2m uses the imported blob {model} "
              f"({model.stat().st_size} B)", flush=True)
    else:
        model = BUILD / "mobilenet" / "model.blob"
        run([sys.executable, PREP / "emit_mobilenet_blob.py",
             "--width", "0.25", "--res", "96", "--write", "--out", str(model.parent)])
    run([sys.executable, PREP / "emit_blob_image.py", "--segment",
         "--name", "kat_k2m_blob", model, GENERATED / "kat_k2m_blob.h"])


def build_host_emit(cc, chain):
    out = BUILD / f"{chain}_emit"
    if chain == "k2":
        emit_k2_model_header()
    elif chain == "k2i8":
        emit_k2i8_headers()
    elif chain == "k2i8":
        emit_k2i8_headers()
    elif chain == "k2m":
        emit_k2m_blob_header()
    # `-DKAT_EMIT` is what selects the emitter `main`. Without it this step compiled the
    # *check* binary, ran it, and appeared to work only because a `<chain>_expected.bin`
    # from an earlier run happened to be lying around -- i.e. step 1 of 4 was a no-op and
    # a model that changed numerically would have left the baked header stale while every
    # target kept passing against it (found 2026-11-27, gaps G-100).
    #
    # The objects go to their own directory: emit and check builds differ by one macro,
    # and object files are named by source, so sharing a directory makes the second build
    # silently overwrite the first.
    #
    # `-DKAT_EMIT` 才决定选哪个 `main`。没有它，这一步编出的是**校验**二进制，跑一下也
    # "成功"——只因为上一次留下的 `<chain>_expected.bin` 还躺在旁边。也就是说四步里的
    # 第一步其实什么都没做，而一个数值上变了的模型会让烘进去的头部过期，同时所有目标继续
    # 对着它通过（2026-11-27 发现，gaps G-100）。对象要放进自己的目录：两个构建只差一个
    # 宏，而对象名按源文件命名，共用一个目录就是让后一个悄悄覆盖前一个。
    objects = objects_for(chain, cc, cxx_for(cc), BUILD / f"{chain}_emit_objs", ["-DKAT_EMIT"])
    link = cxx_for(cc) if chain in CHAINS_NEEDING_CPP else cc
    run([link, "-O2", *TRIM_LINK, *objects, "-lm", "-o", out])
    blob = BUILD / f"{chain}_expected.bin"
    run([out, blob])
    return blob


def emit_header(blob, chain):
    header = GENERATED / f"kat_{chain}_image.h"
    run([sys.executable, PREP / "emit_kat_image.py", blob, header,
         "--id", CHAIN_IDS[chain]])
    return header


def build_host_check(cc, chain):
    out = BUILD / f"{chain}_host_check"
    objects = objects_for(chain, cc, cxx_for(cc), BUILD, [])
    link = cxx_for(cc) if chain in CHAINS_NEEDING_CPP else cc
    run([link, "-O2", *TRIM_LINK, *objects, "-lm", "-o", out])
    done = subprocess.run([str(out)], capture_output=True, text=True)
    print(f"[run_kat] host check ({chain}) -> exit {done.returncode}", flush=True)
    if done.returncode != 0:
        if done.stderr:
            print(done.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"run_kat: host KAT {chain} failed with {done.returncode}")


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


def build_target(board_name, chain, gcc_arm, gcc_rv, variant=None, tier=None):
    board = load_board(board_name)
    vid = variant["id"] if variant else "base"
    tag = f"{chain}_{board_name.replace('-', '_')}_{vid}" if variant else f"{chain}_{board_name.replace('-', '_')}"
    vflags = list(variant.get("flags", [])) if variant else []
    opt = variant.get("opt", "-O2") if variant else "-O2"
    cc = gcc_arm if gcc_arm is not None else gcc_rv
    # A tier build is one (board, tier) pair, so its objects go to their own directory: every tier
    # has a `register.c`, and a shared directory would have one overwrite the other.
    out_dir = BUILD / f"{chain}_{tier}" if tier else BUILD
    out_dir.mkdir(parents=True, exist_ok=True)
    if tier:
        spec = load_board(board_name).get("tier", {})
        vflags = [*vflags, *spec.get("flags", [])]
    if board.get("mode") == "userspace":
        # Linux userspace: the C runtime does the startup, so there is no linker
        # script, no `_start` and no semihosting. `-static` makes the ELF
        # self-contained, which is what lets qemu-user run it without a `-L` sysroot
        # (measured: without it the run needs the target's glibc, which we do not
        # stage). The exit status of the process is the verdict, exactly as on the
        # host.
        #
        # Linux 用户态：启动由 C 运行时完成，所以没有链接脚本、没有 `_start`、没有半主机。
        # `-static` 让 ELF 自包含，qemu-user 才能不带 `-L` sysroot 跑（实测：动态链接就需要
        # 目标端 glibc，而这一步没铺）。进程退出码就是判定，与宿主一致。
        elf = BUILD / f"{tag}.elf"
        if chain == "k2":
            emit_k2_model_header()
        elif chain == "k2i8":
            emit_k2i8_headers()
        objects = objects_for(chain, gcc_arm, cxx_for(gcc_arm), out_dir, vflags, opt, tier=tier)
        link = cxx_for(gcc_arm) if chain in CHAINS_NEEDING_CPP else gcc_arm
        run([link, "-static", opt, *vflags, *TRIM_LINK, *objects, "-lm", "-o", elf])
        check_artifact(cc, elf, variant.get("asserts") if variant else None, objects)
        return elf
    if board["qemu"].endswith("arm"):
        ld = arm_ld_for(board, BUILD / f"{tag}.ld")
        elf = BUILD / f"{tag}.elf"
        if chain == "k2":
            emit_k2_model_header()
        elif chain == "k2i8":
            emit_k2i8_headers()
        target_flags = [f"-mcpu={board['cpu']}", "-mthumb", *vflags]
        objects = objects_for(chain, gcc_arm, cxx_for(gcc_arm), out_dir, target_flags, opt, tier=tier)
        link = cxx_for(gcc_arm) if chain in CHAINS_NEEDING_CPP else gcc_arm
        run([link, *target_flags, opt, "-nostartfiles", "--specs=nosys.specs",
             *TRIM_LINK, "-T", ld, *objects, "-lm", "-o", elf])
        image = BUILD / f"{tag}.bin"
        run([gcc_arm.parent / "arm-none-eabi-objcopy", "-O", "binary", elf, image])
        check_artifact(cc, elf, variant.get("asserts") if variant else None, objects)
        return image
    elf = BUILD / f"{tag}.elf"
    if chain == "k2":
        emit_k2_model_header()
    elif chain == "k2i8":
        emit_k2i8_headers()
    elif chain == "k2i8":
        emit_k2i8_headers()
    rv_spec = load_board(board_name).get("tier", {}) if tier else {}
    rv_flags = [f"-march={rv_spec.get('march', board['march'])}",
                f"-mabi={rv_spec.get('mabi', board['mabi'])}", "-mcmodel=medany", *vflags]
    objects = objects_for(chain, gcc_rv, cxx_for(gcc_rv), out_dir, rv_flags, opt, tier=tier)
    link = cxx_for(gcc_rv) if chain in CHAINS_NEEDING_CPP else gcc_rv
    run([link, *rv_flags, opt, "-nostartfiles", "-T", HERE / "kat_rv.ld",
         *TRIM_LINK, "-Wl,-e,_start", *objects, "-lm", "-o", elf])
    check_artifact(cc, elf, variant.get("asserts") if variant else None, objects)
    return elf


def run_chain(chain, boards, cc, use_variants=True):
    print(f"[run_kat] === {chain} ({CHAIN_IDS[chain]}) ===", flush=True)
    if chain == "k2m":
        kept = [name for name in boards if name in K2M_BOARDS]
        if not kept:
            print(f"[run_kat] k2m: no FPU MCU in this board selection, nothing to run", flush=True)
            return []
        if len(kept) != len(boards):
            print(f"[run_kat] k2m: restricted to {', '.join(kept)} (see K2M_BOARDS)", flush=True)
        boards = kept
    print(f"[run_kat] {chain} 1/4 host emit", flush=True)
    blob = build_host_emit(cc, chain)
    print(f"[run_kat] {chain} 2/4 emit image header", flush=True)
    emit_header(blob, chain)
    print(f"[run_kat] {chain} 3/4 host check", flush=True)
    build_host_check(cc, chain)

    print(f"[run_kat] {chain} 4/4 targets", flush=True)
    failures = []
    for board_name in boards:
        board = load_board(board_name)
        toolchain = board["toolchain"]
        probe = toolchain["version_probe"]
        if board.get("mode") == "userspace":
            gcc_arm, gcc_rv = tool_for("ARM_LINUX_BIN", "arm-t*",
                                       f"{toolchain['triplet']}-gcc", probe), None
        elif toolchain["triplet"] == "riscv-none-elf":
            gcc_arm, gcc_rv = None, tool_for("RV_ELF_BIN", "rv-*", "riscv-none-elf-gcc", probe)
        else:
            gcc_arm, gcc_rv = tool_for("ARM_EABI_BIN", "arm-*", "arm-none-eabi-gcc", probe), None
        for variant in (variants_of(board) if use_variants else [None]):
            cell = board_name if variant is None else f"{board_name}:{variant['id']}"
            try:
                image = build_target(board_name, chain, gcc_arm, gcc_rv, variant)
            except subprocess.CalledProcessError as exc:
                print(f"[run_kat] BUILD FAILED {chain}/{cell} (rc={exc.returncode})",
                      file=sys.stderr)
                failures.append(cell)
                continue
            except RuntimeError as exc:
                print(f"[run_kat] ARTIFACT FAILED {chain}/{cell}: {exc}", file=sys.stderr)
                failures.append(cell)
                continue
            done = subprocess.run([sys.executable, str(HERE.parent / "qemu/run_kat.py"),
                                   board_name, str(image), cell])
            if done.returncode != 0:
                failures.append(cell)
    return failures


def run_k3(boards, cc, use_variants=True):
    """k3: no expectation stage -- the chain compares the tier row against the scalar row.

    The other chains bake an expectation on the host and compare bytes on the target, which assumes
    the host is the reference. Here the reference is the scalar row **on the same target**, so an
    expectation would be meaningless, and the verdict is the exit code:

        0            every generated cell agreed
        12           a cell the matrix calls `done` was not registered for the tier
        20..27       the first disagreeing cell index (+20, capped at 7)

    A board profile that declares no tier is skipped with its reason printed: the ARMv7-A baseline,
    for example, has no NEON at all, and the claim lives in its `neon-vfpv4` variant.
    """
    print(f"[run_kat] === k3 ({CHAIN_IDS['k3']}) ===", flush=True)
    failures = []
    for board_name in boards:
        board = load_board(board_name)
        spec = board.get("tier")
        if not spec:
            print(f"[run_kat] k3 SKIP {board_name}: the profile declares no tier", flush=True)
            continue
        tier = spec["name"]
        run([sys.executable, PREP / "emit_k3_cells.py", MATRIX, tier,
             TIER_CELLS_HEADER])
        toolchain = board["toolchain"]
        probe = toolchain["version_probe"]
        if board.get("mode") == "userspace":
            gcc_arm, gcc_rv = tool_for("ARM_LINUX_BIN", "arm-t*",
                                       f"{toolchain['triplet']}-gcc", probe), None
        elif toolchain["triplet"] == "riscv-none-elf":
            gcc_arm, gcc_rv = None, tool_for("RV_ELF_BIN", "rv-*", "riscv-none-elf-gcc", probe)
        else:
            gcc_arm, gcc_rv = tool_for("ARM_EABI_BIN", "arm-*", "arm-none-eabi-gcc", probe), None
        # `claims` is the whole rule: a build either claims the tier or it does not. The skip is
        # printed with the profile's reason, because "no measurement here" and "measured green"
        # must never look the same in a log.
        claims = spec.get("claims", ["base"])
        for variant in (variants_of(board) if use_variants else [None]):
            vid = variant["id"] if variant else "base"
            cell = f"{board_name}:{vid}"
            if vid not in claims:
                print(f"[run_kat] k3 SKIP {cell}: {spec.get('skip_why', 'this build does not claim the tier')}",
                      flush=True)
                continue
            try:
                image = build_target(board_name, "k3", gcc_arm, gcc_rv, variant, tier=tier)
            except subprocess.CalledProcessError as exc:
                print(f"[run_kat] BUILD FAILED k3/{cell} (rc={exc.returncode})", file=sys.stderr)
                failures.append(cell)
                continue
            except RuntimeError as exc:
                print(f"[run_kat] ARTIFACT FAILED k3/{cell}: {exc}", file=sys.stderr)
                failures.append(cell)
                continue
            # `--qemu-arg=-cpu` rather than `--qemu-arg -cpu`: argparse reads a leading dash as a
            # flag and refuses the value (it did, and the run died as a usage error).
            qemu_extra = [f"--qemu-arg={arg}" for arg in spec.get("qemu_args", [])]
            done = subprocess.run([sys.executable, str(HERE.parent / "qemu/run_kat.py"),
                                   board_name, str(image), cell, "--tier", tier, *qemu_extra])
            if done.returncode != 0:
                failures.append(cell)
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", choices=["k0", "k1", "k2", "k2m", "k2i8", "k3", "both", "all"],
                    default="both")
    ap.add_argument("--cc", default=os.environ.get("CC", "cc"))
    ap.add_argument("--boards", nargs="*", default=None,
                    help="board stems to run (default: every board profile)")
    ap.add_argument("--skip-targets", action="store_true", help="host pipeline only")
    ap.add_argument("--no-variants", action="store_true",
                    help="skip the per-board compile-option variants")
    args = ap.parse_args()

    BUILD.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)

    chains = {"both": ["k0", "k1"], "all": ["k0", "k1", "k2", "k2m", "k3"]}.get(args.chain, [args.chain])
    boards = args.boards or sorted(p.stem for p in BOARDS.glob("*.json"))

    all_failures = {}
    for chain in chains:
        if chain in TIER_CHAINS:
            if args.skip_targets:
                print(f"[run_kat] {chain}: tier parity needs a target; nothing to do on the host "
                      f"(the x86 tier runs its parity cases in `conan create`)", flush=True)
                continue
            failures = run_k3(boards, args.cc, use_variants=not args.no_variants)
            if failures:
                all_failures[chain] = failures
            continue
        if args.skip_targets:
            print(f"[run_kat] {chain}: host pipeline only", flush=True)
            blob = build_host_emit(args.cc, chain)
            emit_header(blob, chain)
            build_host_check(args.cc, chain)
            continue
        failures = run_chain(chain, boards, args.cc, use_variants=not args.no_variants)
        if failures:
            all_failures[chain] = failures

    if all_failures:
        for chain, failed in all_failures.items():
            print(f"[run_kat] FAILED {chain}: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"[run_kat] all chains pass on host + {len(boards)} QEMU targets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
