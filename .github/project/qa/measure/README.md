# qa/measure — the harnesses behind every performance number

Every throughput number quoted in `workspace/audit/gaps.md` came from one of these
files. They live here rather than in `workspace/` because `workspace/qa/**` was
moved into the repository on 2026-11-27 (gaps G-105) — a harness that only exists
on one machine makes its own numbers unverifiable.

These are **heavy checks**: they need a machine with the host toolchain, the
Conan cache and (for two of them) the XNNPACK package, so they are declared in
`toolchains.json:heavy_checks` and never run from `ci.sh` (rule Y5).

| file | what it answers | needs |
|---|---|---|
| `measure_tier_ab.cxx` | scalar vs this build's vector tier, per convolution layer | nothing but the library |
| `measure_xnnpack_ab.cxx` | fp32 conv: this library vs XNNPACK, one thread | `xnnpack/cci.20241203`, `pthreadpool`, `cpuinfo` |
| `measure_xnnpack_i8.cxx` | int8 conv: this library vs XNNPACK qs8/qc8w | same |
| `op_census.py` | which ONNX operators real exported models need, checked against `matrix.yml` | `torch`, `torchvision`, `onnx`, `pyyaml` |

## Running them

The two C++ harnesses link against a **clean, uninstrumented** build of the
library, because the Conan `conan create` archive carries gcov counters and those
distort the timings. Build the objects yourself and pass them on the command line:

```sh
mkdir -p /tmp/cleanlib
for f in $(find src -name '*.c' | grep -vE 'src/(ctest|doctest)'); do
    gcc -O2 -std=c11 -Iinclude -ffp-contract=off -c "$f" -o "/tmp/cleanlib/$(echo "$f" | tr '/' '_').o"
done
for f in $(find src -name '*.cpp' | grep -vE 'src/(cpptest|etl)'); do
    g++ -O2 -std=c++17 -Iinclude -ffp-contract=off -fno-exceptions -fno-rtti -c "$f" -o "/tmp/cleanlib/$(echo "$f" | tr '/' '_').o"
done
```

Then link one harness against those objects (plus XNNPACK for the two comparisons):

```sh
g++ -O2 -std=c++17 -Iinclude -I$XNN/include -I$PP/include -I$CPU/include \
    .github/project/qa/measure/measure_xnnpack_ab.cxx $(ls /tmp/cleanlib/*.o | grep -vE 'src_(ctest|doctest|cpptest|etl)') \
    $XNN/lib/libXNNPACK.a $XNN/lib/libmicrokernels-prod.a $PP/lib/libpthreadpool.a \
    $(find $CPU -name 'libcpuinfo*.a' | head -1) -lm -lpthread -o /tmp/measure_xnn
```

`INFR_MB_REPS=1` reduces the repetition count, which the emulator comparisons need
because emulated runs are orders of magnitude slower.

The census needs no build:

```sh
python3 .github/project/qa/measure/op_census.py
```

## What these numbers are not

Absolute throughput is only meaningful on the machine that produced it, and on a
host with `avx512_vnni` / `amx_int8` the XNNPACK comparison is measuring an ISA
gap as much as a code-quality gap (gaps G-128/G-130). Read
`workspace/PLAN-2026-11-25.md` §3.11 (ADR-008) before quoting a number from here:
direction claims need two instruments agreeing, magnitude claims need an
instrument calibrated against ground truth.
