#!/usr/bin/env bash
# Does the C++17 orchestration link on bare metal?
#
# `src/runtime` is C++17 by design, and the QEMU KATs are C images. Before a graph
# KAT can run on a target, one question has to be answered with a measurement rather
# than a plan: can these translation units be compiled and linked for Cortex-M with
# no exceptions, no RTTI and no hosted C++ library?
#
# This probe compiles the library sources for the target, links a tiny C driver that
# calls one real operator through the public C ABI, and runs it under QEMU. Exit 0
# means the call returned INFR_OK; anything else prints what failed.
#
# Usage: workspace/qa/kat/arm_link_probe.sh [cpu] [board]
#        defaults: cortex-m3 / mps2-an385
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
CPU=${1:-cortex-m3}
BOARD=${2:-mps2-an385}
OUT=${OUT:-/tmp/infr-arm-probe}

BIN_DIR=${ARM_EABI_BIN:-/home/zc/.conan2/p/b/arm-tfa6e1855f3530/p/bin}
GXX="$BIN_DIR/arm-none-eabi-g++"
GCC="$BIN_DIR/arm-none-eabi-gcc"

mkdir -p "$OUT"
rm -f "$OUT"/*.o "$OUT"/probe.elf "$OUT"/probe.bin

COMMON=(-mcpu="$CPU" -mthumb -O2 -ffreestanding -fno-exceptions -fno-rtti -fno-threadsafe-statics)
INCLUDES=(-I"$ROOT/include")

echo "[probe] compiling C++ orchestration for $CPU"
for src in "$ROOT"/src/runtime/*.cpp; do
  "$GXX" "${COMMON[@]}" -std=c++17 "${INCLUDES[@]}" -c "$src" -o "$OUT/$(basename "${src%.cpp}").o"
done

echo "[probe] compiling C kernels for $CPU"
for src in "$ROOT"/src/kernel/contract.c "$ROOT"/src/kernel/scalar/*.c; do
  "$GCC" "${COMMON[@]}" -std=c11 "${INCLUDES[@]}" -c "$src" -o "$OUT/$(basename "${src%.c}").o"
done

echo "[probe] compiling the driver"
"$GCC" "${COMMON[@]}" -std=c11 "${INCLUDES[@]}" -c "$ROOT/workspace/qa/kat/arm_link_probe.c" -o "$OUT/probe_main.o"

echo "[probe] linking"
# The C++ objects may reference libstdc++ symbols; linking the same driver with the
# C driver keeps the probe honest about what the target actually needs.
"$GXX" "${COMMON[@]}" -nostartfiles --specs=nosys.specs -Ttext=0x00000000 -Wl,-e,_start \
  "$OUT"/*.o -o "$OUT/probe.elf"

"$BIN_DIR/arm-none-eabi-objcopy" -O binary "$OUT/probe.elf" "$OUT/probe.bin"
SIZE=$(stat -c %s "$OUT/probe.bin")
echo "[probe] image $SIZE B"

echo "[probe] running under QEMU ($BOARD)"
set +e
# `-kernel` with an ELF honours the ELF entry point; loading the raw binary at 0
# would start at whatever the linker happened to place first.
qemu-system-arm -M "$BOARD" -nographic -monitor none -serial none \
  -kernel "$OUT/probe.elf" \
  -semihosting-config enable=on,target=native >/dev/null 2>&1
RC=$?
set -e

if [ "$RC" -eq 0 ]; then
  echo "[probe] PASS  the C++ runtime linked and ran on bare metal ($CPU, $SIZE B)"
else
  echo "[probe] FAIL  qemu exit $RC (see $OUT)" >&2
fi
exit "$RC"
