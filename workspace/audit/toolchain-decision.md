# 工具链版本收敛（P0.6 证据，2026-11-25）

> 结论先行：**ARM 锁 `arm-toolchain/11.3.rel1`（维持既定决策）；RV 主线锁 `rv-toolchain/14.3.rel1`，
> `12.3.rel1` 只作 scalar 兼容抽查，`15.3.rel1` 为候选不承诺。**
> 每条结论后附"实测/原文"，不凭记忆。

## 1. ARM：锁 11.3.rel1

**实测**（本机缓存 `arm-toolchain/11.3.rel1`，Arm GNU Toolchain 11.3.1）：

| 探测 | 结果 |
|---|---|
| `-mcpu=cortex-a7 / a55`（NEON 档） | OK |
| `-mcpu=cortex-m4 / m7 / m33`（DSP 档） | OK |
| `-mcpu=cortex-m55` + `arm_mve.h` 的 `vaddq_u32` | OK（`lib/gcc/arm-none-linux-gnueabihf/11.3.1/include/arm_mve.h`） |
| `-mcpu=cortex-m85` | **UNSUPPORTED**（GCC 13 release notes 才加入 cortex-m85） |
| 本项目 `src/kernel/contract.c` | 编成 ARM 32-bit ELF（EABI5）通过 |

**决策**：维持 11.3.rel1（与 PLAN §3.3/§7 及团队既定决策一致），理由：既定 + 覆盖 A7/A55 + M4/M7/M33 + M55/MVE 可编译（仅作功能验证）+ 本机已构建。
**范围修正（2026-11-25，用户确认 + 矩阵核实）**：`supportable-board-matrix.yml`（gold standard）的 MCU **只有 M0/M3/M4/M7/M23/M33**——**M55 与 M85 都不在支持范围**。因此：`arm_m` 档的实机形态是 **M4/M7/M33（DSP）**；MVE 只能做编译与 QEMU 功能验证，**不得写入平台矩阵**（INV-7）。`cortex-m85` 在 11.3.1 上本来也不支持，不再是缺口。

## 2. RV：主线 14.3.rel1（RVV 收益是唯一决定因素）

**实测**（`conan create` 构建 xpack 裸机包后逐项量；构建方式见文末）：

| 事实 | `rv-toolchain/12.3.rel1`（xpack GCC 12.3.0） | `rv-toolchain/14.3.rel1`（xpack GCC 14.2.0） |
|---|---|---|
| `riscv_vector.h` 是否存在 | **不存在**（`fatal error: No such file`） | 存在 |
| `__riscv_v_intrinsic` | — | **12000（v0.12 intrinsic API）** |
| RVV 自动向量化（axpy 循环，`-O3 -march=rv64gcv`，数 `vsetvli`） | **0** | **2** |
| intrinsic 编译 | 不可能（无头文件） | 通过（`__riscv_vadd_vv_u32m1`） |

**GCC 官方 release notes（原文）**：
- GCC 13：*"Support for vector intrinsics as specified in version 0.11 of the RISC-V vector intrinsic specification"* → 13 才有最初级 intrinsics，且是 **v0.11**。
- GCC 14：*"The SLP and loop vectorizer are now enabled for RISC-V when the vector extension is enabled"* + 新增 `-mrvv-max-lmul`、标准 vector calling convention、`target` attribute → **14 是第一个 autovec 真正可用的 train**。

**论证（RVV 面的收益）**：
1. 我们的 RVV 档由两层构成：L-k1 `autovec`（同源 C 交给向量化）与 L-k2 `rvv`（intrinsics）。
   在 12.3 上**两层都不存在**（无头文件、向量化器关闭）→ 锁 12.3 等于放弃 RVV 档，只剩 scalar。
2. 13.3 有 v0.11 intrinsics 但**仍无 autovec**（GCC 14 notes 明说 "now enabled"），且 v0.11 与后续 API 不兼容 → 只适合过渡。
3. 14.3：autovec 可用（实测 2×`vsetvli`）+ intrinsics v0.12 可用 → **最小可行 RVV 主线**（PLAN §3.3 的"rvv 最低 13.3"应改为 **14.3**；13.3 只能跑 scalar/intrinsic 手写，没有 autovec 兜底）。
4. 15.3：更新，但其 Linux 侧是 glibc **2.44**（板卡过新）、实际战龄最短；且 intrinsic API 是否到 ratified 1.0 **尚未实测**（release notes 措辞与宏值在 14 上已经证明会不一致——14 的 notes 写 "version 1.0"，实测宏是 12000/v0.12）。→ 列为候选，不进主线；若将来 rvv 档要用 1.0 intrinsic API，先建 15.3 实测 `__riscv_v_intrinsic` 再决策。

**API 版本的纪律**：intrinsic API 版本以 **`__riscv_v_intrinsic` 宏实测值**为准，写进 `gate/toolchain`（PLAN P3），不信 release notes 措辞。

## 3. 构建方式（可复现）

```bash
# 14.3.rel1 裸机（配方硬编码 version=12.3.rel1，用临时副本改版本，不动共享仓）
cp -r ~/Projects/HeT-FTI/rv-toolchain /tmp/rv-toolchain-143
sed -i 's/^    version = "12.3.rel1"/    version = "14.3.rel1"/' /tmp/rv-toolchain-143/conanfile.py
cd /tmp/rv-toolchain-143 && conan create . -pr:b=default \
  -s:h os=baremetal -s:h arch=riscv64 -s:h compiler=gcc -s:h compiler.version=14 \
  -s:h compiler.libcxx=libstdc++11 -s:h compiler.cppstd=gnu17 -s:h build_type=Release \
  --build-require --build=missing

# 12.3.rel1 裸机（原仓默认版本，直接建）
cd ~/Projects/HeT-FTI/rv-toolchain && conan create . -pr:b=default \
  -s:h os=baremetal -s:h arch=riscv64 -s:h compiler=gcc -s:h compiler.version=12 \
  -s:h compiler.libcxx=libstdc++11 -s:h compiler.cppstd=gnu17 -s:h build_type=Release \
  --build-require --build=missing
```

## 4. 已知问题（本次构建暴露）

- **14.3.rel1 的 test_package 链接失败**：`libc.a(libc_a-getentropyr.o): undefined reference to '_getentropy'`（xpack 14.2 自带 newlib 的已知缺口）。包本身已产出并能编译（本文件全部实测都用它），但**裸机链接 C++ 运行库**会踩到；P4 裸机 KAT 前须解决（补 `_getentropy` 桩或换 newlib 版本）。12.3 的 test_package 链接正常。
- Linux RV64（Bootlin）变体**本次未构建、未验证**（需要 ~677MB 下载）；RVV 档只承诺裸机 + Linux 用户态两条路，后者在 P4.1 前按 INV-6 显式 skip。
- RVV intrinsic 只验证了「编译通过」，**未运行**（QEMU 已于 2026-11-25 装好：8.2.2，六个目标冒烟通过；但 intrinsic 的运行/逐位一致性属 `gate/kat`，P1.3/P4.1 前不算证过）。

## 5. 落档要求

- 版本锁的**唯一来源**最终要进 `metadata.json` 消费方（CI workflow 的 toolchain pin）与 `gate/toolchain`（P3）；当前 metadata 只有 `on_board_validation.toolchain_version`（选板用），**不承载构建工具链锁**。
- 本文件的结论在 P0.6 完成时应转成 CI 常量 + gate 断言（`__riscv_v_intrinsic == 12000`、`-mcpu=cortex-m85` 必须失败等）。
