# P1.3 KAT 设计（agent 自拟 · 2026-11-25）

> 本文件回答"KAT 是什么、为什么存在、跑哪些目标、怎么判"。目标矩阵以 fcpp 的
> `supportable-board-matrix.yml`（12 个已验硬件）为参考，但**按 ISA 类别定义覆盖面**，
> 因为本库的目标是"大覆盖面"，不是只覆盖手上这几块板。

## 1. KAT 是什么（一句话版）

**KAT = Known-Answer Test（已知答案测试）**：把**同一个模型 blob + 同一份输入**放到
**每一个目标**上跑，结果与**已入库的期望输出**比：

- 与 golden（ORT/PyTorch 产的期望值）比 → 证明**算法对**（容差按 dtype）；
- **跨目标之间逐位一致** → 证明"同一张图在 M3/M7/A55/RV64 上给出一模一样的结果"。

没有 KAT 时，平台矩阵只能写"能编过"；有了 KAT，"结果一致"才是可复算的证据（M2 里程碑）。

## 2. 目标矩阵（自拟，按 ISA 类覆盖）

### Tier A · 宿主快车道（每次 PR）

| 目标 | 覆盖 |
|---|---|
| Linux x86_64（宿主） | 参考实现 + oracle 数值对齐；ASan/UBSan |

### Tier B · QEMU（本地车道，`LOCAL_ONLY`）

| id | QEMU 机器 | CPU | 覆盖的 ISA 类 | 退出协议 |
|---|---|---|---|---|
| `qemu-m3-an385` | `mps2-an385` | cortex-m3 | Thumb-2，无 DSP 无 FPU | ARM semihosting SYS_EXIT |
| `qemu-m4-an386` | `mps2-an386` | cortex-m4 | **DSP（SMLAD 等）** | 同上 |
| `qemu-m7-an500` | `mps2-an500` | cortex-m7 | DSP + 双精度 FPU | 同上 |
| `qemu-m33-an505` | `mps2-an505` | cortex-m33 | **ARMv8-M（含 DSP）** | 同上 |
| `qemu-rv32-virt` | `virt` | rv32imac | RV32 标量（无 RVV 的现实面） | SiFive finisher（0x100000←0x5555） |
| `qemu-rv64-virt` | `virt` | rv64imac / rv64gcv | RV64 标量；RV64+RVV 自动向量化 | 同上 |
| `qemu-aarch64-user` | qemu-aarch64 | aarch64 | Linux ARM64 用户态（NEON） | 进程退出码 |

**矩阵依据**：`supportable-board-matrix.yml` 的 MCU 类是 M0/M3/M4/M7/M23/M33；QEMU 只提供
M3/M4/M7/M33 四种（M0/M23 无 QEMU 机型，靠实机补）；A 类实机是 A7/A53/A55/A76（全 NEON，
QEMU 用户态 aarch64 可覆盖同一 ISA 面）；RV 侧主线 RV64+V，RV32 是标量现实面。

### Tier C · 实机（`LOCAL_ONLY`，有板才跑）

直接引用 `supportable-board-matrix.yml` 的 12 个 target id（6 Linux + 6 baremetal）；
每块板跑**同一份 KAT blob**，结论带 `{commit, host, date}` 写 `claim-basis.json`。

## 3. 判据

| 比较 | FP32 | INT8/INT16 |
|---|---|---|
| 对 golden（ORT） | 相对 ≤1e-5 且绝对 ≤1e-6 | 逐位相等（MAC 序固定） |
| 跨目标 | **逐位相等** | **逐位相等** |

## 4. 执行机制（关键决定）

**目标内比对 + 退出码**：KAT 程序把期望值以字节数组编进镜像，在目标上跑算子后自己比对，
`exit(0)` = pass，非零 = fail。

理由（本机实测）：
- ARM semihosting 的 `SYS_EXIT` 在所有 M 板可用（M3/M4/M7，rc=0）；
- **`printf` 走 semihosting `_write` 在无控制台连接时会阻塞**（M3 实测挂起，rdimon init 单独测试 rc=0）→ 不把"打印"放进判定路径；
- 控制台输出只作可选的调试通道（需要时配 `-chardev`），不参与 pass/fail。

## 5. 与 QEMU 的边界（复述 PLAN §3.2）

QEMU 证明**正确性/鲁棒性/ABI**，不证明**性能**；任何 QEMU 数字不进入 README 矩阵。

## 6. 待办切片（按最小返工排序）

1. ✅ QEMU 六目标冒烟（含 M33 机型确认）+ 收尾协议定型（本文件）
2. ✅ runner 骨架：`workspace/qa/qemu/boards/*.json`（6 份）+ `run_kat.py`（本轮实测：**6/6 PASS**，任一镜像死循环会 **FAIL timeout**）
3. ✅ **管线端到端（k0_actquant）**：`workspace/qa/kat/` + `workspace/qa/prep/emit_kat_image.py`，链路 = 量化→反量化→relu（走注册表）→量化；期望值只由**宿主 EMIT 构建**产出（`KATR` blob），再由脚本烘成 C 字节数组，目标端 `memcmp` 比对——**宿主 + 6 个 QEMU 目标全部 exit 0**；把期望改一个字节 → 目标 exit **11**、runner FAIL（反证有效）
4. ▶ 第一个"模型级" KAT：`k1_mlp3`（16→32→4）**阻塞于 Gemm 格未落地**（P2.x）；落地即复用 k0 的管线，只需换链
5. 矩阵固化：census 的 tier 列与 KAT 目标表对齐（`gate/ports` 前身）

## 7. 环境事实（实测，避免二次踩坑）

| 事实 | 证据 |
|---|---|
| `mps2-an385/an386/an500` 的 `-kernel` 直接装裸 bin 到 0x0，SP@0/PC@4 | 三目标 rc=0 |
| **`mps2-an505`(M33) 复位向量在 `0x10000000`**，`-kernel` 不落在那里 → 必须 `-device loader,file=…,addr=0x10000000,force-raw=on`，且镜像要**链接到 0x10000000** | 0x0 链接+`-kernel` → `Lockup: can't escalate 3 to HardFault`（PC 为垃圾值）；0x10000000 链接+loader → rc=0 |
| AN505 上 RAM `0x20000000` 可用 | 同一镜像（RAM 未改）跑通 |
| RISC-V 无 semihosting：用 SiFive finisher（`0x100000` 写 `0x5555`），`0x80000000` + `-mcmodel=medany` | rv32/rv64 rc=0 |
| semihosting 的 `SYS_EXIT`(0x18 / 0x20026) 在 M3/M4/M7/M33 全可用；`printf` 的 `_write` 在没有控制台时会**阻塞** | rdimon-init-only rc=0；带 printf 挂起 |
| 失败码要用 **`SYS_EXIT_EXTENDED`(0x20)**：两字块 `{0x20026, code}`，QEMU 以 code 为退出码；朴素 SYS_EXIT 只能表达 0 | 篡改期望字节后 M7 exit=11 |
| **AN505 的内存图**：`0x10000000` 只有 **32 KB**（armsse.sram0，复位向量所在），4 MB ssram-0 在 **`0x00400000`**；`0x20000000` 起只是 sram0 的别名窗口 | `info mtree` + 栈放 0x20400000 → **Data Abort → Lockup**；改到 0x00400000 后 rc=0 |
| **RISC-V `-Ttext` 不可靠**：QEMU virt 跳到了 text 段首字节而不是 `e_entry`（`-d in_asm` 实证：入口落在 `kat_k0_run` 而非 `_start`） | 改用 `kat_rv.ld` 把 `.text.start` 钉在 0x80000000，入口 == 镜像起点 |
| ARM 裸机链接：`-nostartfiles --specs=nosys.specs`（rdimon 的 `_write` 会阻塞；nosys 的桩只告警不阻断） | 四块 M 板全部链接并跑到 SYS_EXIT |

⇒ runner 的 `load` 字段因此有三种值：`raw`（`-kernel` 裸 bin）、`loader`（`-device loader` + `load_addr`）、`elf`（RV）。
