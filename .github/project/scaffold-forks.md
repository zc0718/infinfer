# Scaffold forks（模板拥有文件的分叉台账）

> 这份文档只登记**模板拥有的文件**（`.github/misc/scaffold-paths.txt` 覆盖面）上存在的分叉。
> 唯一事实源是 `.github/project/scaffold.json`：每个条目的 `why` / `exit` / `patch` 是**可执行数据**，
> `.github/project/scaffold_gate.py` 的 G1–G4 会逐条核对。本文件是给人读的**理由与验证记录**，
> 不重复字段值，也不构成第二份判据。
>
> `src/`、`include/`、`test_package/test/**`、`metadata.json`、`.github/project/**` 都是本工程自己的面，
> 在那里改动不是分叉，不进这张表（INV-10/INV-11 与 §2.4 的分区规则）。

## 记录格式

每条分叉写四段：

1. **改哪儿** —— 文件 + 一句话说明行为变化；
2. **为什么** —— 不改会怎样（要能指出**可复现的现象**，不是"更好看"）；
3. **怎么验证** —— 能重跑的命令与读到的数字；
4. **退出条件** —— 什么情况下这条分叉应当消失，以及消失时同步删哪几样东西。

补丁文件放在 `.github/project/patches/`，一律用 `diff -u` 生成、路径带 `a/` `b/` 前缀，
可用 `patch -p1` 直接打到上游原字节上。**判断补丁是否忠实的唯一方法**是打上去后与被修改的树
比 sha256——不是"看起来一样"。

---

## F2 · `conanfile.py`：编译期开关 `with_cmsis_nn`（2026-11-27）

**改哪儿.** 五处：`options`/`default_options` 增加 `with_cmsis_nn`；新增 `_with_cmsis_nn()`
（读 **option** 而不是 metadata，这样 `-o with_cmsis_nn=True` 能覆盖文件里的默认值）；
`requirements()` 关掉时跳过 `cmsis-nn`；`generate()` 把 `WITH_CMSIS_NN` 交给 toolchain；
`_preparing_deps_links()` 关掉时把 `cmsis-nn` 从链目标里摘掉。
补丁：`patches/conanfile-with-cmsis-nn-option.patch`（5 个 hunk）。

**为什么.** 模板把功能开关建模成 metadata 布尔值，但**没有任何机制让一个依赖变成条件的**：
`requirements()` 无条件 `self.requires(req)`。于是"可选厂商后端"只能二选一——要么永远依赖
CMSIS-NN（Cortex-M 之外的每个目标都被迫下载），要么永远不接。厂商后端又恰恰只对 MCU 有意义，
所以这个开关是它存在的前提。

**怎么验证.** ① 关（默认）：`conan create . -s build_type=Release` 不解析 cmsis-nn，
174 tests 绿；② 开：`conan create . -s build_type=Release -o with_cmsis_nn=True` 解析并链接
`cmsis-nn/8.0.0`，`src/backend/cmsis_nn.c` 用真实头文件编译，覆盖率 91.9%；
③ 补丁忠实性：`patch -p1` 打到上游 v0.3.11 原字节后 `diff -q` 与当前文件相同。

**退出条件.** 上游 v0.4+ 提供"由 metadata 驱动的条件依赖"机制后，删本条、删补丁、
删 `tolerance_budget` 里为它加的那一位。

## F3 · `CMakeLists.txt`：`INFR_WITH_CMSIS_NN` 跟随开关（2026-11-27）

**改哪儿.** `add_library(${LIB_NAME}_c ...)` 之后增加一段 `if(WITH_CMSIS_NN)` /
`target_compile_definitions(... PRIVATE INFR_WITH_CMSIS_NN)`。
补丁：`patches/CMakeLists-with-cmsis-nn-option.patch`（1 个 hunk）。

**为什么.** `src/backend/cmsis_nn.c` 整份内容都在 `#if defined(INFR_WITH_CMSIS_NN)` 里，
关闭时必须编译成**零符号**；这需要一个能跟着功能开关走的编译定义，而模板的钩子只覆盖
`ENABLE_COVERAGE` 这类固定变量，没有"跟随 metadata/option 的定义"的位置。
变量走 Conan toolchain（不是让 CMake 自己去读 metadata），否则 `-o` 覆盖到不了编译器。

**怎么验证.** 同 F2 的两条构建；另外 `python3 .github/project/vendor_gate.py` 断言关闭时
`nm` 里没有该后端符号、打开时有（V1/V2）。

**退出条件.** 与 F2 同批移除——上游提供编译选项钩子后，两条一起消失。

## F1 · `test_package/conanfile.py`：覆盖率收集按对象目录分槽（2026-11-26）

**改哪儿.** `_code_coverage_gcc()` 里把 `.gcda`/`.gcno` 复制进收集目录的那两行，从"按 basename
平铺"改成"每个对象目录一个槽位"；新增模块级 `_slot_name()`（对象目录的 sha256 前 16 位）。
补丁：`patches/test_package-conanfile-coverage-slots.patch`（43 行，`@@ -16,6 +16,20 @@` 与 `@@ -481,8 +495,14 @@` 两处）。

**为什么.** 模板的收集器把每个 `.gcda`/`.gcno` **按 basename** 复制到同一个目录：

```python
shutil.copy2(v1, coverage_folder + sep + _get_file_name(v1))   # basename only
```

于是**两个同名源文件互相覆盖，后处理者胜，另一个从报告里静默消失**——不报错、不告警，
只是那一格数字变少。这不是理论风险：

- `register.c` 在本仓有 **5** 份（`scalar/`、`arm_m/`、`arm_neon/`、`rvv/`、`x86/`）。分槽之前
  报告里只出现 `arm_neon/register.c`；`src/kernel/scalar/register.c` **长期没被算进 `core`**。
- 加 `x86` 档后同名的是六组（`add/fc/reduce/relu/scale/sum`），`core` 从 588 行掉到 491 行、
  `all` 从 91.81% 掉到 **90.87%**，击穿 91.00 地板——**一次纯粹的测量事故**，代码质量没有变化。

**怎么验证.** 三步，缺一不可：

1. **补丁忠实度**：把上游原字节放到同一相对路径、`patch -p1` 打上本补丁，与被修改的树比 sha256
   ——相等（`c073a0213d7e30ff…`）。
2. **同名源都在**：`conan create` 后 `coverage_summary.json` 的 `per_file` 里出现全部同名项——
   `register.c` **×5**、`fc/add/relu/scale/reduce` 各 **×2**（`scalar/` 与 `x86/`）。
   报告文件数 19 → **28**。
3. **判据仍在线上**：`bash .github/project/ci.sh` 的 `coverage` 两条地板都过——
   `all 91.77% (747/814)`、`core 92.26% (548/594)`，地板 91.00/91.00。

**退出条件.** 上游 `test_package/conanfile.py` 自己按对象目录分槽（或改用 lcov 的
`--rc geninfo_auto_base=1` + 不搬运的直采方式）之后：删补丁文件、删 `scaffold.json` 的这条
`tolerated`、把 `tolerance_budget` 减 1、本文件 F1 标注"已被上游吸收"，
并**顺手检查** `_slot_name()` 是否还有别的调用点（当前只有一个）。

**为什么它是"真分叉"而不是"调门禁".** 它改的是**测量工具的输入组织**，判据（地板数值、
`core_scopes` 前缀、由计数算率）一个字没动；修好之后 `core` 与 `all` 都是**更好**的数，不是更宽松的数。
反过来做（降地板让红变绿）在同一周已经被明确拒绝过一次（gaps G-55/G-62）。
