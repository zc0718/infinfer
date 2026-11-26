# KICKOFF · `infinfer` 启动手册（day-1 runbook）

> 读这一页就够开工。**计划正文在 `PLAN-2026-11-25.md`**（1000+ 行）；本文件只回答"第一天做什么、按什么顺序、期望看到什么"。
> 交接时把 `KICKOFF.md` + `PLAN-2026-11-25.md` 一起放进新仓的 `workspace/`，并在 `.github/copilot-instructions.md` 里指向它们（模板自带的入口文件，见 §4.6）。

---

## 0. 前置环境（缺一项就停下，别硬开）

| 项 | 要求 | 检查命令 |
|---|---|---|
| Conan | 2.x（模板是 Conan 2 配方） | `conan --version` |
| Python | ≥ 3.10（gates.py / scaffold_check.py 需要） | `python3 --version` |
| CMake | ≥ 3.28（`metadata.cmake_version` 会钉具体版本） | `cmake --version` |
| clang-format / clang-tidy | 模板 `gates.py format/tidy` 依赖 | `clang-format --version && clang-tidy --version` |
| 宿主 C/C++ 编译器 | GCC 或 Clang | `gcc --version` |
| QEMU | **可选**，只有 LOCAL_ONLY 的 `kat` 需要；无则可跳过该门禁 | `qemu-system-arm --version` |
| oracle venv | torch / onnx / onnxruntime / tensorflow（生成 golden 用） | `~/venv/build/bin/python -c "import torch,onnx,onnxruntime,tensorflow"` |
| 私有 remote（可选） | 厂商库配方分发用；没有时按 INV-6 退化为 scalar-only | `conan remote list` |

**网络**：首次 `conan create` 需要访问 Conan Center；若离线，先 `conan install` 预热 cache。

---

## 1. 建仓（P0.1 / P0.2）

```bash
# 1.1 取模板（二选一）
git clone <fcpp 仓> infinfer && cd infinfer && git checkout v0.3.11   # 或从 GitHub template 生成
rm -rf .git && git init -b main                                       # 新历史，不继承 fcpp

# 1.2 写身份（§4.1 的完整 JSON）+ 建契约骨架目录（§4.3–§4.5）
$EDITOR metadata.json
mkdir -p .github/project

# 1.3 先提交一次（关键：neutral 走 git grep，只扫【已跟踪】文件；
#     未提交时它会 vacuous 通过，是假绿）
git add -A && git commit -m "chore(:construction:): derive from fcpp 0.3.11"

# 1.4 再验四件事
python3 .github/misc/gates.py format        # 格式化基线（应绿）
python3 .github/misc/gates.py neutral       # 模板名/机器路径泄漏（应绿，且此时是【真】扫描）
python3 .github/misc/gates.py board         # gold-standard / on_board_validation 一致性（应绿）
python3 .github/misc/scaffold_check.py      # 模板件是否原样（应绿）
```

> `metadata.scaffold` 必须写成 `"0.3.11"`——`scaffold_check.py` 用它决定跟哪个模板版本对账。
>
> **`scaffold_check.py` 的基线来源**（`scaffold_check.py:88,96` 实测）：优先用离线目录
> `.github/project/scaffold-baseline/v0.3.11/`（其中须含 `scaffold-paths.txt`）；
> 没有则 `git clone --depth 1 --branch v0.3.11 https://github.com/HeT-FTI/fcpp.git`。
> → 私网/离线环境请先把模板树**拷进** `scaffold-baseline/v0.3.11/`，否则这一步会卡在网络上。

**目录约定（INV-11）**：仓库根只允许模板条目 + `include/ src/ test_package/ benchmark/ docs/ workspace/`。
新工具一律进 `workspace/`，**不要**新建 `tools/`。

---

## 2. 第一天必须验的 5 个 smoke

| # | 命令 | 期望 |
|---|---|---|
| S1 | `python3 .github/misc/gates.py all && python3 .github/misc/gates.py board && python3 .github/misc/gates.py neutral` | 5 条模板门禁全绿。**`all` 只含 format+tidy+guard**（`gates.py --help` 实测），board/neutral 必须单独跑 |
| S2 | `conan create . -s build_type=Release --build=missing` | 包能建出来（此时可能只有 placeholder TU，正常） |
| S3 | `git log --oneline` | 只有你自己的提交，**没有** fcpp 的历史 |
| S4 | `grep -rn "fcpp" --include="*.json" --include="*.yml" --include="*.py" . \| grep -v scaffold` | 只应命中 `metadata.scaffold` 与 `scaffold.json` 里的模板引用 |
| S5 | `ls .github/workflows` | 11 个 workflow 都在；**确认触发条件**（见 §6.1） |

---

## 3. 前 3 个 commit 的内容（定义清楚，别"先随便提交"）

| # | commit | 内容 | 验收 |
|---|---|---|---|
| C1 | `chore(:construction:): derive infinfer from fcpp 0.3.11` | 模板原样 + `metadata.json` 身份 + `.github/project/{scaffold.json,census.json,claim-basis.json}` 三个空骨架 + `.github/copilot-instructions.md` | S1–S5 全绿 |
| C2 | `feat(:sparkles:): kernel contract and matrix source` | `include/infinfer/kernel/kernel.h`（纯 C11 契约）+ `src/kernel/matrix.yml`（词表种子，§4.2）+ 生成器骨架（生成 census/parity 桩） | `gate/census` 能跑（允许红色，但必须能指名空格） |
| C3 | `test(:test_tube:): three failing contract cases` | P0.5 的三个**故意红**用例：融合激活派发、静态内存边界、quant/dequant 空实现 | `conan create` 后 `ctest` 显示 3 个 FAILED（在 PR 里贴出输出） |

---

## 4. 契约文件模板（agent 直接照抄，不要再发明一套）

### 4.1 `metadata.json`（`metadata.schema.json` 要求必填：`name/version/scaffold/build_type/dependencies/workflow_triggers`）

```json
{
  "name": "infinfer",
  "target": "auto",
  "version": "0.1.0",
  "scaffold": "0.3.11",
  "team": "HeT-FTI",
  "license": "Apache-2.0",
  "description": "A portable C11/C++17 compute-graph runtime for embedded inference and on-device adaptation",
  "authors": ["<owner> <email>"],
  "maintainers": ["<owner> <email>"],
  "topics": ["inference", "embedded", "tinyml", "quantization", "library"],
  "url": "https://github.com/HeT-FTI/infinfer.git",
  "homepage": "https://github.com/HeT-FTI",
  "cmake_version": "4.2.3",
  "build_cppstd": "17",
  "build_cstd": "11",
  "build_type": "Debug",
  "activate_code_coverage": true,
  "coverage": {
    "core_scopes": ["src/runtime/", "src/kernel/scalar/"],
    "floors": { "Linux": { "core": 90.0, "all": 85.0 } },
    "default_floor": { "core": 80.0, "all": 70.0 }
  },
  "is_shared": false,
  "is_header": false,
  "generate_modules_inplace": false,
  "std_modules": "iostream",
  "user_modules": "",
  "dependencies": {
    "common": {},
    "c": {},
    "cpp": { "etl": ["etl::etl"] },
    "infra": { "GTest": ["gtest::gtest"], "pybind11": ["pybind11::module"] }
  },
  "component_requires": { "infinfer_c": ["infinfer_cpp"] },
  "baremetal_white_list": ["etl"],
  "doc_languages": ["en", "zh"],
  "doc_versions": ["0.1.0"],
  "doc_doxygen_folders": ["include", "src", "docs/doxygen/dox"],
  "doc_doxygen_suffix": ["h", "c", "hpp", "cpp", "dox", "cxx"],
  "trigger_tests": true,
  "saving_tests_log": true,
  "enable_python_bindings": false,
  "on_board_validation": { "os": ["Linux"], "profile_arch": ["x86_64"], "toolchain_version": ["14.3.rel1"] },
  "workflow_triggers": {
    "build": true, "tests": true, "release": true, "docs": true,
    "security_scan": true, "cross_compile": true, "on_board": false
  }
}
```

> `on_board` 先置 `false`：没上板之前不要让它跑；P4 有了真板数据再打开。
> 厂商库**不进** `dependencies` 四桶——它们是 Conan option（`with_cmsis_nn` 等），见 PLAN §3.1.1。

### 4.2 `src/kernel/matrix.yml`（词表种子；INV-10 的唯一来源）

```yaml
# 一个 kernel = 名字 + 支持的 (dtype, quant) + 各 tier 的实现状态 + 反向是否可用
schema: 1
kernels:
  conv2d:
    tflite: [CONV_2D]
    onnx:   [{op: Conv, where: "group == 1"}]
    dtype_quant: [[fp32, none], [int8, tensor], [int8, channel]]
    tiers: { scalar: todo, autovec: todo, arm_neon: todo, arm_m: todo, rvv: todo }
    grad: no-grad
  dwconv2d:
    tflite: [DEPTHWISE_CONV_2D]
    onnx:   [{op: Conv, where: "group == in_channels"}]
    dtype_quant: [[fp32, none], [int8, tensor], [int8, channel]]
    tiers: { scalar: todo, autovec: todo, arm_neon: todo, arm_m: todo, rvv: todo }
    grad: no-grad
  relu:
    tflite: [RELU]
    onnx:   [{op: Relu}]
    dtype_quant: [[fp32, none], [int8, none]]
    tiers: { scalar: todo, autovec: todo, arm_neon: todo, arm_m: todo, rvv: todo }
    grad: todo
```

> 完整词表用 `workspace/_tmp/ops-map.json`（25 kernel ↔ 30 TFLite ↔ 27 ONNX）作为种子批量展开。
> **`where` 字段是必需的**：ONNX 的 `Conv` 同时代表 conv 与 dwconv（PLAN §3.7 规则 1）。

### 4.3 `.github/project/census.json`（`gate/census` 的输入）

```json
{
  "schema": 1,
  "cells": {
    "conv2d/int8/tensor/scalar": ["kernel_parity/conv2d_scalar_int8_tensor"],
    "conv2d/int8/tensor/arm_m":  [],
    "relu/fp32/none/scalar":     ["kernel_parity/relu_scalar_fp32"]
  }
}
```

> `[]` = 该格已声明但**没有用例** → `gate/census` 必须报红；键不存在 = 未声明 → 允许，但 `matrix.yml` 里声明了就必须有键。

### 4.4 `.github/project/claim-basis.json`（每个对外 ✅ 的依据）

```json
{
  "schema": 1,
  "claims": [
    {"id": "C-001", "claim": "int8 conv matches TFLite within 1 LSB",
     "kind": "numeric", "gate": "oracle", "evidence": "workspace/qa/results/oracle-<commit>.json",
     "commit": "<sha>", "host": "<hostname>", "date": "2026-11-25", "local_only": true}
  ]
}
```

### 4.5 `.github/project/scaffold.json`（与模板对账的声明）

```json
{
  "scaffold": "0.3.11",
  "checked_files": ["<由 scaffold_check.py 生成>"],
  "tolerated_forks": [
    {"path": "CMakeLists.txt", "reason": "scalar .c 编进 _cpp 目标（PLAN §3.4 一行 fork）"},
    {"path": "conanfile.py", "reason": "新增 with_* option 与条件 requires"},
    {"path": "test_package/CMakeLists.txt", "reason": "resources/ 与 unit/stress 目录接线"},
    {"path": "test_package/conanfile.py", "reason": "同上"}
  ],
  "replaced": [],
  "owned_dirs": ["include/", "src/", "workspace/"]
}
```

### 4.6 `.github/copilot-instructions.md`（agent 入口，模板没有，必须新建）

```markdown
# infinfer — agent 工作须知
- 先读 `workspace/KICKOFF.md`，再读 `workspace/PLAN-2026-11-25.md` §2（不变量）与 §8（执行规约）。
- 每个任务完成前必须：`python3 .github/misc/gates.py all` 绿 + 新门禁能演示变红 + `gaps.md` 增行。
- 硬禁止：include/ 下出现架构头或厂商头；kernel 签名出现 shape/stride；用 #define 裁剪算子；新增顶层目录。
```

---

## 5. 门禁现状：模板只有 5 条，不是 21 条

`fcpp` 的 `.github/misc/gates.py` 自带 **5 条**：`format / tidy / guard / board / neutral`（`gates.py --help` 可验），但
**`all` 的组成只有 `format + tidy + guard`**——board 与 neutral 不在 `all` 里，必须单独调用。PLAN 里的 21 条是**目标态**，扩展路径：

1. 新增项目门禁的实现放 **`.github/project/<name>_gate.py`**（照 foc）；
2. 在 `.github/misc/gates.py` 里加 `cmd_<name>` + argparse 子命令 + `all` 的聚合（这是 tolerated fork，登记进 `scaffold.json`）；
3. 每条必须带 `--selftest`，且能演示"红"。

**第一天不需要 21 条**：C1 阶段只要有 `format/tidy/guard/board/neutral`（模板自带）+ `rootlayout`（20 行，原型已验 V9）就够。

---

## 6. 已知卡点（都在真实项目里踩过）

### 6.0 `neutral` 的假绿
`cmd_neutral` 用 `git grep`（`gates.py:434`）只扫**已跟踪**文件。空仓 / 未 `git add` 时它必然 OK——以为"没有模板名泄漏"，实际是**什么都没扫**。规矩：先 commit，再跑 neutral；C1 之后再跑一次。

### 6.1 CI 静默不跑
模板的 `ci-build-test.yml` / `full-test-automation.yml` / `docs-build.yml` 用 **commit message 里的 emoji 前缀**当开关（`:building_construction:`、`:book:` 等）。
**第一件事**：打开 workflow 看触发条件；否则你以为在等 CI，其实它根本没启动。改法：要么按 emoji 提交，要么在 P0 就把触发条件改成 `on: [push, pull_request]`（登记 tolerated）。

### 6.2 `conan create` 拿不到私有 remote
厂商库走 `workspace/conan/recipes/` + 私有 remote。拿不到时**必须**能退化为 scalar-only（INV-6）：
`conan create . -o "&:with_cmsis_nn=False" -o "&:with_csi_nn2=False"`。


### 6.3 工具链没装
`arm-toolchain/11.3.rel1` 与 `rv-toolchain/14.3.rel1` 需要先 `conan create --build-require`（首次会下载几百 MB）。
**在 P0.6 之前，交叉编译相关门禁应显式 skip 并打印原因**，不要让它假绿。

### 6.4 `workspace/` 被 gitignore
PLAN 与 KICKOFF 都在 `workspace/` 里 → **不会进 git**。交接方式二选一：
- `git add -f workspace/PLAN-2026-11-25.md workspace/KICKOFF.md`（只强制加这两份文档）；
- 或把它们移到 `docs/dev/`（那就会进 docs 构建，注意 `gate/docvoice` 会把开发域词报出来 → 这类文档要标"内部"）。

### 6.5 `docs/build.py` 与 doxygen
模板的文档链默认能跑通（`docs-build.yml`）。不要在第一周动它；PLAN §9 已把文档定为 M5 之后。

---

## 7. 交接检查清单（leader 用；全勾上就"可以 launch"）

### 7.1 验收状态回填（复核日期 2026-11-25，执行 agent 实测）

| # | 检查项 | 状态 | 证据（可复跑） |
|---|---|---|---|
| 1 | 新仓已从 fcpp v0.3.11 派生，`metadata.scaffold = "0.3.11"` | ✅ | `metadata.scaffold=0.3.11`；`python3 .github/misc/scaffold_check.py` → `74 file(s) match v0.3.11` |
| 2 | `metadata.json` 按 §4.1 写好（`name=infinfer`、`build_cstd=11`、`component_requires` 有 `_c → _cpp`） | ✅ | 逐字段核对：`name=infinfer`、`build_cstd=11`、`build_cppstd=17`、`build_type=Debug`、`cmake_version=4.2.3`、`component_requires={"infinfer_c":["infinfer_cpp"]}`、`on_board_validation.workflow on_board=false`、`workflow_triggers` 与 §4.1 完全一致 |
| 3 | `.github/project/{scaffold,census,claim-basis}.json` 三个骨架已建 | ✅ | 三文件均在（另加 `.github/project/census_gate.py`，见 gaps G-10） |
| 4 | `.github/copilot-instructions.md` 已建并指向 KICKOFF/PLAN | ✅ | 文件前两行即指向两份文档 |
| 5 | 模板 5 条门禁全绿；`neutral` 无 fcpp 泄漏 | ✅ | 5 条全绿（`format/tidy/guard` + `board` 12 targets/1 citation + `neutral: OK`）。**注意 `neutral` 的作用域是 `NEUTRAL_SCOPE = ('.github/skills', '.github/misc/Codegen-Starter.txt', '.github/CONTRIBUTING.md')`**（`gates.py:322`），它管的是**指引文本**；配置/源码面的身份泄漏由 S4 与 C7 的 `scaffold_gate.py` 守 |
| 6 | CI 触发条件已确认（或已改） | ✅ | `ci-orchestrator.yml` 为唯一入口：`on: push[main] / pull_request[main] / workflow_dispatch`；`ci-build-test`、`cross-compile`、`full-test-automation`、`metadata-controller`、`security-linters`、`semver-release` 六条全是 `workflow_call` 由编排器调用；emoji 门控在 `metadata-controller` 内部 → **§6.1 不需要改触发条件**。（仓库尚未 push，CI 从未真跑过，此条为"读配置确认"） |
| 7 | `arm-toolchain/11.3.rel1`、`rv-toolchain/14.3.rel1` 已构建 | ✅ | `conan list`：`arm-toolchain/11.3.rel1`；`rv-toolchain/12.3.rel1` + `rv-toolchain/14.3.rel1`（比 KICKOFF 预期更早完成，见 `workspace/audit/toolchain-decision.md`） |
| 8 | oracle venv 可用（torch/onnx/onnxruntime/tensorflow） | ✅ | `~/venv/build/bin/python`：torch 2.12.0+cu130 / onnx 1.21.0 / onnxruntime 1.26.0 / tensorflow 2.21.0 |
| 9 | `workspace/referential_repos/` 与 `workspace/compliance/` 已随文档交接 | ✅ | 2026-11-25 从 `~/Projects/HeT-FTI/IE/workspace/` 拷入：`referential_repos` 633 MB / 6 仓（sha 见 `index.txt`，与 PLAN 记录一致）、`compliance` 40 KB |
| 10 | 第一个 agent 的任务已指定为 **C1** | ✅ | `git log`：`c3faa33`(C1) → `faa0a11`(C2) → `e0abb3c`(C3) → `5dd7c36`(C4) → `7e31729`(C5) → `b5a585c`(C6)，顺序与 §3 一致 |

**§0 前置环境实测**：Conan 2.25.2、Python 3.12.3、CMake **4.4.4**（metadata pin 写的是 4.2.3，构建实测兼容，pin 目前只是文档值）、clang-format 23.1.0（与 CI 同版本）、clang-tidy 23.1.3、QEMU 8.2.2、宿主 GCC 13.3.0。

**§2 五个 smoke 实测**：

| # | 命令 | 结果 |
|---|---|---|
| S1 | `gates.py all` + `board` + `neutral` | ✅ 全绿（tidy 需要 `EIGEN_INC`/`ETL_INC`，本机用 Conan 缓存路径） |
| S2 | `conan create . -s build_type=Debug --build=missing` | ✅ `rc=0`，`100% tests passed, 0 tests failed out of 11` |
| S3 | `git log --oneline` | ✅ 无 fcpp 上游历史；只有模板实例化留下的 `Initial commit` + `prepare the instance after template instantiation` 两个提交 |
| S4 | `grep -rn "fcpp" --include="*.json" --include="*.yml" --include="*.py" . \| grep -v scaffold` | ✅ **已清（C7）**：`HETAI_PLAN_ROOT` 改 `${{ vars.HETAI_PLAN_ROOT }}`、README badge 指向本仓。重扫剩下的**三类都不是残留**：① `gates.py` 自指（门禁本身拼写禁用词）；② `fcpp_vec_add_f32/sub_f32` 模板示例符号（P0.4 换素材消失）；③ **`supportable-board-matrix.yml` 的 12 处 `/tmp/fcpp_benchmark_*` = 运维服务器端写死的部署落点，属例外**（C7 曾误改，2026-11-26 已回退到模板原字节，见 gaps G-24/G-42） |
| S5 | `ls .github/workflows` | 🟡 **10 个**（KICKOFF 记的是 11），已记 gaps G-7 |

### 7.2 未闭合项（按优先级）

1. ~~**referential_repos / compliance 为空**~~ —— **已闭合（2026-11-25）**：633 MB / 6 仓拷入，sha 与 `index.txt` 一致（gaps G-23）。
2. ~~**身份残留（S4 红）**~~ —— **已闭合（C7）**：`remote_path` 前缀、`HETAI_PLAN_ROOT`、README badge 全改；重扫只剩 `gates.py` 自指（gaps G-24）。
3. ~~**`workspace/` 未入库**~~ —— **决策：保持 ignored**（用户 2026-11-25 拍板）。依据 foc 先例：`HeT-FTI/foc` 与 `zc0718/foc` 的 `git ls-files workspace` 都是 **0**，且 foc 已不再调模板的 `scaffold_check.py`，改用自有 `scaffold_gate.py` + `scaffold.json`（gaps G-25）。

### 7.4 架构决策（C7，需求来源：KICKOFF 是一次性手册）

模板的 "逐字节与上游一致" 契约与 "允许多个诚实分叉" 不可兼得：`scaffold_check.py` 的 `main()` 把 tolerated 校验结果和字节比对结果并进同一个问题列表且从不减掉 tolerated 路径 ⇒ 第一个分叉就永久红。做法改为**把契约变成数据**（foc 已验证的形态）：

- `.github/project/scaffold_gate.py`：G1 checked 字节 / G2 tolerated 棘轮（陈旧即红、必须带 why+exit）/ G3 表面文件不得漏登 / G4 预算不得上涨；`--pin` 从上游树机器生成 sha（不手写），`--selftest` 用**每条规则只让自己红**的 fixture 逐条自证。
- `.github/project/scaffold.json`：`checked` 74 条 sha256 + `tolerated` 3 条（编排器改调本门禁、board matrix 前缀、hetai 变量），预算 = 3。
- `ci-orchestrator.yml` 第一步改跑 `python3 .github/project/scaffold_gate.py`。
- `.github/project/**` 不在模板表面（`scaffold-paths.txt` 不含它）⇒ 项目门禁/数据放这里不会反过来制造新分叉。

### 7.3 原始清单（保留作对照）

- [x] 新仓已从 fcpp v0.3.11 派生，`metadata.scaffold = "0.3.11"`
- [x] `metadata.json` 按 §4.1 写好，`name=infinfer`，`build_cstd=11`，`component_requires` 有 `_c → _cpp`
- [x] `.github/project/{scaffold,census,claim-basis}.json` 三个骨架已建
- [x] `.github/copilot-instructions.md` 已建并指向 KICKOFF/PLAN
- [~] 模板 5 条门禁全绿；`neutral` 无 fcpp 泄漏（门禁绿；neutral 只覆盖指引文本，配置/源码面由 S4 + C7 守）
- [x] CI 触发条件已确认（或已改）
- [x] `arm-toolchain/11.3.rel1`、`rv-toolchain/14.3.rel1` 已构建（或明确记为 P0.6 未完成）
- [x] oracle venv 可用（`torch/onnx/onnxruntime/tensorflow`）
- [x] `workspace/referential_repos/` 与 `workspace/compliance/` 已随文档交接（2026-11-25 拷入）
- [x] 第一个 agent 的任务已指定为 **C1**（不是"把 P1 全部做完"）

