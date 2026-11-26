# IE（`d03ebfc`）× PLAN 耦合分析（2026-11-25）

> 事实源：`workspace/referential_repos/IE`（用户已把它移入对拍源目录）+ `PLAN-2026-11-25.md` §1/§2.4/§4 P0.4。
> 目的：回答"IE 与本计划耦合最深的是哪一部分、以什么形式进入、重构幅度多大"。只读分析，不改 IE。

## 1. 规模（实测，排除 `build/`）

| 区域 | 文件 | 行数 | 性质 |
|---|---|---|---|
| `src/layers/` | 26 | 2864 | 26 个算子：conv2d、dw_conv2d、dense、max_pool2d、mean、pad(_v2)、reshape、transpose、strided_slice、resize_nearest、concat、add/sub/mul、relu(_6)、leaky_relu、hard_swish、sigmoid、tanh、softmax、quant/dequant、inputs、lstm |
| `src/backend/` | 4 | 368 | `ie_backend.c` + `cpu/ gpu/ npu/`（后两者是占位） |
| `src/ie_graph.c` + `ie_tensor.c` | 2 | ~614 | 图生命周期 + **后处理**（classification / yolov5 / yolo） |
| `include/` | 10 | 1836 | `ie_api.h`、`ie_def.h`、`api/*`、`private/{ie_struct,ie_tensor,ie_operator}.h`、`private/config/*` |
| `IE/SMT` | **0** | **0** | **子模块未初始化：SMT 源码不在本仓**（gaps G-29） |

## 2. 耦合最深的三处（这是回答的主体）

### ① 执行内存模型 —— 最深的一处

`ie_graph_compute(const IeGraph*, ieUT32 memorySize, void* pMemory)`：**单块外部内存 + LayerDesc 里按索引取 shape/quant**。
我们的 `infr_workspace`（bump 分配器）就是它的后代，且**已经落地并通过门禁**。
PLAN §2.3 的 `activation`（反向保活）与 `state`（持久）两条腿在 IE 里**完全不存在**——这是"扩展"，不是"迁移"。

- 进入模块：`include/infinfer/runtime/workspace.h` + `src/runtime/workspace.cpp`（已实现）；后续 `runtime/{graph,memory}.h`。
- 重构幅度：概念直接继承，代码已重写（绝对地址对齐、越界守卫、字节偏移）。**没有可直接编译搬过来的部分。**

### ② 张量描述符 —— 决定"能不能保持 INV-3"

`IeTensor_` 是 **32 位位域 + union**：
```c
struct IeTensor_ { ieUT32 dims:3; ieUT32 dType:5; ieUT32 len:24;
                   union { IeDim array[4]; struct{...} nhwc; nchw; ohwi; bsf; }; ... };
```
配合 `buf.pf32` 在 `src/` 里 **86 处**直接引用 ⇒ dtype 与 shape 被硬编码进每个层的循环。
INV-3（kernel 只认 `ptr + len + dtype + quant`）与 INV-4（quant 一等公民）**必须推翻这个结构**。

- 进入模块：`include/infinfer/runtime/tensor.h`（P2.x）+ `src/kernel/scalar/*`。
- 进入形式：**只保留设计思想**（union 形状视图 `nhwc/nchw/ohwi/bsf` 值得照抄命名与用法）；位域、`len:24`、`dType:5` 全部弃用。
- 重构幅度：**重写**（结构性）。

### ③ 派发表 + 层函数签名 —— L1 与 L2 被缝在同一个签名里

```c
typedef ieStaCode (*opFunc)(const IeBackend*, const IeLayerDesc*, const IeRuntimeCtx*);
static const opFunc __gOps[] = { ... };            /* src/backend/cpu/ie_cpu_backend.c:213 */
.pContext = &__gOps, .opsNum = sizeof(__gOps)/sizeof(__gOps[0]);
```
- **位置化数组** ⇒ 裁剪任意中间算子即错位（PLAN §1 的"裁剪即编译错误"）。
- **层函数签名里带 LayerDesc** ⇒ 数值内核看得见 shape，正是 INV-3 禁止的。
- 已处置：注册表 + `infr_kernel_find()` 已在 `src/kernel/contract.c` 落地（C2/C5），**这一处的替换已经完成**，IE 只剩反面教材价值。

- 进入模块：`src/kernel/contract.c`（已替换）、`src/runtime/graph`（层语义上移）。
- 重构幅度：**已完成替换**（无需再从 IE 搬）。

## 3. 逐区域映射（进哪个模块 / 形式 / 幅度）

| IE 区域 | 目标模块（PLAN §2.4） | 进入形式 | 重构幅度 |
|---|---|---|---|
| `IeLayerDesc` 索引式 blob（shapes/quant 表 + 索引） | `test_package/resources/kat/*.blob` 格式 + `runtime/graph` | **格式原型**：v2 blob 会采用"共享表 + 索引"，v1 已是张量表 + 名字 | 格式重写，思想保留 |
| 26 个 layer 的**算子语义**（shape 推导、边界、激活取值、pad 规则） | `src/runtime/graph`（L2）+ 调 `infr_*` | **语义素材 + 测试用例来源**；代码按新契约重写 | 大改（签名、张量、量化全变） |
| 层内数值循环（`buf.pf32` 86 处、直调 `smt_*`） | `src/kernel/scalar/*` | 循环体作参考；**主蓝本应是 SMT（当前缺失，G-29）** | 逐算子重写为 `(ptr,len,dtype,quant)` |
| `ie_graph.c` 生命周期 + **后处理**（classification/yolov5/yolo，~300 行） | `src/runtime/{graph,memory,postprocess}` | 算法级可搬（NMS、阈值、分类 argmax） | 中改（拆模块、接 workspace/state） |
| `ie_backend.c` + `cpu/gpu/npu/` | `src/backend/*`（P2.5 option 化） | 装配点作蓝图 | 中改（去位置化表、Conan option） |
| `ie_quant.c` / `ie_dequant.c`（空实现返回 OK） | `src/kernel/scalar/quant.c` | 只取"要做什么" | **已替换**（新写 + parity 用例） |
| `include/api/*`、`ie_def.h` | `include/infinfer/**` | API 形状参考 | 重写（`infr_` 前缀、C11、无位域） |
| `IE/SMT`（空）/ `workspace/materials/smt`（缺） | `src/kernel/scalar/` 主蓝本 | **缺失** | 需补交（G-29） |

## 4. 幅度结论（按行数）

| 类别 | 行数占比 | 处置 |
|---|---|---|
| 后端装配（368 行） | 12.6% | 概念搬、代码重写 |
| 26 个层的语义 + 数值（2864 行） | 74.4% | **语义/用例可复用，代码逐算子重写**；数值主蓝本是 SMT |
| 图生命周期 + 后处理（~614 行） | 16.0% | 后处理算法级可搬，图部分拆模块重写 |
| 公共头（1836 行） | — | 只作形状参考，全部重写 |

**一句话**：IE 与本计划的耦合不在"代码能否编译进新库"，而在**三个设计决定**——单块外部内存、索引式 layer/shape 表、以及"层函数看得见 shape"的签名。前两个被继承并扩展（workspace → +activation/state；blob v1 → v2 索引表），第三个已被注册表替换。

## 5. 未闭合

- **G-29**：`IE/SMT` 为空子模块，`workspace/materials/smt` 不存在 ⇒ `src/kernel/scalar/` 的数值主蓝本缺失。目前 `fc`/`relu`/`quant`/`activation` 是按 PLAN 语义新写的，其余 20+ 算子在补交 SMT 之前只能按 IE 调用点的注释推断语义。
- IE 的 `build/` 目录（10179 个 `.c`，含第三方）**不属于素材**，分析一律排除。
