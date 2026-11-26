# 标准清单（standards）

> 只收"会被代码、测试或对外宣称引用"的标准；无关的一律不收。
> 每条都必须有 `source_url` 与 `verified_at`；付费的标 `paywalled` 并进 `BLOCKERS.md`。

| # | 标准 / 规范 | 与本工程的关系 | 获取 | 状态 |
|---|---|---|---|---|
| S1 | **ISO/IEC 9899:2018（C17）** | 公开 API 的语言基线；`build_cstd` 的上位规范 | 付费（草案 N2176 公开） | `to-collect` |
| S2 | **IEEE 754-2019** | FP32 语义、NaN/Inf、舍入；"逐位一致"承诺的前提 | 付费 | `blocked` |
| S3 | **Arm Architecture Reference Manual, ARMv7-M（DDI0403）** | M3/M4/M7 的指令与异常语义；DSP 指令（SMLAD/SDOT） | 免费（需注册） | `to-collect` |
| S4 | **Armv8-M ARM（DDI0553）** | M33/M35P/M55/M85；Helium（MVE）与 TrustZone | 免费（需注册） | `to-collect` |
| S5 | **AAPCS（IHI0042）** | C ABI：参数传递、对齐、结构体布局；"冻结 C ABI"的依据 | 免费 | `to-collect` |
| S6 | **CMSIS-Core / CMSIS-NN 规范** | Cortex-M 软件接口与 int8/int16 算子约定 | 免费（Apache-2.0 仓库内） | `have`（`referential_repos/CMSIS-NN`） |
| S7 | **RISC-V ISA Manual（含 V 扩展 1.0）** | RV32/RV64 与 RVV 档的正确性依据 | 免费（GitHub `riscv/riscv-isa-manual`） | `to-collect` |
| S8 | **RISC-V psABI** | lp64d 调用约定、ABI 变体选择 | 免费 | `to-collect` |
| S9 | **ONNX IR 规范 + Operators 文档** | ONNX 子集的语义来源（opset 18 基线） | 免费（`onnx/onnx`） | `to-collect` |
| S10 | **TFLite FlatBuffer schema + 量化规范** | TFLite 输入格式与 int8/int16 量化语义 | 免费（Apache-2.0） | `have`（`referential_repos/tflite-micro` + pip 包） |
| S11 | **GGUF 格式规范** | C4 路径的模型容器（若接 llama.cpp） | 免费（`llama.cpp/docs`） | `have`（`referential_repos/llama.cpp`） |
| S12 | **MISRA C:2012** | 若要对汽车/工业客户宣称"安全编码" | 付费 | `blocked` |
| S13 | **ISO 26262:2018 / IEC 61508** | 功能安全等级（仅当产品要进车规/IEC61508 场景） | 付费 | `not-needed-yet` |
| S14 | **RFC 3629（UTF-8）** | 分词器/文本接口（C4） | 免费 | `to-collect` |

## 与 Sphinx 的关系

`docs/sphinx/source/compliance/` 下三页由 `metadata/standards.json` 生成：

- `standards.rst` —— 本表（按"已具备 / 待收集 / 阻塞"分组）
- `claims.rst` —— **每条"符合 X"的宣称 → 对应检查名 + 证据 id**（与 `.github/project/claim-basis.json` 打通）
- `tier.rst` —— 按目标场景（通用/工业/车规）列出"已做 / 未做"的差距

> 参照 foc 的做法：`foc/docs/sphinx/source/compliance/{index,industries,language,tier}.rst`。
