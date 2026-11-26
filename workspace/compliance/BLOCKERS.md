# 需要人工获取的材料（BLOCKERS）

> 这里只放**公开渠道拿不到或需要账号/采购**的材料。**不阻塞编码**，只阻塞"对外宣称符合”。
> 拿到后：把文件放到 `compliance/materials/<id>.<ext>`，把下表状态改为 `have`，并回填 `metadata/standards.json`。

| id | 材料 | 为什么需要 | 获取途径（建议） | 优先级 | 状态 |
|---|---|---|---|---|---|
| S1 | ISO/IEC 9899:2018 正式版 | 引用条款号时需要正式文本；草案 N2176 可先用 | ISO 官网 / 国家标准馆 | 中 | `to-collect` |
| S2 | IEEE 754-2019 | FP32 行为与"逐位一致"承诺的上位依据 | IEEE Xplore | 高 | `blocked` |
| S3 | ARMv7-M ARM（DDI0403E） | DSP 指令语义、对齐/异常行为 | Arm Developer（免费注册） | 高 | `to-collect` |
| S4 | Armv8-M ARM（DDI0553） | MVE/Helium 的 int8/int16 语义 | Arm Developer（免费注册） | 高 | `to-collect` |
| S5 | AAPCS（IHI0042） | C ABI 冻结的依据 | Arm Developer | 中 | `to-collect` |
| S7 | RISC-V ISA Manual（含 V 1.0） | RVV 档正确性 | GitHub 直接下载 PDF | 高 | `to-collect` |
| S8 | RISC-V psABI | 调用约定 | GitHub | 中 | `to-collect` |
| S9 | ONNX Operators 文档（离线快照） | 子集语义逐条引用 | 可直接从 `onnx/onnx` 生成 HTML | 中 | `to-collect` |
| S12 | MISRA C:2012（含 Amendment） | 若对外宣称安全编码 | MISRA 官网购买 | 低（按需） | `blocked` |
| P-* | 专利全文 PDF | 见 `patents/INDEX.md` 的检索台账 | 专利局公开库（CNIPA/USPTO/Espacenet） | 低（按需） | `to-search` |

## 规则

- **不要**把"没拿到的材料"当成已具备的依据；`gate/compliance` 会检查 `metadata/standards.json` 里 `status=have` 的条目是否真的有本地文件。
- 材料文件不放进 git（版权）；只把**元数据与引用的条款号**入库。
