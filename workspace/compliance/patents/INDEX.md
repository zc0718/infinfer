# 专利检索台账（INDEX）

> **这不是法律意见，也不判断有效性/侵权/规避设计。**
> 本表**刻意留空**：任何专利号都必须来自实际检索并留存原文，不得凭记忆填写。
> 目标是"有据可查的技术地形图"，不是"规避方案"。

## 建议检索范围（CPC 分类）

| CPC | 覆盖 | 与本工程的关系 |
|---|---|---|
| `G06N 3/063` | 神经网络实现的物理手段（硬件/加速器） | 加速后端（delegate/NPU）相关 |
| `G06N 3/04` | 网络架构（含卷积、循环） | 我们支持的算子族 |
| `G06N 3/08` | 学习方法 | C3（在线适应）相关 |
| `G06F 17/16` | 矩阵/向量运算 | kernel 实现层 |
| `G06F 7/544` | 专用算术（乘加、饱和） | requantize / 定点乘加 |
| `G06T 3/40` | 图像缩放 | resize 算子 |
| `G06V 10/20~10/80` | 图像识别流程（含检测后处理） | NMS/后处理 |

## 建议检索式（在 CNIPA / USPTO / Espacenet 使用）

1. `(神经 or neural) and (量化 or quantiz*) and (推理 or inference)` 且分类号 `G06N3/063`
2. `depthwise convolution` / `深度可分离卷积` 且 `inference`
3. `requantization` / `重定标` 且 `integer` 且 `neural network`
4. `Winograd` 且 `convolution` 且 `neural network`
5. `non-maximum suppression` / `非极大值抑制` 且 `hardware`
6. 申请人：`Arm`、`Google`、`Qualcomm`、`NVIDIA`、`阿里巴巴/平头哥`、`Andes`、`SiFive`、`芯来`

## 台账（每行必须人工核验后填写）

| id | 标题 | 申请人 | 公开号 | 优先权日 | 法律状态 | 相关度(1-5) | 原文位置 | 复核人/日期 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| P-001 |  |  |  |  |  |  | `materials/P-001.pdf` |  |  |
| P-002 |  |  |  |  |  |  |  |  |  |

## 与工程的联动

- 若某条专利被认为**相关且未过期**，`metadata/standards.json` 增加一条 `kind=patent`，
  并在 `capabilities.rst` 的"法律与许可"小节列出（用中性措辞："存在相关专利族，见 compliance/patents/INDEX.md"）。
- `gate/compliance` 只检查**台账完整性**（字段非空、有原文、有复核人），**不检查法律结论**。
