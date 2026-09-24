# MSFE-RSVP 项目快照

这是只包含 Uyghur MSFE-RSVP 最终路线的独立复现实验仓库。

## 模型边界

`[B,3,32,128] → PatchEmbed → ViT Block 1–6 → MSFE → Block 7–12 → LayerNorm → RSVP → CTC`。
最终 Encoder 从原实验代码原样复制，以保持 state_dict key、初始化顺序和已有
checkpoint 的严格兼容性。

## 目录职责

| 目录 | 职责 |
| --- | --- |
| `configs/` | 唯一最终训练、评估配置 |
| `openrec/modeling/` | ViT/MSFE/RSVP、识别器和 CTC |
| `openrec/preprocess/` | 图片解码、增强、标签编码和尺寸变换 |
| `openrec/losses/`、`postprocess/`、`metrics/` | CTC 损失、解码和识别指标 |
| `openrec/optimizer/` | AdamW 和 OneCycleLR |
| `tools/` | 训练、评估、推理和运行工具 |
| `tests/` | 配置、组件、前向、CTC、state_dict 和 checkpoint 验证 |

## 删除原则

不包含检测、文档解析、RDBAFormer 及历史消融实现、其他配置、数据、权重、
输出和分析结果。动态注册表只声明实际存在的最终组件。

## 验证

```bash
python tools/validate_repository.py
python -m compileall .
```
