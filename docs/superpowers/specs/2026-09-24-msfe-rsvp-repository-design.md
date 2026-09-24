# MSFE-RSVP 精简复现实验仓库设计

## 目标

从当前 OpenOCR 工作区提取一个可独立训练、评估和推理的 MSFE-RSVP
维吾尔文识别仓库，最终放置在 `/home/ghzhang/Code/msfe_rsvp`。仓库沿用
现有 `configs/`、`openrec/`、`tools/` 目录结构，只保留
`configs/rec/uyghur/msfe_rsvp.yml` 实际需要的代码。

## 最终模型边界

模型数据流固定为：

```text
RGB image [B, 3, 32, 128]
  -> PatchEmbed
  -> ViT Block 1-6
  -> MSFE
  -> ViT Block 7-12
  -> LayerNorm
  -> RSVP
  -> projection
  -> CTC classifier
```

`openrec/modeling/encoders/vit_learnable_fusion_rsvp_clean.py` 已经移除
RDBAFormer、DFE、RCSM、ACF、IRF、Patch/Gate 等历史结构，因此原样复制，
不继续改写。这样可以保持现有 checkpoint 的全部参数名和严格加载兼容性。

## 保留范围

- 最终配置 `configs/rec/uyghur/msfe_rsvp.yml`。
- `ViTLearnableFusionRSVP`、ViT 公共层、基础识别器和 CTC Decoder。
- SimpleDataSet 及最终配置使用的 DecodeImagePIL、PARSeqAugPIL、
  CTCLabelEncode、RecTVResize、KeepKeys。
- CTCLoss、CTCLabelDecode、RecMetric、AdamW 和 OneCycleLR。
- 训练、评估、单图推理入口及其日志、配置、checkpoint 工具。
- Apache-2.0 LICENSE、依赖文件、README、`.gitignore` 和 `project.md`。
- 最小测试：配置解析、模型构建、前向维度、CTC loss、checkpoint 严格加载。

## 删除范围

- `opendet/` 和所有检测、端到端 OCR、文档解析代码。
- 其他识别模型、配置、损失、解码器和数据集。
- RDBAFormer 及其历史实验、消融配置、调试与分析脚本。
- `analysis_results/`、`output/`、`prediction_results/`、`汇报/`。
- 本地数据、训练权重、日志、缓存和 IDE 文件。

## 配置与数据

最终模型结构、优化器、学习率、batch size、epoch、随机种子和数据比例保持
`msfe_rsvp.yml` 一致。公开仓库不能依赖机器专用 `/ghData`，因此只将数据与
输出路径改成相对路径：

```text
data/uyghur_dict.txt
data/merged_labels.txt
data/real_train-aug35.txt
data/real_eval.txt
output/msfe_rsvp/
```

数据本身不进入仓库。README 说明标注格式为每行
`relative/image/path<TAB>label`，并提供命令行 `-o` 覆盖路径的示例。

## 依赖裁剪方式

目录结构保持不变，但动态注册表只声明仓库实际包含的组件，防止 README
暗示支持不存在的模型。无法在不改变行为的情况下安全裁剪的公共文件按原样
复制。Encoder 文件必须原样复制。

## 错误处理

- 配置缺失、字典缺失、标注文件缺失时沿用现有入口的明确错误。
- checkpoint 恢复继续使用 `strict=True`，保证结构不匹配立即失败。
- README 明确区分 `checkpoints`（续训）与 `pretrained_model`（预训练加载）。
- `.gitignore` 排除数据、权重、输出、日志和 Python 缓存。

## 验证标准

1. 所有 Python 文件可编译。
2. 最终 YAML 可解析，注册项均能导入。
3. CPU 上构建模型，并验证 `[2, 3, 32, 128]` 输出为
   `[2, 32, vocab_size]`。
4. 随机标签可完成一次 CTC loss 前向与反向传播。
5. 当前 `/ghData/output/msfe_rsvp/best.pth` 可对精简模型执行
   `strict=True` 加载，且没有 missing/unexpected keys。
6. 训练、评估和推理入口的 `--help` 可运行。
7. 仓库中不存在数据、checkpoint、输出和历史实验文件。

## 不在范围内

- 不重新训练模型，不承诺复现已有精度数值。
- 不上传 GitHub，不创建远程仓库。
- 不修改当前 `/home/ghzhang/Code/OpenOCR-main` 工作区。
