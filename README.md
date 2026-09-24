# MSFE-RSVP for Uyghur Scene Text Recognition

这是论文最终 MSFE-RSVP 模型的精简复现实验仓库，基于 OpenOCR 的识别训练
调用链整理而来。仓库只保留一条成功路径：ViT、Block 6 后的多尺度特征增强
（MSFE）、残差静态垂直池化（RSVP）和 CTC。

## 模型

```text
RGB 32×128
  → PatchEmbed
  → ViT Blocks 1–6
  → MSFE
  → ViT Blocks 7–12
  → LayerNorm
  → RSVP
  → projection
  → CTC
```

最终配置为 `configs/rec/uyghur/msfe_rsvp.yml`。Encoder 文件保持实验代码原样，
以保证已有 checkpoint 的参数名和模块层级兼容。

## 安装

建议使用已经验证过的 PyTorch/CUDA 环境，然后安装：

```bash
pip install -r requirements.txt
```

## 数据准备

数据不会随仓库发布。默认目录为 `./data`，字典文件为
`data/uyghur_dict.txt`。每个标注文件每行一条记录：

```text
relative/image/path.jpg<TAB>ئۇيغۇرچە
```

路径相对于配置中的 `data_dir`；`<TAB>` 是真实制表符。准备以下文件：

```text
data/uyghur_dict.txt
data/merged_labels.txt
data/real_train-aug35.txt
data/real_eval.txt
```

## 训练、评估和推理

```bash
python tools/train_rec.py -c configs/rec/uyghur/msfe_rsvp.yml
python tools/eval_rec.py -c configs/rec/uyghur/msfe_rsvp.yml \
  -o Global.checkpoints=./output/msfe_rsvp/best.pth
python tools/infer_rec.py \
  -c configs/rec/uyghur/msfe_rsvp.yml \
  --checkpoint ./output/msfe_rsvp/best.pth \
  --image ./data/images/example.jpg
```

如需使用其他数据目录，按照当前入口的 `-o` 语法覆盖配置：

```bash
python tools/train_rec.py \
  -c configs/rec/uyghur/msfe_rsvp.yml \
  -o Global.character_dict_path=/path/to/uyghur_dict.txt \
     Train.dataset.data_dir=/path/to/data \
     Train.dataset.label_file_list='[/path/to/train1.txt,/path/to/train2.txt]' \
     Eval.dataset.data_dir=/path/to/data \
     Eval.dataset.label_file_list='[/path/to/eval.txt]'
```

## Checkpoint

`Global.pretrained_model` 只加载模型参数，适合预训练或微调；
`Global.checkpoints` 恢复模型、优化器、学习率调度器和训练进度，适合中断
训练续训。恢复 checkpoint 使用严格 state_dict 加载，结构不匹配会立即报错。

仓库不包含权重。已有最终权重可通过 `Global.checkpoints` 指向
`best.pth`；验证脚本会对当前实验权重执行 `strict=True` 检查。

## 目录

```text
configs/  最终训练配置
openrec/  识别模型、数据预处理、损失、解码、指标和优化器
tools/    训练、评估、推理入口及必要运行工具
tests/    配置、模型、CTC 和 checkpoint 回归测试
```

本仓库不包含检测、文档解析、RDBAFormer、历史消融配置、训练数据、权重和
实验输出。

## 许可证与致谢

代码遵循 Apache License 2.0，部分基础识别训练代码来自 OpenOCR；详见
`LICENSE`。MSFE 和 RSVP 的实现及实验配置保留其原始参数初始化和模块路径。
