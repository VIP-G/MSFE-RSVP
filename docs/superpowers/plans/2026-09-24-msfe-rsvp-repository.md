# MSFE-RSVP 精简仓库实现计划

> **面向 AI 代理的工作者：** 在隔离的 `/tmp/msfe_rsvp` 工作区执行；每个任务完成后运行指定验证并提交。

**目标：** 从 OpenOCR 提取只支持 Uyghur MSFE-RSVP 最终配置的可复现实验仓库，并验证原 checkpoint 严格兼容。

**架构：** 保持现有 `configs/`、`openrec/`、`tools/` 调用链。Encoder 文件原样复制；通过依赖闭包复制训练、评估、推理所需模块，精简动态注册表但不改变模型 state_dict。

**技术栈：** Python、PyTorch、PyYAML、NumPy、Pillow、OpenCV、RapidFuzz、tqdm。

---

### 任务 1：建立最小仓库骨架和元数据

**文件：**
- 创建：`README.md`、`requirements.txt`、`LICENSE`、`.gitignore`、`project.md`
- 创建：`data/README.md`
- 创建：`configs/rec/uyghur/msfe_rsvp.yml`

- [ ] **步骤 1：编写仓库验收测试清单**

建立 `tests/test_repository.py`，先断言目标文件和注册组件存在；此时应因目标文件不存在而失败。

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/ghzhang/miniconda3/envs/openocr312/bin/python -m pytest tests/test_repository.py -q`

预期：FAIL，报告目标配置或目录文件缺失。

- [ ] **步骤 3：复制配置并创建元数据**

复制最终配置，除机器专用路径外保持当前真实配置参数（seed=56）；将数据和输出路径改为 `./data`、`./output/msfe_rsvp`。README 只描述 MSFE-RSVP，包含数据格式、训练、评估、推理和 `pretrained_model`/`checkpoints` 区别。

- [ ] **步骤 4：运行测试验证骨架通过**

运行同一 pytest 命令，预期：目录和配置断言通过。

- [ ] **步骤 5：Commit**

`git add README.md requirements.txt LICENSE .gitignore project.md data configs tests && git commit -m "chore: scaffold minimal MSFE-RSVP repository"`

### 任务 2：复制并裁剪运行时注册组件

**文件：**
- 创建：`openrec/__init__.py`、`openrec/modeling/**`
- 创建：`openrec/preprocess/**`、`openrec/losses/**`、`openrec/postprocess/**`
- 创建：`openrec/metrics/**`、`openrec/optimizer/**`
- 创建：`tools/data/**`、`tools/engine/**`、`tools/utils/**`

- [ ] **步骤 1：编写组件构建测试**

在 `tests/test_components.py` 中解析 YAML，构建 `CTCLabelDecode`、`CTCLoss`、`RecMetric`、`ViTLearnableFusionRSVP` 和 `CTCDecoder`，并断言旧注册名不在精简注册表中；先运行确认因模块不存在而失败。

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/ghzhang/miniconda3/envs/openocr312/bin/python -m pytest tests/test_components.py -q`

预期：FAIL，报告精简模块尚未创建。

- [ ] **步骤 3：按依赖闭包复制代码**

保留最终 Encoder 原文件、`common.py`、基础识别器、CTC Decoder 及其直接公共依赖；保留最终数据变换、SimpleDataSet、CTC loss/postprocess、RecMetric、AdamW/OneCycleLR 和训练工具；删除未使用模型注册、检测目录、分析目录与历史配置。对于共享公共文件，只在删除无关分支不改变 import/行为时裁剪。

- [ ] **步骤 4：运行组件测试**

运行同一 pytest 命令，预期：全部通过，且 `ViTLearnableFusionRSVP` 的有效参数路径与源代码一致。

- [ ] **步骤 5：Commit**

`git add openrec tools && git commit -m "feat: add MSFE-RSVP runtime dependency closure"`

### 任务 3：加入模型、CTC 和 state_dict 回归测试

**文件：**
- 创建：`tests/test_model.py`、`tests/test_ctc.py`、`tests/test_checkpoint.py`

- [ ] **步骤 1：编写失败测试**

测试 CPU 前向 `[2,3,32,128] -> [2,32,vocab_size]`、合法随机标签的 CTC forward/backward，以及源模型和精简模型 state_dict key 顺序/shape 一致和最终 checkpoint `strict=True` 加载。

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/ghzhang/miniconda3/envs/openocr312/bin/python -m pytest tests/test_model.py tests/test_ctc.py tests/test_checkpoint.py -q`

预期：在精简模型未完成前失败。

- [ ] **步骤 3：完成最小实现调整**

仅修正精简仓库中的 import、注册表和路径；不得重写 Encoder、修改参数名、删除已注册的兼容模块或使用 `strict=False`。

- [ ] **步骤 4：运行模型测试**

运行同一 pytest 命令，预期：前向、反向、key/shape 对齐和 checkpoint 严格加载全部通过。

- [ ] **步骤 5：Commit**

`git add tests && git commit -m "test: verify model and checkpoint compatibility"`

### 任务 4：入口和全仓库验收

**文件：**
- 修改：`README.md`、`project.md`（同步实际最终目录）
- 创建：`tools/validate_repository.py`

- [ ] **步骤 1：添加验证脚本**

脚本依次执行 Python 编译、YAML/组件构建、CPU 前向、CTC backward、checkpoint strict load、入口 import，并检查仓库不含 `*.pth`、`*.pt`、`*.ckpt`、`*.log`、`output/` 内容、历史实验目录和机器绝对路径。

- [ ] **步骤 2：运行完整验证**

运行：`/home/ghzhang/miniconda3/envs/openocr312/bin/python tools/validate_repository.py`

预期：每项打印 `[PASS]`，退出码为 0。

- [ ] **步骤 3：验证 CLI 入口**

运行：`/home/ghzhang/miniconda3/envs/openocr312/bin/python tools/train_rec.py --help`、`... tools/eval_rec.py --help`、`... tools/infer_rec.py --help`；入口至少可 import 和解析参数。

- [ ] **步骤 4：检查源工作区未改动**

运行：`git -C /home/ghzhang/Code/OpenOCR-main status --short`，确认只有任务开始前已有的 `result.csv` 修改。

- [ ] **步骤 5：Commit**

`git add . && git commit -m "docs: finalize MSFE-RSVP reproducibility checks"`
