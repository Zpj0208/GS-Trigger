# GS-Trigger 复现代码与数据

本仓库包含训练代码、实际配置、结果数据、数据集索引、绘图脚本、依赖说明及原始许可文件，不包含论文正文、补充文档、投稿信或正式图稿。

## 快速校验

在仓库根目录依次运行：

```text
python scripts/check_inputs.py
python scripts/recompute_results.py
python scripts/verify_gtsrb.py
python figures/scripts/check_plot_inputs.py
```

以上命令只校验已有记录和重算统计，不需要GPU。主实验包含22次训练、每次100轮；RNG隔离控制有9次不同训练，其中Free的3次运行在两项对照中复用。

训练入口与命令见英文README。完整图像数据集和模型权重不在仓库内；数据下载与GTSRB子集准备见 `data_manifests/README.md`。绘图入口见 `figures/README.md`，可重绘11张数据图。训练与绘图依赖应放在不同环境中。

## 上传到GitHub

1. 解压后，将本文件所在目录的全部内容作为仓库根目录；首页应能直接看到 `README.md`、`code/`、`configs/`、`results/` 和 `figures/`。
2. 可以用GitHub Desktop提交整个目录。网页上传需要分批，每批最多100个文件，单文件上限25 MiB。
3. 不要只上传ZIP本身并期待GitHub自动展开，也不需要上传论文Word或投稿材料。

上游许可与原始数据来源说明予以保留。原始测量值未因发布整理而改写；新的训练运行需按记录配置自行执行。
