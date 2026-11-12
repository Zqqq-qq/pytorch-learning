# 数据清单与准备

| 数据 | 原笔记用途 | 仓库中的处理 |
|---|---|---|
| CIFAR-10 | DataLoader、线性层、激活、损失、优化、训练 | 用 torchvision 下载到 data/cifar10；不把整套原图提交到 Git |
| hymenoptera_data | 蚂蚁/蜜蜂 Dataset | 提供官方教程下载脚本，解压到 data/hymenoptera_data |
| Unsplash 示例图片 | 猫/狗/车等单图推理 | 原链接保留在原始 Notebook；整理版使用本地图片路径，避免下载依赖 |
| 训练产生的 .pth、日志 | 恢复权重、TensorBoard | 原 Notebook 引用了 tudui_10.pth，但实际权重文件未随链接提供；需重新训练 |
| toy_sequences.json | 新增 Seq2Seq 小练习 | 4 条符号反转样本，已经入库，无外部依赖 |
| generated_shapes | 新增 Dataset 离线练习 | 第 01 节生成 6 张简单几何图；不是真实分类基准 |

当前拿到的是 12 个 Notebook 文件；Notebook 中的图表输出不等于它引用的原始数据或 checkpoint。
未找到独立上传的私人数据文件。若之后提供 Drive 数据文件夹，可再接入当前目录规范。

从仓库根目录运行：

```bash
python scripts/prepare_data.py --dataset hymenoptera
python scripts/prepare_data.py --dataset cifar10
```

CIFAR-10 train=True 用于训练，train=False 用于最终评估。练习阶段不要反复根据测试集挑超参数；
正式实验应从训练集另划验证集。下载失败时先执行离线练习，不自动把测试集换成训练集。
数据来源及使用条件以各提供方为准：[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)、
[PyTorch 迁移学习教程](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)。
