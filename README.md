# PyTorch 学习笔记：从数据管道到自注意力

根据 12 份个人 Colab 笔记整理的中文学习仓库。按前置知识排序，保留原文件，并用独立可运行的小例子补齐断点。
原笔记来源见 [文件清单](originals/manifest.json)，改动与缺口见 [整理说明](notes/整理说明.md)。

**推荐顺序：00–10 图像分类基础 → 11 GRU Seq2Seq → 12 因果自注意力 → reference 中的 GPT 结构。**

## 学习目录

| 章节 | 阅读 | 实践 |
|---|---|---|
| 00 · Tensor、维度与梯度 | [笔记](notes/00_tensor_shapes.md) | [Notebook](notebooks/00_tensor_shapes.ipynb) · [Colab](https://colab.research.google.com/github/Zqqq-qq/pytorch-learning/blob/study-notes-2026-11-12/notebooks/00_tensor_shapes.ipynb) |
| 01 · Dataset 与图像预处理 | [笔记](notes/01_dataset.md) | [Notebook](notebooks/01_dataset.ipynb) · [Colab](https://colab.research.google.com/github/Zqqq-qq/pytorch-learning/blob/study-notes-2026-11-12/notebooks/01_dataset.ipynb) |
| 02 · DataLoader 与批次 | [笔记](notes/02_dataloader.md) | [Notebook](notebooks/02_dataloader.ipynb) · [Colab](https://colab.research.google.com/github/Zqqq-qq/pytorch-learning/blob/study-notes-2026-11-12/notebooks/02_dataloader.ipynb) |
| 03 · nn.Module 与 forward | [笔记](notes/03_module.md) | [Notebook](notebooks/03_module.ipynb) · [Colab](https://colab.research.google.com/github/Zqqq-qq/pytorch-learning/blob/study-notes-2026-11-12/notebooks/03_module.ipynb) |
| 04 · nn.Linear 与 logits | [笔记](notes/04_linear.md) | [Notebook](notebooks/04_linear.ipynb) · [Colab](https://colab.research.google.com/github/Zqqq-qq/pytorch-learning/blob/study-notes-2026-11-12/notebooks/04_linear.ipynb) |
| 05 · ReLU 与 Sigmoid | [笔记](notes/05_activation.md) | [Notebook](notebooks/05_activation.ipynb) · [Colab](https://colab.research.google.com/github/Zqqq-qq/pytorch-learning/blob/study-notes-2026-11-12/notebooks/05_activation.ipynb) |
| 06 · CNN 与 Sequential | [笔记](notes/06_sequential.md) | [Notebook](notebooks/06_sequential.ipynb) · [Colab](https://colab.research.google.com/github/Zqqq-qq/pytorch-learning/blob/study-notes-2026-11-12/notebooks/06_sequential.ipynb) |
| 07 · 损失函数与反向传播 | [笔记](notes/07_loss_backward.md) | [Notebook](notebooks/07_loss_backward.ipynb) · [Colab](https://colab.research.google.com/github/Zqqq-qq/pytorch-learning/blob/study-notes-2026-11-12/notebooks/07_loss_backward.ipynb) |
| 08 · 优化器与训练闭环 | [笔记](notes/08_optimizer.md) | [Notebook](notebooks/08_optimizer.ipynb) · [Colab](https://colab.research.google.com/github/Zqqq-qq/pytorch-learning/blob/study-notes-2026-11-12/notebooks/08_optimizer.ipynb) |
| 09 · 训练、评估、设备与记录 | [笔记](notes/09_training.md) | [Notebook](notebooks/09_training.ipynb) · [Colab](https://colab.research.google.com/github/Zqqq-qq/pytorch-learning/blob/study-notes-2026-11-12/notebooks/09_training.ipynb) |
| 10 · 保存、恢复与推理 | [笔记](notes/10_checkpoint.md) | [Notebook](notebooks/10_checkpoint.ipynb) · [Colab](https://colab.research.google.com/github/Zqqq-qq/pytorch-learning/blob/study-notes-2026-11-12/notebooks/10_checkpoint.ipynb) |
| 11 · GRU Seq2Seq、Teacher Forcing 与 Padding | [笔记](notes/11_seq2seq.md) | [Notebook](notebooks/11_seq2seq.ipynb) · [Colab](https://colab.research.google.com/github/Zqqq-qq/pytorch-learning/blob/study-notes-2026-11-12/notebooks/11_seq2seq.ipynb) |
| 12 · 从历史均值到因果自注意力 | [笔记](notes/12_attention.md) | [Notebook](notebooks/12_attention.ipynb) · [Colab](https://colab.research.google.com/github/Zqqq-qq/pytorch-learning/blob/study-notes-2026-11-12/notebooks/12_attention.ipynb) |

## 先跑起来

本地建议 Python 3.10+。从仓库根目录执行；Notebook 可以在任意工作目录独立跑，生成文件保存在该工作目录。
依赖版本与实际验证边界见 [验证记录](notes/验证记录.md)。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify.py
python scripts/train_cifar10.py --synthetic --epochs 1
```

Windows 激活虚拟环境使用 `.venv\Scripts\activate`。训练脚本默认使用真实 CIFAR-10；
上面的 --synthetic 专用于无网络流程自检，其准确率没有真实任务意义。

Colab：点击章节的 Colab 链接；在独立单元安装缺失依赖：

```python
%pip install torch torchvision pillow matplotlib
```

每章代码自包含，按顺序运行本章单元即可；不用先执行其他 Notebook。
在 Colab 跑仓库脚本时，先 clone 本仓库并切换到 study-notes-2026-11-12 分支及仓库目录，再安装 requirements。
Colab 临时运行时重置会删除本地生成的数据和模型，需自行下载保存或放到已挂载的 Drive。

## 完整图像分类实验

```bash
python scripts/train_cifar10.py --epochs 5
tensorboard --logdir runs
python scripts/train_cifar10.py --predict /path/to/image.jpg --checkpoint checkpoints/cifar10.pt
```

训练使用 CIFAR-10 训练集，评估使用测试集；包含设备选择、日志、平均 loss、准确率、权重和优化器保存、单图推理。
推理预处理与训练一致；普通高清图片与 CIFAR-10 存在分布差异，识别结果可能不准。
没有随仓库提供已训练的 CIFAR-10 权重，需先完成训练。

## 文件布局

- `notes/`：便于 GitHub 阅读和搜索的中文 Markdown，含自测。
- `notebooks/`：13 份整理版 Notebook；默认练习不下载外部数据。
- `originals/`：12 份未改动的 Colab 原文件，含原始输出和交互演示。
- `scripts/`：下载数据、完整图像训练/预测和验证入口。
- `data/`：数据来源说明、玩具序列样本；真实数据按需下载。
- `reference/`：原笔记提取的 GPT 实现，作为进阶阅读。

## 参考资料

[nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)、
[CIFAR10 数据接口](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)、
[保存和恢复模型](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)、
[SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。
原笔记含教学资料衍生代码；未替原作者授予新的开源许可，复用第三方内容请核对原始出处。
