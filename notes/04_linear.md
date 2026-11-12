# 04 · nn.Linear 与 logits

根据原笔记 [Nn.Linear](https://colab.research.google.com/drive/1_MU-wjX8cQd6sjSdUi3deWY4LxxL3y8Y) 整理。原文件保存在 `originals/`。

一张 `3×32×32` 图像展平后有 3072 个特征，线性层输出 10 个分类得分。
logits 是原始得分，不保证为正，也不要求和为 1。原始数据下载替换为形状相同的随机张量，专注维度。

```python
import torch
from torch import nn
torch.manual_seed(42)
model = nn.Sequential(nn.Flatten(start_dim=1),nn.Linear(3*32*32,10))
x = torch.randn(7,3,32,32)
logits = model(x)
assert logits.shape == (7,10)
assert model[1].weight.shape == (10,3072)
print('输入:',x.shape,'输出:',logits.shape,'权重:',model[1].weight.shape)
```

## 自测与练习

把 batch size 改成 1，权重形状是否改变？为什么最后一批不必强制凑齐 64 张？
