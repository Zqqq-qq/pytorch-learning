# 07 · 损失函数与反向传播

根据原笔记 [Nn.Loss and backward](https://colab.research.google.com/drive/1P9dAhi94bIy-5DE7e4EAGAyVuN7dtYgb) 整理。原文件保存在 `originals/`。

L1 是绝对误差均值，MSE 是平方误差均值。分类常用 CrossEntropyLoss：输入 `[B,K]` logits 和 `[B]` 的 long 类别索引。
此处讨论整数类别标签形式；CrossEntropyLoss 也支持合法的浮点类别概率目标。
不需要先给 logits 做 Softmax。`backward()` 会把梯度累积到 `.grad` 中，优化前通常先清零。

```python
import torch
from torch import nn
x = torch.tensor([1.,2.,3.]); y = torch.tensor([1.,2.,5.])
assert torch.allclose(nn.L1Loss()(x,y),torch.tensor(2/3))
assert torch.allclose(nn.MSELoss()(x,y),torch.tensor(4/3))
logits = torch.tensor([[0.1,0.2,0.3]],requires_grad=True)
target = torch.tensor([1],dtype=torch.long)
loss = nn.CrossEntropyLoss()(logits,target)
loss.backward()
assert logits.grad is not None and torch.isfinite(logits.grad).all()
print('loss:',loss.item(),'梯度:',logits.grad,'预测类别:',logits.argmax(1))
```

## 自测与练习

解释为什么 loss.backward() 后模型权重数值尚未改变。
