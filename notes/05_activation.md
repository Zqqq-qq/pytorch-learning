# 05 · ReLU 与 Sigmoid

根据原笔记 [None-linear activation](https://colab.research.google.com/drive/17PgDYQAPVqg1gjhyOaqr_yTDr7jTH1QK) 整理。原文件保存在 `originals/`。

只有线性层的多层网络仍可合成为一个线性映射。激活函数引入非线性。
ReLU 将负数变为 0；Sigmoid 将输入映射到 (0,1)，大幅值区域容易梯度很小。
原笔记通过 CIFAR-10 和 TensorBoard 看图片，本节先用明确数值理解，再用图查看曲线。

```python
import torch
from torch import nn
import matplotlib.pyplot as plt
x = torch.tensor([-2.,-1.,0.,1.,2.],requires_grad=True)
print('ReLU:',nn.ReLU()(x).detach().tolist())
print('Sigmoid:',nn.Sigmoid()(x).detach().tolist())
nn.Sigmoid()(x).sum().backward()
print('Sigmoid 梯度:',x.grad.tolist())
axis = torch.linspace(-8,8,200)
plt.plot(axis,nn.ReLU()(axis),label='ReLU')
plt.plot(axis,nn.Sigmoid()(axis),label='Sigmoid')
plt.legend(); plt.grid(); plt.show()
```

## 自测与练习

在十分类输出层后先加 Sigmoid 再传给 CrossEntropyLoss 是否合适？回看第 07 节。
