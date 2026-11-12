# 08 · 优化器与训练闭环

根据原笔记 [Nn. Optimizer](https://colab.research.google.com/drive/13X1fKE8CwszC8wbcgbOTqhd3amjvWBGp) 整理。原文件保存在 `originals/`。

训练顺序：清梯度→前向→计算 loss→反向→更新参数。
原优化器笔记在 `train=False` 的测试集上更新参数，作为 API 练习可以理解，但不能据此报告独立测试效果。
本节用明确可学习的 y=2x+1 检查 loss 下降，正式图像训练见第 09 节。

```python
import torch
from torch import nn
torch.manual_seed(42)
x = torch.linspace(-1,1,32).reshape(-1,1); y = 2*x+1
model = nn.Linear(1,1)
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
loss_fn = nn.MSELoss()
before = loss_fn(model(x),y).item()
for _ in range(100):
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(x),y)
    loss.backward()
    optimizer.step()
after = loss_fn(model(x),y).item()
assert after < before*0.01
print('loss:',before,'->',after,'w:',model.weight.item(),'b:',model.bias.item())
```

## 自测与练习

去掉 zero_grad 后会发生什么？实验前先重新创建模型和优化器。
