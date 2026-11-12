# 03 · nn.Module 与 forward

根据原笔记 [Container-nn.Module](https://colab.research.google.com/drive/1qKaDGbAHdV0HpqFr9GRjHjRE2Qk9oW4f) 整理。原文件保存在 `originals/`。

`__init__` 定义层，`forward` 定义计算；`model(x)` 经过模块调用机制并支持钩子。
原注释“直接调用 forward 会错过梯度流”不准确：直接调用通常仍被 autograd 跟踪，但会绕过模块调用钩子等逻辑。
这里用钩子直接观察差别。注册在模块上的 Parameter 会出现在 `parameters()` 和 `state_dict()` 中。

```python
import torch
from torch import nn
class AddOne(nn.Module):
    def forward(self, x): return x+1
model = AddOne()
calls = []
handle = model.register_forward_hook(lambda module, args, output: calls.append('hook'))
x = torch.tensor(1., requires_grad=True)
y = model(x)
y.backward()
assert x.grad.item() == 1 and calls == ['hook']
model.forward(x)  # 仅用于对照，日常使用 model(x)
assert calls == ['hook']
handle.remove()
print('输出:',y.item(),'梯度:',x.grad.item(),'钩子次数:',len(calls))
```

## 自测与练习

为什么模型参数应该放在 `__init__` 中，而不是每次 forward 都新建一层？
