# 00 · Tensor、维度与梯度

新增预备知识。

先学会读 shape，再读神经网络。图像通常是 `[B,C,H,W]`，文本特征通常是 `[B,T,C]`。
`flatten(start_dim=1)` 保留批次维度；`argmax(dim=1)` 对每个样本在类别维度选最大得分。
`backward()` 计算梯度；它本身不更新参数。

```python
import torch
torch.manual_seed(42)
images = torch.randn(4, 3, 32, 32)
print('图像:', images.shape, '展平:', images.flatten(1).shape)
logits = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
labels = torch.tensor([0, 1])
accuracy = (logits.argmax(dim=1) == labels).float().mean()
assert accuracy.item() == 0.5
x = torch.tensor(3.0, requires_grad=True)
(x*x).backward()
assert x.grad.item() == 6
print('准确率:', accuracy.item(), 'd(x²)/dx:', x.grad.item())
```

## 自测与练习

把 batch size 改为 7，观察展平后的第一维。为什么不直接 `flatten()`？
