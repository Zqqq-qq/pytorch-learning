# 02 · DataLoader 与批次

根据原笔记 [DataLoader](https://colab.research.google.com/drive/1XzTDI7B8izQh2_KTftucXK01zS6ujp1b) 整理。原文件保存在 `originals/`。

`DataLoader` 把单个样本组成 batch。训练通常 `shuffle=True`，评估通常 `shuffle=False`。
默认保留最后不足一个 batch 的样本；原笔记的 `drop_last=True` 可用于观察固定批次，但评估时会漏样本。
本节用已知整数数据，让最后一批大小可直接验证，完全不依赖 CIFAR-10 下载。

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(torch.arange(10).float().reshape(10,1), torch.arange(10))
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
sizes = []
for x, y in loader:
    sizes.append(len(y))
    print('x:', x.shape, 'y:', y.tolist())
assert sizes == [4,4,2]
dropped = DataLoader(dataset, batch_size=4, drop_last=True)
assert sum(len(y) for _,y in dropped) == 8
```

## 自测与练习

若有 10000 个样本、batch size 为 64，保留最后一批时有多少批？最后一批多大？
