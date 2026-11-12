# 06 · CNN 与 Sequential

根据原笔记 [Sequential](https://colab.research.google.com/drive/1_8_5WqCQ1SOYUSHY04epmjfLW2HegOML) 整理。原文件保存在 `originals/`。

`Sequential` 按顺序调用每层。原模型的三次池化使尺寸从 32→16→8→4，所以展平后是 `64×4×4=1024`。
原笔记连续的两个 Linear 之间没有激活；整理版明确加入 ReLU，属于结构调整，旧 checkpoint 不能直接混用。
打印每层输出是检查尺寸错误最直接的方法。

```python
import torch
from torch import nn
model = nn.Sequential(
    nn.Conv2d(3,32,5,padding=2),nn.ReLU(),nn.MaxPool2d(2),
    nn.Conv2d(32,32,5,padding=2),nn.ReLU(),nn.MaxPool2d(2),
    nn.Conv2d(32,64,5,padding=2),nn.ReLU(),nn.MaxPool2d(2),
    nn.Flatten(),nn.Linear(1024,64),nn.ReLU(),nn.Linear(64,10))
x = torch.randn(2,3,32,32)
for layer in model:
    x = layer(x)
    print(type(layer).__name__,tuple(x.shape))
assert x.shape == (2,10)
```

## 自测与练习

把卷积 padding 改为 0，推导为何 Linear(1024,64) 会报维度错误。
