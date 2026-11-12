# 09 · 训练、评估、设备与记录

根据原笔记 [model_train](https://colab.research.google.com/drive/1UoR3KC9pIfJE9YQIJlZ4-l46fqK28VT-) 整理。原文件保存在 `originals/`。

原笔记把 GPU 片段、重复训练、推理和日志混在同一文件。整理后先用离线二维分类完成整个闭环。
训练集更新权重，评估集只测量。`eval()` 控制模块行为，`no_grad()` 控制梯度记录，作用不同。
loss 必须按样本数加权汇总；不能把不同大小 batch 的均值直接相加当作全体均值。
真正 CIFAR-10 训练、checkpoint、TensorBoard、独立图片预测已整合到 `scripts/train_cifar10.py`。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(160,2); y = (x[:,0]+x[:,1]>0).long()
train = DataLoader(TensorDataset(x[:128],y[:128]),batch_size=32,shuffle=True)
test = DataLoader(TensorDataset(x[128:],y[128:]),batch_size=20,shuffle=False)
model = nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Linear(16,2)).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.03)
criterion = nn.CrossEntropyLoss()
for epoch in range(15):
    model.train()
    for features, labels in train:
        features, labels = features.to(device),labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features),labels)
        loss.backward(); optimizer.step()
model.eval(); loss_sum=0.; correct=0; count=0
with torch.no_grad():
    for features,labels in test:
        features,labels=features.to(device),labels.to(device)
        logits=model(features)
        loss_sum += criterion(logits,labels).item()*len(labels)
        correct += (logits.argmax(1)==labels).sum().item()
        count += len(labels)
print('设备:',device,'评估 loss:',loss_sum/count,'准确率:',correct/count)
assert count == 32
```

## 自测与练习

为什么评估最后一批大小为 12 时，loss 应乘以 12 再累加？这组二维数据上的准确率能否代表 CIFAR-10 效果？
