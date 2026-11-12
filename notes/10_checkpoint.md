# 10 · 保存、恢复与推理

根据原笔记 [model_save](https://colab.research.google.com/drive/1GSSVlNn9fl6JKidZuk3ZyR-5dwVas-Rz) 整理。原文件保存在 `originals/`。

推荐保存 `state_dict`，恢复时先创建相同网络结构，再加载参数。
原笔记用 VGG16 演示保存，会产生很大的文件；这里用小模型完整验证恢复前后的输出一致。
`map_location` 处理设备映射，`weights_only=True` 用于加载权重字典。原始整模型 pickle 示例只保存在归档中。

```python
from pathlib import Path
import torch
from torch import nn
torch.manual_seed(42)
def make_model(): return nn.Sequential(nn.Linear(3,4),nn.ReLU(),nn.Linear(4,2))
model=make_model().eval(); x=torch.randn(2,3)
path=Path('checkpoints/example.pt'); path.parent.mkdir(exist_ok=True)
torch.save(model.state_dict(),path)
restored=make_model()
restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
restored.eval()
with torch.no_grad():
    before,after=model(x),restored(x)
torch.testing.assert_close(before,after)
print('恢复一致:',torch.allclose(before,after),'预测:',after.argmax(1).tolist())
```

## 自测与练习

若要恢复训练而非只做预测，还应保存优化器状态、轮次及哪些随机状态？
