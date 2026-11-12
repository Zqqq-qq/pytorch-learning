# 01 · Dataset 与图像预处理

根据原笔记 [Dataset](https://colab.research.google.com/drive/1tWBD-ksfM_nBeSGyfJ6BNBsR0DcwI-64) 整理。原文件保存在 `originals/`。

`Dataset` 描述“第 i 个样本是什么”，需要实现 `__len__` 和 `__getitem__`。
原笔记用蚂蚁/蜜蜂目录演示；这里保留相同目录读取思路，自动生成 6 张几何图作为离线练习数据。
新增：过滤图片后缀、固定排序、RGB 转换、整数类别标签。字符串标签适合展示，交叉熵训练需要整数索引。
这些几何图不是真实蚂蚁/蜜蜂，不用于评估图像识别效果。真实数据准备见 `data/README.md`。

```python
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor

root = Path('data/generated_shapes')
for name in ['circle', 'square']:
    (root/name).mkdir(parents=True, exist_ok=True)
    for i in range(3):
        image = Image.new('RGB', (40, 40), 'white')
        draw = ImageDraw.Draw(image)
        box = (5+i, 5+i, 30+i, 30+i)
        if name == 'circle': draw.ellipse(box, fill='red')
        else: draw.rectangle(box, fill='blue')
        image.save(root/name/f'{i}.png')

class FolderDataset(Dataset):
    def __init__(self, root):
        root = Path(root)
        self.classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        self.samples = [(p, label) for label, name in enumerate(self.classes)
                        for p in sorted((root/name).iterdir())
                        if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
        self.transform = Compose([Resize((32,32)), ToTensor()])
    def __len__(self): return len(self.samples)
    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            x = self.transform(image.convert('RGB'))
        return x, label

dataset = FolderDataset(root)
image, label = dataset[0]
assert image.shape == (3,32,32) and isinstance(label,int)
print('样本数:',len(dataset),'类别:',dataset.classes,'第一个标签:',label)
```

## 自测与练习

将 `root` 改为已下载的 `hymenoptera_data/train`。这个数据集返回单个样本还是一批样本？
