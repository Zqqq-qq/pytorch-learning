# 12 · 从历史均值到因果自注意力

根据原笔记 [self attention](https://colab.research.google.com/drive/1mOYA9QNLTRnAxPj0Y4PVn4B42szQzm-c) 整理。原文件保存在 `originals/`。

原笔记从双循环历史均值，逐步引入下三角矩阵、Softmax、Q/K/V 和缩放，本节保留这条推导线。
注意力公式：`softmax(Q @ Kᵀ / sqrt(d_k) + mask) @ V`。
未允许的位置在 Softmax 前设为负无穷；用 0 遮蔽会得到非零概率。
全零合法分数配合因果掩码可重现历史均值；可学习的 Q/K/V 则产生内容相关的权重。
原笔记称存在 SDPA API 就“一律 Flash Attention”，需修正：实际后端由设备、dtype 等条件选择。

```python
import torch
from torch import nn
import torch.nn.functional as F
torch.manual_seed(42)
B,T,C=2,5,8
x=torch.randn(B,T,C)
expected=torch.stack([x[:,:t+1].mean(1) for t in range(T)],dim=1)
mask=torch.ones(T,T,dtype=torch.bool).tril()
weights=torch.zeros(T,T).masked_fill(~mask,float('-inf')).softmax(-1)
torch.testing.assert_close(weights@x,expected)
query,key,value=[nn.Linear(C,4,bias=False) for _ in range(3)]
def attention(x):
    q,k,v=query(x),key(x),value(x)
    score=q@k.transpose(-2,-1)/(q.size(-1)**0.5)
    weights=score.masked_fill(~mask,float('-inf')).softmax(-1)
    return weights@v, weights
out,weights=attention(x)
assert out.shape==(B,T,4)
assert (weights.triu(diagonal=1)==0).all()
torch.testing.assert_close(weights.sum(-1),torch.ones(B,T))
altered=x.clone(); altered[:,3:]+=100
changed,_=attention(altered)
torch.testing.assert_close(out[:,:3],changed[:,:3])
q,k,v=query(x),key(x),value(x)
fast=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=True)
torch.testing.assert_close(out,fast,atol=1e-5,rtol=1e-5)
print('输出:',out.shape,'行权重和:',weights.sum(-1),'因果检查通过')
```

## 自测与练习

删除因果掩码，重新修改未来 token，先前位置的输出还相同吗？
