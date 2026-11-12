# 11 · GRU Seq2Seq、Teacher Forcing 与 Padding

根据原笔记 [DL（lm）](https://colab.research.google.com/drive/1vkmD-aMaSCzxEe0WiPkAAoQF4zXBztAk) 整理。原文件保存在 `originals/`。

原文件名包含 lm，但主体是 GRU 编码器—解码器和 padding loss，不是完整语言模型。
原笔记只有组件；整理版新增一个极小符号反转任务，把训练、teacher forcing 和自回归推理连起来。
训练时 decoder 输入为 BOS 加右移目标；推理时把上一步预测作为下一步输入，遇到 EOS 停止。
这里使用纯 PyTorch，去掉原笔记对旧版 d2l 的安装依赖。为缩短演示，源序列都为固定长度 2；
扩展到变长源序列时还需使用 pack_padded_sequence 或取每条序列真正结束位置的状态。
四个训练样本只能演示记忆和流程，不能证明翻译质量；真实语料、验证集和 BLEU 尚未包含。

```python
import torch
from torch import nn
import torch.nn.functional as F
torch.manual_seed(42)
torch.set_num_threads(1)
PAD,BOS,EOS=0,1,2
class Encoder(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding=nn.Embedding(8,8,padding_idx=PAD)
        self.rnn=nn.GRU(8,16,batch_first=True)
    def forward(self,x): return self.rnn(self.embedding(x))
class Decoder(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding=nn.Embedding(8,8,padding_idx=PAD)
        self.rnn=nn.GRU(8+16,16,batch_first=True); self.output=nn.Linear(16,8)
    def forward(self,x,hidden,context):
        # context 始终保留编码器输出；逐 token 推理时不能把它换成更新后的 hidden。
        context=context.unsqueeze(1).expand(-1,x.size(1),-1)
        output,hidden=self.rnn(torch.cat([self.embedding(x),context],dim=-1),hidden)
        return self.output(output),hidden
encoder,decoder=Encoder(),Decoder()
# 玩具任务：反转两个符号，非真实翻译。PAD=0, BOS=1, EOS=2, 普通符号=3..7。
src=torch.tensor([[3,4],[4,5],[5,3],[6,7]])
target=torch.tensor([[4,3,EOS,PAD],[5,4,EOS,PAD],[3,5,EOS,PAD],[7,6,EOS,PAD]])
dec_input=torch.cat([torch.full((4,1),BOS),target[:,:-1]],dim=1)
optimizer=torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=0.03)
for step in range(200):
    optimizer.zero_grad(set_to_none=True)
    _,hidden=encoder(src); context=hidden[-1]
    logits,_=decoder(dec_input,hidden,context)
    # 以有效 token 数为分母，PAD 不参与损失。区别于原笔记按总序列长度取 mean 的定义。
    loss=F.cross_entropy(logits.reshape(-1,8),target.reshape(-1),ignore_index=PAD)
    loss.backward(); optimizer.step()
encoder.eval(); decoder.eval()
with torch.no_grad():
    _,hidden=encoder(src[:1]); context=hidden[-1]
    token=torch.tensor([[BOS]]); generated=[]
    for _ in range(4):
        scores,hidden=decoder(token,hidden,context)
        token=scores[:,-1].argmax(-1,keepdim=True)
        generated.append(token.item())
        if token.item()==EOS: break
print('训练 loss:',loss.item(),'输入:',src[0].tolist(),'生成:',generated)
assert generated==[4,3,EOS]
# 单独验证 padding 的梯度为 0。
probe=torch.randn(1,4,8,requires_grad=True)
F.cross_entropy(probe.reshape(-1,8),target[:1].reshape(-1),ignore_index=PAD).backward()
assert probe.grad[0,3].abs().sum()==0
```

## 自测与练习

改变一个 PAD 位置的 logits，观察 loss 是否变化；再尝试一个未见过的符号组合，讨论过拟合。
