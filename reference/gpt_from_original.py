"""
完整的 GPT 语言模型定义，全部浓缩在这一个单文件中。
参考资料：
1) OpenAI 官方 GPT-2 TensorFlow 实现：
https://github.com/openai/gpt-2/blob/master/src/model.py
2) HuggingFace/Transformers 官方的 PyTorch 实现：
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# 1. 基础组件：LayerNorm (带可选偏置)
class LayerNorm(nn.Module):
    """
    自定义的层归一化，支持可选的偏置项 bias。
    因为标准 PyTorch 的 nn.LayerNorm 强制要求 bias=True，为了实现偏置开关，我们进行重写。
    """
    def __init__(self, ndim, bias):
        super().__init__()
        # 初始化权重为全 1，偏置为全 0（若配置中开启 bias）
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        # 实时调用底层的层归一化函数，ε（缩放平滑项）设为 1e-5
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# 2. 核心组件：CausalSelfAttention (带因果掩码的多头自注意力机制)
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 确保 Embedding 特征维度可以被注意力头数整除
        assert config.n_embd % config.n_head == 0

        # 优化：
        # 我们不单独定义 q、k、v 三个 Linear 层。而是合并成一个 3 倍大小的 Linear。
        # 一次性算出所有头的 Q, K, V，随后通过 split 进行切分，极大提升矩阵计算并行度。
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # 输出的投影变换层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # 正则化层
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # 检测当前系统的 PyTorch 是否大于 2.0，若支持则一律采用极速的 Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("警告: 正在使用慢速注意力。Flash Attention 需要 PyTorch 2.0 及以上版本支持。")
            # 如果不支持 Flash Attention，则退回到传统的下三角因果掩码 (tril) 机制
            # 建立一个 [block_size, block_size] 的下三角矩阵并注册为 Buffer（不参与反向传播梯度更新）
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # 获取输入张量的尺寸：B (批次大小), T (当前序列长度), C (特征维度，即 n_embd)
        B, T, C = x.size()

        # 1. 一键投射 Q, K, V：输入 x [B, T, C] ──> 投射为 3 倍大小 ──> 按维度 2 切分成三个 [B, T, C] 的张量
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # 2. 维度流转（高维多头化）：
        # 以 k 为例，原本为 [B, T, C]，我们将其重塑为 4D: [B, T, n_head, head_size] （其中 head_size = C // n_head）
        # 接着调用 .transpose(1, 2) 交换维度 1 和维度 2，将注意力头提前到维度 1。
        # 最终形状变为: (B, nh, T, hs)，这一步是为了方便各头在后台完全并行进行矩阵相乘
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, nh, T, hs]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, nh, T, hs]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, nh, T, hs]

        # 3. 运行注意力计算
        if self.flash:
            # 方案一（首选）：调用 CUDA 优化的 Flash Attention 内核，速度极快
            # is_causal=True 会自动在 GPU 寄存器级别上执行因果遮蔽（防止偷看未来）
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # 方案二（备用）：手动执行注意力计算
            # 矩阵相乘：[B, nh, T, hs] @ [B, nh, hs, T] ──> 得到注意力关系图 [B, nh, T, T]
            # 同时除以 sqrt(head_size) 的缩放因子，控制方差，防止 Softmax 饱和
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # 利用下三角掩码，将右上角（未来时刻）的特征强制替换为 -inf
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # 经过 Softmax 归一化，未来时刻分得 0% 的注意力
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # 加权求和：[B, nh, T, T] @ [B, nh, T, hs] ──> 融合后的特征 [B, nh, T, hs]
            y = att @ v

        # 4. 多头输出拼接还原（维度流转回 3D）
        # 先将 nh 和 T 轴换回来 ──> 此时内存不连续，调用 .contiguous() 强行将其在内存中物理对齐
        # ──> 最后调用 .view 融合成 3D 形状: [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出线性投射
        y = self.resid_dropout(self.c_proj(y))
        return y

# 3. 基础组件：MLP (多层感知机 / 前馈传播网络)
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 第一层全连接：将特征维度上投影 4 倍 (如 768 ──> 3072)
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # 激活函数：使用 GPT 标配的、比 ReLU 更平滑的 GELU 激活函数
        self.gelu    = nn.GELU()
        # 第二层全连接：将特征维度下投影回原始尺寸 (3072 ──> 768)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# 4. 核心层块：Block (单个完整的 Transformer 解码器块)
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 第一个层归一化（自注意力前的 LN）
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # 自注意力层
        self.attn = CausalSelfAttention(config)
        # 第二个层归一化（MLP 前的 LN）
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # 前馈网络
        self.mlp = MLP(config)

    def forward(self, x):
        # 经典的 Pre-LN 结构与残差相加连接：
        # 输入先过 LN1 ──> 算自注意力 ──> 与输入 x 进行短路相加 (Residual Connection)
        x = x + self.attn(self.ln_1(x))
        # 接着过 LN2 ──> 算多层感知机 ──> 与 x 进行短路相加
        x = x + self.mlp(self.ln_2(x))
        return x

# 5. 配置参数类：GPTConfig
@dataclass
class GPTConfig:
    block_size: int = 1024         # 上下文最大长度
    vocab_size: int = 50304        # 词表大小。这里相比 GPT-2 默认的 50257 进行了向上取整（64的倍数），以迎合显卡硬件对齐提速
    n_layer: int = 12              # 12 个 Block 堆叠
    n_head: int = 12               # 12 个自注意力头
    n_embd: int = 768              # 768 维特征通道数
    dropout: float = 0.0           # Dropout 率
    bias: bool = True              # 是否使用偏置项

# 6. 父类网络：GPT 完全体
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # 用 ModuleDict 建立字典式的子模型管理，便于清晰阅读和维护
        self.transformer = nn.ModuleDict(dict(
            # wte: Word Token Embedding (字符/词嵌入层)
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe: Word Position Embedding (绝对位置嵌入层)
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # 堆叠 n_layer 层的 Block 解码器
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # 最终的层归一化 (Final LayerNorm)
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # 最终的语言模型输出投影头 (Language Modeling Head)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 核心：权重共享（Weight Tying）
        # 将输入词嵌入 `wte` 的权重，直接共享绑定给最后的线性输出层 `lm_head`
        self.transformer.wte.weight = self.lm_head.weight

        # 递归初始化所有的网络层权重
        self.apply(self._init_weights)

        # 针对残差连接的最后一层线性投射进行了特殊的缩放初始化（防止层数加深后输出值发生膨胀，符合 GPT-2 论文规范）
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # 打印并汇报模型的总可学习参数量
        print("网络总参数量: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        返回模型的总参数量。
        如果 non_embedding=True（默认），则扣除位置编码矩阵 wpe 的参数量（因为它不随输入扩展，属于硬参数）。
        因为 wte 与最后的 lm_head 共享了权重，所以 wte 的参数仍会被计算在内。
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """ 初始化权重的规则 """
        if isinstance(module, nn.Linear):
            # 线性层权重初始化为均值为 0，标准差为 0.02 的正态分布
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 嵌入层初始化为标准差为 0.02 的正态分布
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        # 确保传入的序列长度不超过我们在配置里设定的 block_size 上限
        assert t <= self.config.block_size, f"无法处理长度为 {t} 的输入，模型上限仅支持 {self.config.block_size}"

        # 生成对应当前序列长度的时间位置索引，例如 [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device) # 形状: [t]

        # 前向传播流程：
        tok_emb = self.transformer.wte(idx) # 词特征嵌入 ──> [b, t, n_embd]
        pos_emb = self.transformer.wpe(pos) # 位置特征嵌入 ──> [t, n_embd]

        # 将词嵌入和位置嵌入进行相加融合，并随机 Dropout 丢弃部分连接
        x = self.transformer.drop(tok_emb + pos_emb)

        # 顺序通过 12 层 Block 块
        for block in self.transformer.h:
            x = block(x)

        # 经过最后一层 LayerNorm
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 场景 A：训练模式（输入了 Y_targets），我们需要计算 Loss
            logits = self.lm_head(x) # 形状: [b, t, vocab_size]
            # 计算交叉熵时，我们需要将 3D 的 logits 展平为 2D [b*t, vocab_size]，真实目标 Y 展平为 1D [b*t]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 场景 B：预测模式（推理/生成时，Y_targets 为 None）
            # 这是一个硬件计算优化点：
            # 在预测时，因为是自回归生成，我们只需要知道序列中【最后那一个字符】所预测出的下一个词
            # 因此，我们不需要去算前面 1 到 t-1 个字符的 lm_head 矩阵乘法，只取最后一列 [:, [-1], :] 来进行投影
            # 这能节省极大的矩阵计算开销和硬件显存占用。
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """
        支持“模型剪裁”，动态缩减模型的最大上下文长度
        例如：我们加载了官方 1024 上下文的 GPT-2 预训练权重，但我们只想用 256 的长度进行小任务微调
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # 截断位置编码矩阵的大小，只保留前 block_size 行
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                # 截断自注意力里的 tril 缓冲面板大小
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        一键从 Hugging Face 官方服务器下载并导入 OpenAI 原版的 GPT-2 预训练参数
        支持：'gpt2'（124M）、'gpt2-medium'（350M）、'gpt2-large'（774M）、'gpt2-xl'（1.5B）
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("正在从 HuggingFace 官方下载/读取原版 GPT-2 权重: %s" % model_type)

        # 依据加载的模型类型，自动对齐网络层数、多头数和 Embedding 通道维度
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257 # 官方预训练一律使用原版 50257 大小的词表
        config_args['block_size'] = 1024  # 上下文上限一律为 1024
        config_args['bias'] = True        # 官方权重包含 bias 偏置
        if 'dropout' in override_args:
            print(f"将 Dropout 率强制重写为: {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        # 1. 先用当前的配置参数，实例化一个“空的（随机初始化）”本地模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # 过滤并扔掉非可更新参数的 tril 掩码

        # 2. 载入 Hugging Face 官方封装好的原版 GPT2 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 3. 参数对齐与转置（Weight Transpose）
        # 官方的权重文件中，其线性全连接层使用的是名为 "Conv1D" 的特殊模块，
        # 在 PyTorch 中，这需要我们将加载的参数矩阵进行【转置 t()】，然后才能安全地灌入我们的 nn.Linear 中
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"参数 Key 数量不对齐: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 针对那些需要转置（从一维卷积到线性层）的特殊权重矩阵，先转置再拷贝
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 正常的参数，直接进行一键无损复制
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        高等级优化器配置规范（Weight Decay 分组过滤）：
        大模型训练的标配动作。我们不应该对网络里所有的参数一视同仁地施加正则化（Weight Decay）。
        1. 所有 2D 的权重矩阵（如 Linear 里的 w，Embedding 里的参数） -> 开启权重衰减
        2. 所有 1D 的偏置（bias）和归一化参数（LayerNorm 里的 scale 和 bias） -> 禁用权重衰减，避免欠拟合
        """
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # 巧妙依据张量维度 dim() 过滤分组
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"参与 Weight Decay 惩罚的参数权重总数: {len(decay_params)}, 共 {num_decay_params:} 个参数")
        print(f"免除 Weight Decay 惩罚的参数权重总数: {len(nodecay_params)}, 共 {num_nodecay_params:} 个参数")

        # 检测当前硬件是否支持 PyTorch 的 fused AdamW（融合优化器）
        # 融合优化器可以将数十步更新小算子融合在 GPU 内运行，极大提速
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"正在启用 Fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ 评估当前的 MFU 硬件利用效率 """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12 # A100 GPU 理论半精度算力峰值为 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        大模型自回归文本生成器
        - temperature (采样温度): 控制生成文本的创造力与多样性
        - top_k: 是否开启 top_k 截断，限制模型只在概率最高的前 k 个候选词里挑选
        """
        for _ in range(max_new_tokens):
            # 1. 确保输入长度。如果历史上下文变得比 block_size 还要长，则对左侧历史执行截断
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # 2. 前向传播：调用 forward 获得原始 logits 得分
            logits, _ = self(idx_cond)

            # 3. 提取最后一个时间步对应的预测 Logits，并除以温度系数进行缩放
            # - 温度极低 (如 0.2)：概率被拉大拉陡，模型生成结果变得高度固定、死板。
            # - 温度极高 (如 1.5)：概率平滑接近均等，模型极具创造力但可能胡言乱语。
            logits = logits[:, -1, :] / temperature

            # 4. 开启 Top-K 裁剪采样（剪掉概率极低、基本不通顺的胡言乱语词）
            if top_k is not None:
                # 提取概率最高的前 top_k 个词的分数，其余部分强制覆盖为 -inf（指数后概率为0）
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 5. 用 Softmax 将 Logits 转化为 0~1 的概率
            probs = F.softmax(logits, dim=-1)

            # 6. 多项式随机概率抽样 (Biased Die)
            idx_next = torch.multinomial(probs, num_samples=1)

            # 7. 拼接：将新生成的词拼接回原序列的后面
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
