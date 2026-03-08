from transformers import PretrainedConfig

# huggingface的类
class SejuMindConfig(PretrainedConfig):
    model_type = "sejumind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
import math
from torch.nn import init
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

# 继承nn.Module的类
class RMSNorm(nn.Module):
# __init__初始化
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__() #调用父类的初始化方法
        self.eps = eps
        self.dim = dim

        # 可训练参数用来重新分配维度强度 使不同维度可以放大/缩小 -> 让模型仍然有表达能力。
        #
        self.weight = nn.Parameter(torch.ones(dim))#定义一个可训练的参数，初始值为全1，维度为dim 
# _norm
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * x
#forward
    def forward(self,x):
        return self.weight * self._norm(x.float()).type_as(x)


# RoPE
def precompute_freqs(dim: int, end: int=int(32*1024), rope_base:float=1e6, 
                         rope_scaling: Optional[dict]=None):
    
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim//2].float()/dim))
    attn_factor = 1.0

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4),  
            rope_scaling.get("beta_slow", 1),  
            rope_scaling.get("attention_factor", 1.0),
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b:(dim* math.log(orig_max / (b*2*math.pi))) / (
                 2 * math.log(rope_base)
            )
        
        # 计算高频区和低频区的维度切分点
        low, high = (
            max(math.floor(inv_dim(beta_fast)), 0),
            min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
        )

        # 计算混合因子 γ (Ramp)
        ramp = torch.clamp(
            (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )
         # 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
        freqs = freqs * (1 - ramp + ramp / factor)

    # Build position x frequency matrix: [end, dim//2]
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float() #[end, dim//2]

# 返回一个cos和sin
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    
    def rotate_half(x):
        return torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
    
    # apply RoPE
    q_embed = (q * cos.unsqueeze(unsqueeze_dim))+(rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim))+(rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    
    return q_embed, k_embed


def repeat_kv(x:torch.Tensor, n_rep:int):
    # x: [batch, seq_len, num_heads, head_dim]
    batch, seq_len, num_key_value_heads, head_dim = x.shape
    
    if n_rep == 1:
        return x
    
    return (
        x[:,:,:,None,:].expand(batch, seq_len, num_key_value_heads,n_rep,head_dim)
                        .reshape(batch, seq_len, num_key_value_heads * n_rep,head_dim) #复制重复n_rep次
    )

class Attention(nn.Module):
    def __init__(self, args:SejuMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_attention_heads if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        assert args.num_attention_heads % self.num_key_value_heads == 0 #确保num_attention_heads是num_key_value_heads的整数倍

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )
       
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )   


    def forward(
        self, 
        x:torch.Tensor,  
        position_embeddings: Tuple[torch.Tensor, torch.Tensor], 
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        batch_size, seq_len, _ = x.shape
        # 512, 512, 512
        xq = self.q_proj(x)  # [batch, seq_len, num_attention_heads * head_dim]
        xk = self.k_proj(x)  # [batch, seq_len, num_key
        xv = self.v_proj(x)  # [batch, seq_len, num_key_value_heads * head_dim]

        # 将qkv分成多个头
        xq = xq.view(batch_size, seq_len, self.n_local_heads, self.head_dim)  # [batch, seq_len, num_attention_heads, head_dim]
        xk = xk.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)

  
    # 计算 Q K 的RoPE
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
    # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)  # [batch, seq_len_cache + seq_len, num_key_value_heads, head_dim]
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_k_v = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

    # 计算注意力
        if (self.flash 
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq,xk,xv,
                dropout_p= self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            #(xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, num_attention_heads, seq_len, seq_len_cache + seq_len]
            k_len = xk.size(-2)
            diagonal = k_len - seq_len + 1
            causal_mask = torch.triu(
                torch.full(
                    (seq_len, k_len),
                    float("-inf"),
                    device=scores.device,
                    dtype=scores.dtype,
                ),
                diagonal=diagonal,
            ).unsqueeze(0).unsqueeze(0)
            scores = scores + causal_mask  # causal mask

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask.to(scores.dtype)) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores ) # 架构中添加dropout层以防止过拟合
            output = scores @ xv  # [batch, num_attention_heads, seq_len, head_dim]

        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_k_v
    

# silu 激活函数 引入非线性函数的意义 ： 
# 没有非线性函数，无论堆叠多少层，整个网络都只能表示一层线性变换，无法捕捉复杂的模式和关系。（多层等价于一层）
# 引入非线性函数后，网络可以学习和表示更复杂的函数，从而提高模型的表达能力和性能。（多段非线性函数）

class FeedForward(nn.Module):
    def __init__(self, args:SejuMindConfig):
        super().__init__()

    #   init
    #   升维：hidden_size -> intermediate_size
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3) # 2.666倍升维
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.act_fn = ACT2FN[args.hidden_act]
    
    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x) # filter
        return self.dropout(self.down_proj(gated))

class MoEGate(nn.Module):
    def __init__(self, config: SejuMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss
    
class MoEFeedForward(nn.Module):  # ！修正：原MoEFeedForaward拼写错误
    def __init__(self, config: SejuMindConfig):
        super().__init__()
        self.config = config
        # 专家层
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )
        # 门控层
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, h = orig_shape

        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 展开x以便处理
        x = x.view(-1, x.shape[-1])

        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 按照定义的num_experts_per_tok重复输入token
            # 每个token安排num_experts_per_tok个专家处理
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # y是空张量，和x形状相同
            y = torch.empty_like(x, dtype=x.dtype)
            # 遍历所有专家
            for i, expert in enumerate(self.experts):
                # 找到所有指向专家i的token
                # 然后将这些token输入专家i进行处理
                # 最后将结果放回y对应位置
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(
                        p.sum() for p in expert.parameters()
                    )
            # 加权求和
            # 最后的y意义是每个token经过专家处理后的加权结果
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        # 如果是推理阶段
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    # MoE推理方法
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 使用cache，创建一个和x形状相同的零张量
        expert_cache = torch.zeros_like(x)
        # 对专家索引进行排序，最后是[0,0,0,1,1,2,2,2,...]这样的顺序
        # 分拣
        idxs = flat_expert_indices.argsort()
        # 统计每个专家被分配到的token数量
        # 打包
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个token对应的专家索引
        token_idxs = idxs // self.config.num_experts_per_tok
        # 对每个打包好的包进行处理
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前包的起始位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # 取出当前包对应的专家
            expert = self.experts[i]
            # 取出token对应的原始id
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 取出token对应的数据
            expert_tokens = x[exp_token_idx]
            # 计算专家输出，一次性处理当前包的所有token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将结果散点加到缓存中对应位置
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache
    
class SejuMindBlock(nn.Module):
    def __init__(self, layer_id:int, config:SejuMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attention = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = (
            FeedForward(config)
            # if not config.use_moe
            # else MoEFeedForward(config)  # Next step 
        )
    # Transformer layer的前向传播函数
    def forward(self, hidden_states, position_embeddings:Tuple[torch.Tensor, torch.Tensor], 
                past_key_value:Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
                use_cache = False, 
                attention_mask:Optional[torch.Tensor]=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attention(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )

        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value
    


class SejuMindModel(nn.Module):
    def __init__(self, config:SejuMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [SejuMindBlock(l, config) for l in range(self.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None

        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算start_pos：如果存在past，则start_pos为已有past序列长度
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )
        
        # Step 1：Token Embedding + dropout
        # token ids [batch, seq] -> [batch, seq, hidden_size], [1,10] → [1,10,512]
        hidden_states = self.dropout(
            self.embed_tokens(input_ids)
        )  # [bsz, seq_len, hidden]

        # Step 2：计算 start_pos(推理) 
        # 新 token 的位置要从 历史长度开始。
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )

       
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present) #每层的present_key_value都保存在presents列表中，供后续使用（如生成时的缓存）

        hidden_states = self.norm(hidden_states) # 最后的输出经过RMSNorm层进行归一化处理，提升训练稳定性和模型性能 
        aux_loss = sum(
            [
                layer.mlp.aux_loss
                for layer in self.layers
                if isinstance(
                    layer.mlp, MoEFeedForward
                )  # ！修正：原MoEFeedForaward拼写错误
            ],
            hidden_states.new_zeros(1).squeeze(),
        )
        # aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss



class SejuMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = SejuMindConfig

    def __init__(self, config:SejuMindConfig):
        super().__init__(config)
        self.model = SejuMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **args,   
        ):
            hidden_states, past_key_values, aux_loss = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **args,
            )
            
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            output = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
            )
            output.aux_loss = aux_loss

            return output
