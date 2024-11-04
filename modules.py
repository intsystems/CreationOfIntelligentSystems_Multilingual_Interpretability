import torch
from copy import deepcopy
from torch import nn


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2MLP_ND(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.__from_MLP(mlp)
        self.up_proj.impacts = None
        self.down_proj.impacts = None
        self.calculate_impacts = True

    def __from_MLP(self, mlp):
        self.hidden_size = mlp.hidden_size
        self.intermediate_size = mlp.intermediate_size
        self.gate_proj = deepcopy(mlp.gate_proj)
        self.up_proj = deepcopy(mlp.up_proj)
        self.down_proj = deepcopy(mlp.down_proj)
        self.act_fn = deepcopy(mlp.act_fn)

    def forward(self, hidden_state):
        '''Implements the MLP forward and Parallell Neuron Detection
        1. Applies the MLP to the hidden state
        2. Calculates Neurons Impacts = ||(act_fn(W_gate * hidden_state) * W_up * mask) * W_down||_2, 
        where
            mask = np.eye(hidden_state.shape[0])
            W_down = self.down_proj
            W_up = self.up_proj
            W_gate = self.gate_proj
            act_fn = self.act_fn
        return both new hidden state and Neurons Impacts
        '''
        intermediate_state = self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)

        if self.calculate_impacts:
            with torch.no_grad():               
                
                impacts = (torch.sum((intermediate_state ** 2).sum(dim=1).unsqueeze(-1) * (self.up_proj.weight.unsqueeze(0) ** 2), dim=-1) ** 0.5).detach()
                
                if self.up_proj.impacts is None:
                    self.up_proj.impacts = impacts
                else:
                    self.up_proj.impacts = torch.cat((self.up_proj.impacts, impacts), dim=0)
                if self.down_proj.impacts is None:
                    self.down_proj.impacts = impacts
                else:
                    self.down_proj.impacts = torch.cat((self.down_proj.impacts, impacts), dim=0)
                    
        return self.down_proj(intermediate_state)
    

class Qwen2SdpaAttention_ND(nn.Module):
    def __init__(self, sdpa):
        super().__init__()
        self.__from_Sdpa(sdpa)
        self.q_proj.impacts = None
        self.k_proj.impacts = None
        self.calculate_impacts = True

    def __from_Sdpa(self, sdpa):
        self.hidden_size = sdpa.hidden_size
        self.num_heads = sdpa.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = sdpa.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = sdpa.max_position_embeddings
        self.rope_theta = sdpa.rope_theta
        self.is_causal = sdpa.is_causal
        self.attention_dropout = sdpa.attention_dropout
        self.layer_idx = sdpa.layer_idx

        self.q_proj = deepcopy(sdpa.q_proj)
        self.k_proj = deepcopy(sdpa.k_proj)
        self.v_proj = deepcopy(sdpa.v_proj)
        self.o_proj = deepcopy(sdpa.o_proj)

        self.rotary_emb = deepcopy(sdpa.rotary_emb)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position = None,
    ):
        '''Implements the Sdpa Attention forward and Parallell Neuron Detection
        1. Applies the Sdpa Attention to the hidden state
        2. Calculates Neurons Impacts = ||softmax((W_q(x) * W_k(x)^T -Delta(x) / sqrt(head_dim)) - softmax((W_q(x) * W_k(x)^T) / sqrt(head_dim))||_2, 
        where
            Delta(x) = W_Q(x).resize(l, 1, d_mid) * W_K(x).resize(1, l, d_mid)
        '''
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        if self.calculate_impacts:
            with torch.no_grad():
                Delta_x = (query_states.unsqueeze(-2) * key_states.unsqueeze(-3)).detach()

                L, S = query_states.size(-2), key_states.size(-2)
                scale_factor = 1 / query_states.size(-1) ** 0.5
                attn_bias = torch.zeros(L, S, dtype=query_states.dtype).to(query_states.device)
                if is_causal:
                    assert causal_mask is None
                    temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query_states.device)
                    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                    attn_bias.to(query_states.dtype)

                if causal_mask is not None:
                    if causal_mask.dtype == torch.bool:
                        attn_bias.masked_fill_(causal_mask.logical_not(), float("-inf"))
                    else:
                        attn_bias += causal_mask

                attn_weight = query_states @ key_states.transpose(-2, -1) 

                attn_weight_Delta = (attn_weight.unsqueeze(-1) - Delta_x) * scale_factor
                attn_weight = attn_weight * scale_factor

                attn_weight += attn_bias
                attn_weight_Delta += attn_bias.unsqueeze(-1)

                attn_weight = torch.softmax(attn_weight, dim=-1)
                attn_weight_Delta = torch.softmax(attn_weight_Delta, dim=-2)

                attn_weight = torch.dropout(attn_weight, self.attention_dropout if self.training else 0.0, train=True)
                attn_weight_Delta = torch.dropout(attn_weight_Delta, self.attention_dropout if self.training else 0.0, train=True)
                impacts = torch.norm(attn_weight.unsqueeze(-1) - attn_weight_Delta, dim=[-2,-3]).detach()

                if self.q_proj.impacts is None:
                    self.q_proj.impacts = impacts
                else:
                    self.q_proj.impacts = torch.cat((self.q_proj.impacts, impacts), dim=0).detach()
                if self.k_proj.impacts is None:
                    self.k_proj.impacts = impacts
                else:
                    self.k_proj.impacts = torch.cat((self.k_proj.impacts, impacts), dim=0).detach()


        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value