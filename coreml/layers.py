from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RMSNorm, Qwen2Model


def rotate_half(x, dim=-1):
    """Rotates half the hidden dims of the input."""
    x1, x2 = torch.chunk(x, 2, dim)
    return torch.cat((-x2, x1), dim=dim)

def apply_rotary_pos_emb(embed, cos, sin, dim=-1):
    embed = (embed * cos) + (rotate_half(embed, dim) * sin)
    return embed

class BaseANEModule(nn.Module):
    target_classes = []

def patch_for_ane(model, wrapper_class=BaseANEModule, target_classes=None):
    if target_classes is None:
        target_classes = wrapper_class.target_classes

    # Wrap all nn.Linear layers with ANELinearWrapper
    for name, module in model.named_children():
        is_wrapper = isinstance(module, wrapper_class) or type(module).__name__ == wrapper_class.__name__  # notebook hack
        for target_class in target_classes:
            if isinstance(module, target_class) or is_wrapper:
                parent = model
                if "." in name:
                    parent_name, attr_name = name.rsplit('.', 1)
                    for part in parent_name.split('.'):
                        if part:
                            parent = getattr(parent, part)
                else:
                    attr_name = name
                
                if is_wrapper:
                    module = module.layer
                ane_linear = wrapper_class(module)
                setattr(parent, attr_name, ane_linear)
                break
        else:
            patch_for_ane(module, wrapper_class, target_classes)

class ANELinearWrapper(nn.Module):
    target_classes = [nn.Conv1d, nn.Linear]

    def __init__(self, layer: nn.Linear | nn.Conv1d):
        super().__init__()
        self.layer = layer
        if isinstance(layer, nn.Linear):
            self.weight = layer.weight[..., None, None]
            self.is_linear = True
        elif isinstance(layer, nn.Conv1d):
            self.weight = layer.weight[..., None, :]
            self.is_linear = False
            self.padding = (0, layer.padding[0])
            self.groups = layer.groups
        self.bias = layer.bias

    def forward(self, x):
        if self.is_linear:
            return F.conv2d(x, self.weight, self.bias)
        else:
            return F.conv2d(x, self.weight, self.bias, padding=self.padding, groups=self.groups)


class WrappedRMSNorm(nn.Module):
    target_classes = [nn.RMSNorm, Qwen2RMSNorm]

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        if layer.weight is not None:
            self.register_buffer(
                "weight",
                torch.cat([
                    layer.weight.view(1, -1, 1, 1),
                    torch.zeros_like(layer.weight).view(1, -1, 1, 1),
                ], dim=1)
            )
            self.register_buffer("bias", torch.zeros_like(self.weight))
        else:
            self.weight = None

        if hasattr(layer, 'eps'):
            self.eps = layer.eps
        else:
            self.eps = layer.variance_epsilon


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.concat([x, -x], dim=1)
        input_dtype = x.dtype
        inputs = x.float()
        channels_mean = inputs.mean(dim=1, keepdims=True)
        zero_mean = inputs - channels_mean
        zero_mean_sq = zero_mean * zero_mean
        denom = (zero_mean_sq.mean(dim=1, keepdims=True) + self.eps).rsqrt()
        out = zero_mean * denom
        if self.weight is not None:
            out = (out + self.bias) * self.weight
        out = torch.chunk(out, 2, dim=1)[0]

        return out.to(input_dtype)
    

class AttentionANEWraperChannelsFirstWithCache(nn.Module):
    target_classes = [Qwen2Attention]
    
    def __init__(self, layer: Qwen2Attention):
        super().__init__()
        self.layer = layer
        self.num_heads = layer.config.num_attention_heads
        self.num_key_value_heads = layer.config.num_key_value_heads
        self.head_dim = layer.head_dim
        self.layer_idx = layer.layer_idx

    def forward(
        self,
        # hidden_states: torch.Tensor,
        # position_emb: Tuple[torch.Tensor, torch.Tensor],
        # position_id: torch.Tensor,
        # kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],

        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Tuple[torch.Tensor, torch.Tensor],
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # self = self.layer
        bsz, _, _, q_len = hidden_states.size()

        
        query_states = self.layer.q_proj(hidden_states)
        key_states = self.layer.k_proj(hidden_states)
        value_states = self.layer.v_proj(hidden_states)

        query_states = query_states.view(bsz, self.num_heads, self.head_dim, q_len) # .transpose(2, 3)
        key_states = key_states.view(bsz, self.num_key_value_heads, self.head_dim, q_len).transpose(2, 3)
        value_states = value_states.view(bsz, self.num_key_value_heads, self.head_dim, q_len).transpose(2, 3)

        cos, sin, cos_t, sin_t = position_embeddings
        query_states = apply_rotary_pos_emb(query_states, cos_t, sin_t, -2)
        # query_states = apply_rotary_pos_emb(query_states, cos, sin, -1)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, -1)
        
        key_cache, value_cache = past_key_value
        key_cache[
            self.layer_idx:self.layer_idx + 1,
            :,
            cache_position:cache_position + q_len,
        ] = key_states
        value_cache[
            self.layer_idx:self.layer_idx + 1,
            :,
            cache_position:cache_position + q_len,
        ] = value_states

        key_states = key_cache[self.layer_idx:self.layer_idx + 1].split(1, dim=1)
        value_states = value_cache[self.layer_idx:self.layer_idx + 1].split(1, dim=1)
        query_states = torch.chunk(query_states, self.num_key_value_heads, dim=1)

        attn_outputs = []
        for q, k, v in zip(query_states, key_states, value_states):
            # attn_output = torch.nn.functional.scaled_dot_product_attention(
            #     q,
            #     k,
            #     v,
            #     attn_mask=attn_mask,
            # )
            attn_weights = k @ q # .transpose(-1, -2) # (batch, 1, kv_length, q_length)
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = attn_weights.softmax(2)
            attn_output = attn_weights.transpose(-1, -2) @ v
            attn_outputs.append(attn_output)
        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.transpose(2, 3).contiguous()
        attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim, 1, q_len)
        attn_output = self.layer.o_proj(attn_output)
        return attn_output, None
    
class LMModelANEWrapperWithCache(nn.Module):
    def __init__(self, layer: Qwen2Model, is_causal=True, cache_length=2048, device="cpu", channels_first=True):
        super().__init__()
        
        self.layer = layer
        self.is_causal = is_causal
        self.cache_length = cache_length
        config = layer.config
        self.channels_first = channels_first
        self.register_buffer("key_cache", torch.zeros(
            config.num_hidden_layers,
            config.num_key_value_heads,
            cache_length,
            # config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels,
            config.hidden_size // config.num_attention_heads,
            device=device,
        ))
        self.register_buffer("value_cache", torch.zeros(
            config.num_hidden_layers,
            config.num_key_value_heads,
            cache_length,
            # config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels,
            config.hidden_size // config.num_attention_heads,
            device=device,
        ))
        cos_emb, sin_emb = self.layer.rotary_emb(torch.ones(1, dtype=torch.float32), torch.arange(cache_length, dtype=torch.long, device=device).unsqueeze(0))
        self.register_buffer("cos_emb", cos_emb[0])
        self.register_buffer("sin_emb", sin_emb[0])
    
    def forward(
        self,
        inputs_embeds,
        position_id,
        # position_ids,
    ):
        position_ids = torch.arange(0, inputs_embeds.size(-1 if self.channels_first else 1), dtype=torch.long, device=inputs_embeds.device).unsqueeze(0) + position_id
        position_ids = position_ids.view(1, -1)
        position_emb = (self.cos_emb[position_ids], self.sin_emb[position_ids])

        attention_mask = (torch.arange(self.cache_length, device=inputs_embeds.device)[None, None, :] > position_ids[..., None]).float()
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.where(attention_mask == 0, -torch.inf)
        if self.channels_first:
            position_emb = (*position_emb, position_emb[0].transpose(-1, -2), position_emb[1].transpose(-1, -2))
            attention_mask = attention_mask.transpose(-1, -2)

        hidden_states = inputs_embeds
        # for decoder_layer in self.layer.layers[:1]:
        for decoder_layer in self.layer.layers:
            hidden_states, = decoder_layer(
                hidden_states,
                attention_mask,
                position_ids,
                (self.key_cache, self.value_cache),
                cache_position=position_id,
                position_embeddings=position_emb,
            )
        hidden_states = self.layer.norm(hidden_states)
        return hidden_states
