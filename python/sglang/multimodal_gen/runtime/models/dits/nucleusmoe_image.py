# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Nucleus-Image Team. All rights reserved.
#
# Native SGLang implementation of the NucleusMoE-Image transformer.
# Adapted from the HuggingFace reference: NucleusAI/NucleusMoE-Image

import functools
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous

from sglang.multimodal_gen.configs.models.dits.nucleusmoe import NucleusMoEDitConfig
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm, apply_qk_norm
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    apply_flashinfer_rope_qk_inplace,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.models.dits.qwen_image import QwenEmbedRope
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _is_moe_layer(strategy: str, layer_idx: int, num_layers: int) -> bool:
    if strategy == "leave_first_three_and_last_block_dense":
        return layer_idx >= 3 and layer_idx < num_layers - 1
    elif strategy == "leave_first_three_blocks_dense":
        return layer_idx >= 3
    elif strategy == "leave_first_block_dense":
        return layer_idx >= 1
    elif strategy == "all_moe":
        return True
    elif strategy == "all_dense":
        return False
    return True


class NucleusTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, use_additional_t_cond=False):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=embedding_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            scale=1000,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=embedding_dim,
            time_embed_dim=4 * embedding_dim,
            out_dim=embedding_dim,
        )
        self.norm = RMSNorm(embedding_dim, eps=1e-6)
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(self, timestep, hidden_states, addition_t_cond=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=hidden_states.dtype)
        )

        conditioning = timesteps_emb
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError(
                    "When additional_t_cond is True, addition_t_cond must be provided."
                )
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            addition_t_emb = addition_t_emb.to(dtype=hidden_states.dtype)
            conditioning = conditioning + addition_t_emb

        return self.norm(conditioning)


class NucleusMoECrossAttention(nn.Module):
    """Cross-attention for NucleusMoE: image Q attends to [image, text] K/V with GQA."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_key_value_heads: int | None = None,
        joint_attention_dim: int = 3584,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.num_kv_heads = num_key_value_heads or num_attention_heads
        self.inner_kv_dim = self.num_kv_heads * attention_head_dim

        self.to_q = ReplicatedLinear(dim, self.inner_dim, bias=False)
        self.to_k = ReplicatedLinear(dim, self.inner_kv_dim, bias=False)
        self.to_v = ReplicatedLinear(dim, self.inner_kv_dim, bias=False)

        self.add_k_proj = ReplicatedLinear(dim, self.inner_kv_dim, bias=False)
        self.add_v_proj = ReplicatedLinear(dim, self.inner_kv_dim, bias=False)

        self.norm_q = RMSNorm(attention_head_dim, eps=eps)
        self.norm_k = RMSNorm(attention_head_dim, eps=eps)
        self.norm_added_k = RMSNorm(attention_head_dim, eps=eps)

        self.to_out = ReplicatedLinear(self.inner_dim, dim, bias=False)

        self.attn = USPAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        batch_size, img_seq_len, _ = hidden_states.shape
        txt_seq_len = encoder_hidden_states.shape[1]

        img_query, _ = self.to_q(hidden_states)
        img_key, _ = self.to_k(hidden_states)
        img_value, _ = self.to_v(hidden_states)

        txt_key, _ = self.add_k_proj(encoder_hidden_states)
        txt_value, _ = self.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (self.num_heads, self.head_dim))
        img_key = img_key.unflatten(-1, (self.num_kv_heads, self.head_dim))
        img_value = img_value.unflatten(-1, (self.num_kv_heads, self.head_dim))

        txt_key = txt_key.unflatten(-1, (self.num_kv_heads, self.head_dim))
        txt_value = txt_value.unflatten(-1, (self.num_kv_heads, self.head_dim))

        img_query = apply_qk_norm(self.norm_q, img_query)
        img_key = apply_qk_norm(self.norm_k, img_key)
        txt_key = apply_qk_norm(self.norm_added_k, txt_key)

        if image_rotary_emb is not None:
            img_cos_sin_cache, txt_cos_sin_cache = image_rotary_emb
            img_query, img_key = apply_flashinfer_rope_qk_inplace(
                img_query, img_key, img_cos_sin_cache
            )
            txt_key_for_rope = txt_key.clone()
            _, txt_key = apply_flashinfer_rope_qk_inplace(
                txt_key_for_rope, txt_key, txt_cos_sin_cache
            )

        joint_key = torch.cat([img_key, txt_key], dim=1)
        joint_value = torch.cat([img_value, txt_value], dim=1)

        attn_output = self.attn(img_query, joint_key, joint_value)
        attn_output = attn_output.reshape(batch_size, img_seq_len, self.inner_dim)

        output, _ = self.to_out(attn_output)
        return output


class NucleusMoELayer(nn.Module):
    """Mixture-of-Experts layer with expert-choice routing and a shared expert.

    Each expert is a separate FeedForward module. The router uses a timestep-
    conditioned gating mechanism. A shared expert processes all tokens and its
    output is combined with the routed expert outputs via scatter-add.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_dim: int,
        num_experts: int,
        capacity_factor: float,
        use_sigmoid: bool,
        route_scale: float,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.use_sigmoid = use_sigmoid
        self.route_scale = route_scale

        self.gate = nn.Linear(hidden_size * 2, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                FeedForward(
                    dim=hidden_size,
                    dim_out=hidden_size,
                    inner_dim=moe_intermediate_dim,
                    activation_fn="swiglu",
                    bias=False,
                )
                for _ in range(num_experts)
            ]
        )
        self.shared_expert = FeedForward(
            dim=hidden_size,
            dim_out=hidden_size,
            inner_dim=moe_intermediate_dim,
            activation_fn="swiglu",
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_unmodulated: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, slen, dim = hidden_states.shape

        if timestep is not None:
            timestep_expanded = timestep.unsqueeze(1).expand(-1, slen, -1)
            router_input = torch.cat(
                [timestep_expanded, hidden_states_unmodulated], dim=-1
            )
        else:
            router_input = hidden_states_unmodulated

        logits = self.gate(router_input)

        if self.use_sigmoid:
            scores = torch.sigmoid(logits.float()).to(logits.dtype)
        else:
            scores = F.softmax(logits.float(), dim=-1).to(logits.dtype)

        affinity = scores.transpose(1, 2)  # (B, E, S)
        capacity = max(1, math.ceil(self.capacity_factor * slen / self.num_experts))

        topk = torch.topk(affinity, k=capacity, dim=-1)
        top_indices = topk.indices  # (B, E, C)
        gating = affinity.gather(dim=-1, index=top_indices)  # (B, E, C)

        batch_offsets = (
            torch.arange(bs, device=hidden_states.device, dtype=torch.long).view(
                bs, 1, 1
            )
            * slen
        )
        global_token_indices = (
            (batch_offsets + top_indices)
            .transpose(0, 1)
            .reshape(self.num_experts, -1)
            .reshape(-1)
        )
        gating_flat = (
            gating.transpose(0, 1).reshape(self.num_experts, -1).reshape(-1)
        )

        token_score_sums = torch.zeros(
            bs * slen, device=hidden_states.device, dtype=gating_flat.dtype
        )
        token_score_sums.scatter_add_(0, global_token_indices, gating_flat)
        gating_flat = gating_flat / (token_score_sums[global_token_indices] + 1e-12)
        gating_flat = gating_flat * self.route_scale

        x_flat = hidden_states.reshape(bs * slen, dim)
        routed_input = x_flat[global_token_indices]

        tokens_per_expert = bs * capacity
        routed_output_parts = []
        for i, expert in enumerate(self.experts):
            start = i * tokens_per_expert
            end = start + tokens_per_expert
            expert_out = expert(routed_input[start:end])
            routed_output_parts.append(expert_out)

        routed_output = torch.cat(routed_output_parts, dim=0)
        routed_output = (routed_output.float() * gating_flat.unsqueeze(-1)).to(
            hidden_states.dtype
        )

        out = self.shared_expert(hidden_states).reshape(bs * slen, dim)

        scatter_idx = global_token_indices.reshape(-1, 1).expand(-1, dim)
        out = out.scatter_add(dim=0, index=scatter_idx, src=routed_output)
        out = out.reshape(bs, slen, dim)

        return out


class NucleusMoEImageTransformerBlock(nn.Module):
    """Single-stream DiT block with optional MoE MLP and cross-attention to text."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_key_value_heads: int | None = None,
        joint_attention_dim: int = 3584,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        mlp_ratio: float = 4.0,
        moe_enabled: bool = False,
        num_experts: int = 128,
        moe_intermediate_dim: int = 1344,
        capacity_factor: float = 8.0,
        use_sigmoid: bool = False,
        route_scale: float = 2.5,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dim = dim
        self.moe_enabled = moe_enabled

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 4 * dim, bias=True),
        )

        self.encoder_proj = nn.Linear(joint_attention_dim, dim)

        self.pre_attn_norm = nn.LayerNorm(
            dim, eps=eps, elementwise_affine=False, bias=False
        )
        self.attn = NucleusMoECrossAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_key_value_heads=num_key_value_heads,
            joint_attention_dim=dim,
            qk_norm=qk_norm,
            eps=eps,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        self.pre_mlp_norm = nn.LayerNorm(
            dim, eps=eps, elementwise_affine=False, bias=False
        )

        if moe_enabled:
            self.img_mlp = NucleusMoELayer(
                hidden_size=dim,
                moe_intermediate_dim=moe_intermediate_dim,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
                use_sigmoid=use_sigmoid,
                route_scale=route_scale,
            )
        else:
            mlp_inner_dim = int(dim * mlp_ratio * 2 / 3) // 128 * 128
            self.img_mlp = FeedForward(
                dim=dim,
                dim_out=dim,
                inner_dim=mlp_inner_dim,
                activation_fn="swiglu",
                bias=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        scale1, gate1, scale2, gate2 = (
            self.img_mod(temb).unsqueeze(1).chunk(4, dim=-1)
        )
        scale1, scale2 = 1 + scale1, 1 + scale2

        gate1 = gate1.clamp(min=-2.0, max=2.0)
        gate2 = gate2.clamp(min=-2.0, max=2.0)

        context = self.encoder_proj(encoder_hidden_states)

        img_normed = self.pre_attn_norm(hidden_states)
        img_modulated = img_normed * scale1

        img_attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=context,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate1.tanh() * img_attn_output

        img_normed2 = self.pre_mlp_norm(hidden_states)
        img_modulated2 = img_normed2 * scale2

        if self.moe_enabled:
            img_mlp_output = self.img_mlp(img_modulated2, img_normed2, timestep=temb)
        else:
            img_mlp_output = self.img_mlp(img_modulated2)

        hidden_states = hidden_states + gate2.tanh() * img_mlp_output

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class NucleusMoEImageTransformer2DModel(OffloadableDiTMixin, CachableDiT):
    """SGLang-native NucleusMoE Transformer for image generation.

    Single-stream DiT with cross-attention to text and optional MoE feed-forward
    layers. Uses expert-choice routing with a shared expert.
    """

    _non_quantizable_patterns = {
        "fp8_e4m3fn": [
            "img_mod",
            "attn.to_out",
        ],
        "awq_w4a16": [
            "img_mod",
        ],
    }

    def __init__(
        self,
        config: NucleusMoEDitConfig,
        hf_config: dict[str, Any],
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(config=config, hf_config=hf_config)
        arch = config.arch_config
        patch_size = arch.patch_size
        in_channels = arch.in_channels
        out_channels = arch.out_channels
        num_layers = arch.num_layers
        attention_head_dim = arch.attention_head_dim
        num_attention_heads = arch.num_attention_heads
        num_key_value_heads = getattr(arch, "num_key_value_heads", None)
        joint_attention_dim = arch.joint_attention_dim
        axes_dims_rope = arch.axes_dims_rope
        mlp_ratio = arch.mlp_ratio

        moe_enabled = arch.moe_enabled
        dense_moe_strategy = arch.dense_moe_strategy
        num_experts = arch.num_experts
        moe_intermediate_dim = arch.moe_intermediate_dim
        capacity_factor = arch.capacity_factor
        use_sigmoid = arch.use_sigmoid
        route_scale = arch.route_scale

        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.rotary_emb = QwenEmbedRope(
            theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True
        )

        self.time_text_embed = NucleusTimestepProjEmbeddings(
            embedding_dim=self.inner_dim
        )

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)
        self.img_in = nn.Linear(in_channels, self.inner_dim)

        # Per-layer capacity factors from HF config, fallback to uniform
        capacity_factors = hf_config.get(
            "capacity_factors", [capacity_factor] * num_layers
        )

        self.transformer_blocks = nn.ModuleList(
            [
                NucleusMoEImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    num_key_value_heads=num_key_value_heads,
                    joint_attention_dim=joint_attention_dim,
                    mlp_ratio=mlp_ratio,
                    moe_enabled=moe_enabled
                    and _is_moe_layer(dense_moe_strategy, idx, num_layers),
                    num_experts=num_experts,
                    moe_intermediate_dim=moe_intermediate_dim,
                    capacity_factor=capacity_factors[idx],
                    use_sigmoid=use_sigmoid,
                    route_scale=route_scale,
                    quant_config=quant_config,
                    prefix=f"transformer_blocks.{idx}",
                )
                for idx in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=False
        )

        self.layer_names = ["transformer_blocks"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] = None,
        additional_t_cond: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]

        hidden_states = self.img_in(hidden_states)
        timestep = (timestep / 1000).to(hidden_states.dtype)

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)

        temb = self.time_text_embed(timestep, hidden_states, additional_t_cond)

        image_rotary_emb = freqs_cis
        for index_block, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        return output


EntryClass = NucleusMoEImageTransformer2DModel
